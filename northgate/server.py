"""Northgate's repair-estimate agent: a plain A2A SDK executor, no LangGraph anywhere.

The wire is the same one Risk's LangGraph executor speaks. Nothing in a caller's
RemoteAgentNode can tell which runtime is behind a card, which is the point.

Northgate serves more than one insurer. Its estimate history is kept per insurer (per
tenant), and since Chapter 6 the tenant comes from the verified delegation chain's root,
never from the request. `verifies_chains=False` deploys Northgate as it was before: any
valid Meridian service token from any Meridian workload is accepted, and the tenant is
whatever insurer the request names.

Northgate publishes its seam terms on its card (Chapter 4), as revisions with a date.
Release v8 prices parts per model and so requires the vehicle's VIN; a v7 deployment
may announce v8's terms ahead of the release. The executor enforces the terms it is
deployed with: a request missing a required input is rejected.

Since Chapter 5, Northgate signs its card (JWS over the card's canonical JSON, Ed25519)
with a key whose public half Meridian pinned at onboarding, by key id. It also serves
that key at the URL the signature's header names, which a careful caller never needs.
`books=True` deploys a later card that adds a `book_repair` skill no insurer approved.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.types import ASGIApp, Receive, Scope, Send

from meridian.agents import PART_PRICES, estimate_repair
from meridian.identity import MERIDIAN
from meridian.incident import introspect
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.delegation import Guard, Trust, Verified, peer_of
from seamkit.cards import jwk, sign_card
from seamkit.server import a2a_app, agent_card, key_set_route

NORTHGATE_URL = "http://northgate.example.test/"

# Northgate's card-signing key. The private half never leaves Northgate; Meridian pinned
# the public half by its key id when the partner agreement was signed. The tests play
# both companies, so both halves are here.
NORTHGATE_KID = "northgate-cards-2026-06"
NORTHGATE_SIGNING_KEY = Ed25519PrivateKey.generate()
NORTHGATE_PUBLIC_JWK = jwk(NORTHGATE_SIGNING_KEY.public_key())
KEY_SET_PATH = "/.well-known/jwks.json"

# Northgate's terms. Northgate owns them; Meridian's code only reads them off the card.
ESTIMATE_REPAIR = SkillTerms(
    inputs=(Field("claim_id", "string"), Field("damaged_parts", "string[]")),
    outputs=(Field("estimate", "number", unit="claim currency"),),
    asks=("photo_url",),  # frame damage is never estimated unseen
    p95_seconds=4.0, cost_per_task=3.5)
ESTIMATE_HISTORY = SkillTerms(
    inputs=(Field("claim_id", "string", required=False),
            Field("shop", "string", required=False)),
    outputs=(Field("shop", "string"), Field("estimates", "object[]")),
    p95_seconds=2.0, cost_per_task=0.5)
NORTHGATE_V7 = SeamTerms(
    "v7", date(2026, 6, 1),
    {"estimate_repair": ESTIMATE_REPAIR, "estimate_history": ESTIMATE_HISTORY},
    auth=("mtls", "delegation"), skew_seconds=30,
    accepts=frozenset({"claim", "vehicle"}))
# v8 prices parts per model, so it needs the VIN: one new required input.
NORTHGATE_V8 = replace(
    NORTHGATE_V7, revision="v8", effective=date(2026, 11, 2),
    skills={**NORTHGATE_V7.skills, "estimate_repair": replace(
        ESTIMATE_REPAIR, inputs=(*ESTIMATE_REPAIR.inputs, Field("vin", "string")))})
TERMS = {"v7": NORTHGATE_V7, "v8": NORTHGATE_V8}

# Another insurer on Northgate's network, with its own root issuer and keys. The private
# halves would live with that insurer; the tests play it, so they are kept here.
INSURER_B_ISSUER = "insurer-b-idp"
INSURER_B_KEYS = {owner: Ed25519PrivateKey.generate()
                  for owner in (INSURER_B_ISSUER, "insurer-b")}
INSURER_B = Trust("insurer-b", INSURER_B_ISSUER,
                  {o: k.public_key() for o, k in INSURER_B_KEYS.items()})

# The issuers Northgate pinned out of band, one per insurer it serves.
TRUSTS: Mapping[str, Trust] = {MERIDIAN.root: MERIDIAN, INSURER_B.root: INSURER_B}

# The skills each insurer's agreement covers. A chain says what the insurer authorised;
# the agreement says what Northgate will do for that insurer at all.
AGREEMENTS = {"meridian": {"estimate_repair", "estimate_history"},
              "insurer-b": {"estimate_repair", "estimate_history"}}

# Repairs booked on Northgate's network: which shop has the car, and which parts.
BOOKINGS: dict[str, dict[str, tuple[str, list[str]]]] = {
    "meridian": {"CLM-1001": ("SHOP-3", ["bumper", "door"]),
                 "CLM-1002": ("SHOP-7", ["frame", "door"])},
    "insurer-b": {"B-2201": ("SHOP-7", ["frame"])},
}

# Every estimate a shop on the network has written this quarter, per insurer:
# (shop, claim, part, amount).
ESTIMATES: dict[str, list[tuple[str, str, str, float]]] = {
    "meridian": [
        ("SHOP-7", "CLM-0871", "frame", 6_100.0),
        ("SHOP-7", "CLM-0871", "door", 1_700.0),
        ("SHOP-7", "CLM-0912", "frame", 6_500.0),
        ("SHOP-7", "CLM-0931", "hood", 900.0),
        ("SHOP-7", "CLM-0958", "bumper", 1_150.0),
        ("SHOP-7", "CLM-0966", "wing", 750.0),
        ("SHOP-7", "CLM-0990", "door", 1_800.0),
        ("SHOP-7", "CLM-0994", "frame", 7_200.0),
        ("SHOP-3", "CLM-0902", "bumper", 1_200.0),
        ("SHOP-3", "CLM-0950", "door", 1_950.0),
    ],
    "insurer-b": [
        ("SHOP-7", "B-2107", "frame", 6_900.0),
        ("SHOP-7", "B-2150", "door", 1_650.0),
        ("SHOP-7", "B-2188", "bumper", 1_300.0),
    ],
}


def estimate_history(tenant: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """A shop's past estimates, inside one insurer's data.

    With a claim id, the history of the shop repairing that claim, for that claim's
    parts only. With a shop id, the shop's whole book for the quarter: every estimate,
    every part, every claim, which Northgate's own network analytics use.
    """
    book = ESTIMATES.get(tenant, [])
    if "shop" in request:
        shop = str(request["shop"])
        rows = [{"claim_id": c, "part": p, "amount": a}
                for s, c, p, a in book if s == shop]
        return {"shop": shop, "estimates": rows}
    shop, parts = BOOKINGS.get(tenant, {}).get(str(request.get("claim_id")), ("", []))
    rows = [{"part": p, "amount": a} for s, _, p, a in book if s == shop and p in parts]
    return {"shop": shop, "estimates": rows}


class NorthgateExecutor(AgentExecutor):
    def __init__(self, verifies_chains: bool = True,
                 terms: SeamTerms = NORTHGATE_V7) -> None:
        self.verifies_chains = verifies_chains
        self.terms = terms
        self.received: list[dict[str, Any]] = []
        self.estimates: dict[str, float] = {}

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        request = get_data_parts(context.message.parts)[0]
        skill = str(context.message.metadata["skill"])
        caller = context.call_context.state.get("auth")
        tenant = self._tenant(caller, request)
        self.received.append({"skill": skill, "request": request, "tenant": tenant,
                              "context_id": context.context_id, **_who(caller)})
        if skill not in AGREEMENTS.get(tenant, set()) or self._lacks(skill, request):
            await updater.reject()
            return
        if skill == "estimate_history":
            await updater.add_artifact(
                [new_data_part(estimate_history(tenant, request))],
                name="estimate_history")
            await updater.complete()
            return
        if not set(request["damaged_parts"]) <= set(PART_PRICES):
            await updater.reject()
            return
        if "frame" in request["damaged_parts"] and "photo_url" not in request:
            # Structural damage is never estimated unseen.
            await updater.requires_input(updater.new_agent_message(
                [new_data_part({"needs": "photo_url",
                                "why": "frame damage needs a photo"})]))
            return
        estimate = estimate_repair(request)
        self.estimates[request["claim_id"]] = estimate
        await updater.add_artifact([new_data_part({"estimate": estimate})],
                                   name="repair_estimate")
        await updater.complete()

    def _lacks(self, skill: str, request: Mapping[str, Any]) -> bool:
        """A required input of the terms this release runs under is missing."""
        terms = self.terms.skills.get(skill)
        return terms is None or any(
            f.required and f.name not in request for f in terms.inputs)

    def _tenant(self, caller: Any, request: Mapping[str, Any]) -> str:
        if isinstance(caller, Verified):
            return caller.tenant  # from the pinned root, never from the message
        # Before Chapter 6: an insurer named in the request wins, for brokers who act
        # for several; otherwise the insurer behind the service token.
        return str(request.get("insurer") or "meridian")

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()


def _who(caller: Any) -> dict[str, Any]:
    if isinstance(caller, Verified):
        return {"caller": caller.path[-1], "chain": caller.chain_id,
                "subject": caller.subject, "path": caller.path}
    if isinstance(caller, LegacyCaller):
        return {"caller": caller.owner, "chain": "", "subject": "", "path": ()}
    return {"caller": "", "chain": "", "subject": "", "path": ()}


def northgate_card(release: str, revisions: Sequence[SeamTerms],
                   books: bool = False) -> AgentCard:
    """Northgate's card for one release, with its terms, before it is signed."""
    needs = "; needs the vehicle's VIN" if release == "v8" else ""
    repair = f"List-price estimate for a claim's damaged parts{needs}"
    skills = {"estimate_repair": repair,
              "estimate_history": "Past estimates by the shop repairing a claim"}
    if books:
        skills["book_repair"] = "Book the repair at a network shop against the estimate"
    return agent_card("Northgate repair estimates", NORTHGATE_URL, release, skills,
                      extensions=[publish(*revisions)] if revisions else [])


@dataclass(frozen=True)
class LegacyCaller:
    owner: str  # the owner of the service token presented
    workload: str  # the Meridian workload that presented it


MERIDIAN_WORKLOADS = {"claims", "risk", "finance"}


@dataclass
class TokenCheck:
    """Northgate before Chapter 6: mTLS says Meridian workload, the bearer is valid."""

    app: ASGIApp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        token = headers.get(b"authorization", b"").decode().removeprefix("Bearer ")
        owner, workload = introspect(token), peer_of(scope)
        if not owner or workload not in MERIDIAN_WORKLOADS:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await self.app({**scope, "auth": LegacyCaller(owner, workload)}, receive, send)


@dataclass
class NorthgatePartner:
    """Northgate's deployment.

    `announces` puts a later revision of the terms on the card beside the one in force.
    `publishes=False` deploys the card as it was before Chapter 4: skills and a
    description, and no terms.
    """

    release: str = "v7"
    verifies_chains: bool = True
    announces: SeamTerms | None = None
    publishes: bool = True
    signs: bool = True  # since Chapter 5; False deploys the unsigned card of Chapter 4
    books: bool = False  # a card that also offers book_repair: capability inflation
    executor: NorthgateExecutor = field(init=False)

    def __post_init__(self) -> None:
        terms = TERMS[self.release]
        self.executor = NorthgateExecutor(self.verifies_chains, terms)
        revisions = [terms, *([self.announces] if self.announces else [])]
        self.card = northgate_card(self.release, revisions if self.publishes else [],
                                   self.books)
        if self.signs:
            sign_card(self.card, NORTHGATE_SIGNING_KEY, NORTHGATE_KID,
                      NORTHGATE_URL.rstrip("/") + KEY_SET_PATH)
        inner = a2a_app(self.card, self.executor, [
            key_set_route(KEY_SET_PATH,
                          {NORTHGATE_KID: NORTHGATE_SIGNING_KEY.public_key()})])
        self.guard = Guard(inner, "northgate", TRUSTS)
        self.app: ASGIApp = self.guard if self.verifies_chains else TokenCheck(inner)

