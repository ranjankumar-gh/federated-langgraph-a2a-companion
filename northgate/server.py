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

Since Chapter 8, terms v7.2 add `notes` to the estimate: whatever the repairing shop typed
into its estimate form, passed through. Northgate is trusted; the shop's text is not
Northgate's.
"""

import asyncio
import time

import httpx
from collections.abc import Callable, Mapping, Sequence
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
from meridian.identity import MERIDIAN, revocation_sources
from meridian.incident import introspect
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.delegation import Guard, Trust, Verified, peer_of
from seamkit.cards import jwk, sign_card
from seamkit.revocation import ActPoint, RevocationSource, checker, still_holds
from seamkit.push import doorbell
from seamkit.server import a2a_app, agent_card, key_set_route, push_sender

NORTHGATE_URL = "http://northgate.example.test/"

# Northgate's card-signing key. The private half never leaves Northgate; Meridian pinned
# the public half by its key id when the partner agreement was signed. The tests play
# both companies, so both halves are here.
NORTHGATE_KID = "northgate-cards-2026-06"
NORTHGATE_SIGNING_KEY = Ed25519PrivateKey.generate()
NORTHGATE_PUBLIC_JWK = jwk(NORTHGATE_SIGNING_KEY.public_key())
KEY_SET_PATH = "/.well-known/jwks.json"
DOORBELL_PATH = "/revocations"  # Chapter 11: Meridian's issuer rings it after a list

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
# Chapter 7: a dated revision adds one term, the longest Northgate will act under a
# revoked Meridian chain. Its release (the card's version) is still v7.
NORTHGATE_V7_1 = replace(NORTHGATE_V7, revision="v7.1", effective=date(2026, 10, 17),
                         revocation_seconds=90)
# Chapter 8: a dated revision adds one output, the repairing shop's notes, passed
# through from the shop's estimate form. It is a string, and it may be empty.
ESTIMATE_REPAIR_NOTES = replace(ESTIMATE_REPAIR, outputs=(
    *ESTIMATE_REPAIR.outputs, Field("notes", "string")))
NORTHGATE_V7_2 = replace(NORTHGATE_V7_1, revision="v7.2", effective=date(2026, 10, 19),
                         skills={**NORTHGATE_V7.skills,
                                 "estimate_repair": ESTIMATE_REPAIR_NOTES})
# v8 prices parts per model, so it needs the VIN: one new required input.
NORTHGATE_V8 = replace(
    NORTHGATE_V7_2, revision="v8", effective=date(2026, 11, 2),
    skills={**NORTHGATE_V7.skills, "estimate_repair": replace(
        ESTIMATE_REPAIR_NOTES,
        inputs=(*ESTIMATE_REPAIR.inputs, Field("vin", "string")))})
# Chapter 12: a dated revision promises a key window on both skills. A repeat of a key
# inside an hour gets the task the first send opened; Northgate bills each task.
NORTHGATE_V7_3 = replace(
    NORTHGATE_V7_2, revision="v7.3", effective=date(2026, 12, 21),
    skills={"estimate_repair": replace(ESTIMATE_REPAIR_NOTES, key_seconds=3600),
            "estimate_history": replace(ESTIMATE_HISTORY, key_seconds=3600)})
TERMS = {"v7": NORTHGATE_V7_2, "v7.3": NORTHGATE_V7_3,
         "v8": NORTHGATE_V8}  # the revision each release runs

# What each repairing shop typed into its estimate form, by claim. Northgate passes the
# notes through as they were written; it did not write them.
SHOP_NOTES: dict[str, str] = {
    "CLM-1001": "Bumper and front door replaced. Door hinge pins worn; do not refit "
                "the old bracket.",
}

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


def estimate_history(tenant: str, request: Mapping[str, Any],
                     remarks: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A shop's past estimates, inside one insurer's data.

    With a claim id, the history of the shop repairing that claim, for that claim's
    parts only. With a shop id, the shop's whole book for the quarter: every estimate,
    every part, every claim, which Northgate's own network analytics use.
    `remarks` is free text a shop attached to its rows (Chapter 8), passed through.
    """
    book = ESTIMATES.get(tenant, [])
    if "shop" in request:
        shop = str(request["shop"])
        rows: list[dict[str, Any]] = [{"claim_id": c, "part": p, "amount": a}
                                      for s, c, p, a in book if s == shop]
    else:
        shop, parts = BOOKINGS.get(tenant, {}).get(str(request.get("claim_id")),
                                                    ("", []))
        rows = [{"part": p, "amount": a} for s, _, p, a in book
                if s == shop and p in parts]
    if remark := (remarks or {}).get(shop):
        rows = [{**row, "remark": remark} for row in rows]
    return {"shop": shop, "estimates": rows}


class NorthgateExecutor(AgentExecutor):
    def __init__(self, verifies_chains: bool = True,
                 terms: SeamTerms = NORTHGATE_V7_1,
                 assessors: asyncio.Event | None = None,
                 revocations: Mapping[str, RevocationSource] | None = None,
                 now: Callable[[], float] = time.time,
                 rechecks: bool = True,
                 notes: Mapping[str, str] | None = None,
                 remarks: Mapping[str, str] | None = None) -> None:
        self.verifies_chains = verifies_chains
        self.terms = terms
        self.assessors = assessors  # set: the assessor desk has finished its reviews
        self.revocations = revocations or {}
        self.now = now
        self.rechecks = rechecks  # False: Chapter 6's Northgate, checked on arrival only
        self.notes = SHOP_NOTES if notes is None else notes  # by claim, shop-written
        self.remarks = remarks or {}  # by shop, on estimate_history rows
        self.received: list[dict[str, Any]] = []
        self.estimates: dict[str, float] = {}
        self.billed: dict[str, float] = {}  # task -> what Meridian is charged for it
        self.pending: dict[str, dict[str, Any]] = {}  # estimates waiting on authority
        self.acts: list[dict[str, Any]] = []  # every act point: when, which task, outcome
        self.canceled: list[str] = []  # tasks that ended canceled, never billed

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        caller = context.call_context.state.get("auth")
        if context.task_id in self.pending:  # a renewal: the resume act point
            await self._commit(updater, context.task_id, caller, ActPoint.RESUME)
            return
        request = get_data_parts(context.message.parts)[0]
        skill = str(context.message.metadata["skill"])
        tenant = self._tenant(caller, request)
        self.received.append({"skill": skill, "request": request, "tenant": tenant,
                              "context_id": context.context_id, **_who(caller)})
        if skill not in AGREEMENTS.get(tenant, set()) or self._lacks(skill, request):
            await updater.reject()
            return
        if skill == "estimate_history":
            await updater.add_artifact(
                [new_data_part(estimate_history(tenant, request, self.remarks))],
                name="estimate_history")
            await updater.complete()
            return
        if not set(request["damaged_parts"]) <= set(PART_PRICES):
            await updater.reject()
            return
        frame = "frame" in request["damaged_parts"]
        if frame and self.assessors is not None:
            # Chapter 7: an assessor reviews frame damage, with the photo the shop
            # sends Northgate directly; it takes about twenty minutes.
            await updater.start_work()
            await self.assessors.wait()
        elif frame and "photo_url" not in request:
            # Structural damage is never estimated unseen.
            await updater.requires_input(updater.new_agent_message(
                [new_data_part({"needs": "photo_url",
                                "why": "frame damage needs a photo"})]))
            return
        self.pending[context.task_id] = {"claim_id": request["claim_id"],
                                         "estimate": estimate_repair(request)}
        if "notes" in {f.name for f in self.terms.skills["estimate_repair"].outputs}:
            self.pending[context.task_id]["notes"] = self.notes.get(
                str(request["claim_id"]), "")
        await self._commit(updater, context.task_id, caller, ActPoint.COMMIT)

    async def _commit(self, updater: TaskUpdater, task_id: str, caller: Any,
                      point: ActPoint) -> None:
        """The act point before a result leaves Northgate: does the chain still hold?"""
        holds = (still_holds(caller, self.revocations, self.now())
                 if self.rechecks and isinstance(caller, Verified) else "ok")
        self.acts.append({"point": point, "task_id": task_id, "at": self.now(),
                          "claim_id": self.pending[task_id]["claim_id"],
                          "holds": holds, **_who(caller)})
        if holds == "revoked":  # final: cancelled, never billed, never resumed
            self.pending.pop(task_id)
            self.canceled.append(task_id)
            await updater.cancel(updater.new_agent_message(
                [new_data_part({"why": "delegation revoked"})]))
            return
        if holds != "ok":  # expired, or a list too stale to check: renewable
            await updater.requires_auth(updater.new_agent_message(
                [new_data_part({"needs": "delegation", "why": f"chain {holds}"})]))
            return
        result = self.pending.pop(task_id)
        self.estimates[result["claim_id"]] = result["estimate"]
        answer = {k: v for k, v in result.items() if k in ("estimate", "notes")}
        await updater.add_artifact([new_data_part(answer)], name="repair_estimate")
        await updater.complete()
        self.billed[task_id] = self.terms.skills["estimate_repair"].cost_per_task or 0.0

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
        self.pending.pop(context.task_id, None)  # a cancelled estimate is never billed
        self.canceled.append(context.task_id)
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
                      extensions=[publish(*revisions)] if revisions else [], push=True)


@dataclass
class Doorway:
    """Routes a fixed path to its own app, outside the guard; everything else to `app`.

    A doorbell carries no chain, and needs none: it only makes Northgate pull a list
    from the issuer's fixed URL, checked with the key Northgate already pinned.
    """

    app: ASGIApp
    doors: Mapping[str, ASGIApp]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        http = scope["type"] == "http"
        door = self.doors.get(scope.get("path", "")) if http else None
        if door is None:
            await self.app(scope, receive, send)
            return
        await door({**scope, "path": "/", "raw_path": b"/"}, receive, send)


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
    # Chapter 7: the assessor desk for frame damage, Meridian's revocation list (pulled
    # from Meridian's URL), the clock, and whether act points re-check the chain.
    assessors: asyncio.Event | None = None
    revocations: Mapping[str, RevocationSource] | None = None
    now: Callable[[], float] = time.time
    rechecks: bool = True
    # Chapter 8: shop notes by claim (None is SHOP_NOTES), and shop remarks on history
    notes: Mapping[str, str] | None = None
    remarks: Mapping[str, str] | None = None
    # Chapter 11: the client the SDK's push sender posts with (None: nobody is rung),
    # and Meridian's revocation doorbell, mounted beside the A2A endpoint
    push: httpx.AsyncClient | None = None
    executor: NorthgateExecutor = field(init=False)

    def __post_init__(self) -> None:
        terms = TERMS[self.release]
        sources = (revocation_sources() if self.revocations is None
                   else self.revocations) if self.rechecks else {}
        self.executor = NorthgateExecutor(self.verifies_chains, terms, self.assessors,
                                          sources, self.now, self.rechecks, self.notes,
                                          self.remarks)
        revisions = [terms, *([self.announces] if self.announces else [])]
        self.card = northgate_card(self.release, revisions if self.publishes else [],
                                   self.books)
        if self.signs:
            sign_card(self.card, NORTHGATE_SIGNING_KEY, NORTHGATE_KID,
                      NORTHGATE_URL.rstrip("/") + KEY_SET_PATH)
        inner = a2a_app(self.card, self.executor, [
            key_set_route(KEY_SET_PATH,
                          {NORTHGATE_KID: NORTHGATE_SIGNING_KEY.public_key()})],
            push=push_sender(self.push) if self.push else None, now=self.now)
        self.guard = Guard(inner, "northgate", TRUSTS, now=self.now,
                           revoked=checker(sources) if sources else None)
        front: ASGIApp = self.guard if self.verifies_chains else TokenCheck(inner)
        self.sources = sources
        self.app: ASGIApp = Doorway(front, {DOORBELL_PATH: doorbell(self.ring)})

    def ring(self) -> None:
        """Meridian's issuer signed a new list: pull it now, through the same checks."""
        for source in self.sources.values():
            source.ring(self.now())

