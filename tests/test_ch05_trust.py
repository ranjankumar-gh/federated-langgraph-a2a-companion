"""The card trust policy: which fetched cards Claims may send data and authority to."""

from dataclasses import replace

import pytest
from a2a.types import TaskState
from a2a.utils.signing import _canonicalize_agent_card
from starlette.types import ASGIApp

from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import (
    EXPECTATIONS,
    OWNERS_A2A,
    SEAMS,
    A2APartners,
    build_a2a_graph,
    connect_partners,
    in_process_http,
    with_authority,
    with_trust,
)
from meridian.inprocess import ClaimState
from meridian.lookalike import (
    LOOKALIKE_URL,
    PROVIDER,
    LookAlike,
    registry_with_lookalike,
    verify_with_card_key,
)
from meridian.risk_server import RiskPartner
from meridian.samples import clean_claim
from meridian.trust import MERIDIAN_TRUST, NORTHGATE
from northgate.server import KEY_SET_PATH, NORTHGATE_URL, NorthgatePartner
from seamkit.cards import canonical, card_digest
from seamkit.errors import CardUntrusted, MalformedAnswer
from seamkit.inventory import inventory_from_graph
from seamkit.remote import RemoteAgentNode, connect, refresh
from seamkit.tools import remote_skill_tool
from seamkit.trust import decide
from tests.test_ch03_mapping import task
from tests.wire import Federation, thread

pytestmark = pytest.mark.anyio

ESTIMATE = EXPECTATIONS["repair_estimate"]


def sideswipe() -> ClaimState:
    """A version of CLM-1001 with both bumpers and two doors: a 6,000 estimate."""
    return {**clean_claim(), "amount": 6_000.0,
            "damaged_parts": ["bumper", "bumper", "door", "door"]}


# The card digest ----------------------------------------------------------------------

def test_the_digest_covers_the_bytes_the_signature_covers() -> None:
    card = NorthgatePartner().card  # signed
    assert canonical(card) == _canonicalize_agent_card(card).encode()  # what it signs
    unsigned = NorthgatePartner(signs=False).card
    assert card_digest(card) == card_digest(unsigned)  # signatures are not in it
    assert card_digest(NorthgatePartner(release="v8").card) != card_digest(card)


# The incident, before the policy ------------------------------------------------------

async def test_the_incident_the_lookalike_passes_every_check_before_policy() -> None:
    lookalike = LookAlike()
    url = registry_with_lookalike().lookup(PROVIDER)[0]  # newest first: the look-alike
    assert url == LOOKALIKE_URL
    http = in_process_http(url, lookalike.app, "claims")
    seam = await connect(url, http, ESTIMATE, MERIDIAN_POLICY)  # Chapter 4's connect
    await verify_with_card_key(seam.card, http)  # valid: the key came from the card
    assert seam.breaches == () and seam.upcoming == ()  # the terms are Northgate's
    trusted = await connect_partners(A2APartners().apps())
    graph = build_a2a_graph({**trusted, "repair_estimate": seam})
    out = await graph.ainvoke(clean_claim(), thread("look-alike, before"))
    assert out["decision"] == "paid"  # on the look-alike's estimate
    [sent] = lookalike.executor.received
    assert set(sent) == {"claim_id", "damaged_parts"}  # to a company with no agreement


# The fix ------------------------------------------------------------------------------

async def test_every_meridian_card_is_trusted_today() -> None:
    conns = await connect_partners(A2APartners().apps())
    decisions = {s: c.trust for s, c in conns.items()}
    assert {s: (d.trusted, d.pinned_by, d.inflated) for s, d in decisions.items()
            if d is not None} == {"fraud_screening": (True, "digest", ()),
                                  "repair_estimate": (True, "key", ()),
                                  "payments": (True, "digest", ())}


async def test_the_registry_lookalike_is_refused_and_receives_nothing() -> None:
    lookalike = LookAlike()
    url = registry_with_lookalike().lookup(PROVIDER)[0]  # the look-alike's
    federation = Federation(A2APartners())
    graph = await federation.graph(apps={"repair_estimate": (url, lookalike.app)})
    decision = federation.connections["repair_estimate"].trust
    assert decision is not None and not decision.trusted
    with pytest.raises(CardUntrusted, match="not an origin pinned for Northgate"):
        await graph.ainvoke(clean_claim(), thread("look-alike"))
    assert lookalike.executor.received == []  # not one claim field reached it


async def test_a_lookalike_at_northgates_own_url_fails_the_signature() -> None:
    # A hijacked name or a compromised host: the origin pin passes, the key pin fails.
    hijack = LookAlike(url=NORTHGATE_URL)
    federation = Federation(A2APartners())
    hijacked = {"repair_estimate": (NORTHGATE_URL, hijack.app)}
    graph = await federation.graph(apps=hijacked)
    decision = federation.connections["repair_estimate"].trust
    assert decision is not None
    assert decision.refused == (
        "no signature on the card verifies with a key pinned for Northgate")
    with pytest.raises(CardUntrusted):
        await graph.ainvoke(clean_claim(), thread("hijack"))
    assert hijack.executor.received == []


async def test_the_key_url_in_the_card_is_never_fetched() -> None:
    paths: list[str] = []
    northgate = NorthgatePartner()

    async def watched(scope: dict[str, object], receive: object, send: object) -> None:
        paths.append(str(scope.get("path")))
        await northgate.app(scope, receive, send)  # type: ignore[arg-type]

    app: ASGIApp = watched  # type: ignore[assignment]
    conns = await connect_partners({"repair_estimate": (NORTHGATE_URL, app)})
    assert conns["repair_estimate"].trust is not None
    assert conns["repair_estimate"].trust.trusted
    assert KEY_SET_PATH not in paths  # the key came from Meridian's own store


async def test_an_unsigned_northgate_card_is_refused() -> None:
    partners = A2APartners(northgate=NorthgatePartner(signs=False))
    seam = (await connect_partners(partners.apps()))["repair_estimate"]
    assert seam.trust is not None and seam.trust.refused == "the card is not signed"
    assert seam.breaches == ()  # an untrusted card's terms are never read


async def test_a_risk_release_nobody_approved_is_refused_by_digest() -> None:
    risk = RiskPartner(release="2027.01")  # shipped; its digest never reached Claims
    federation = Federation(A2APartners(risk=risk))
    graph = await federation.graph()
    decision = federation.connections["fraud_screening"].trust
    assert decision is not None and decision.refused.endswith("pinned for Risk")
    with pytest.raises(CardUntrusted):
        await graph.ainvoke(clean_claim(), thread("unapproved release"))
    assert risk.executor.received == []
    refused = [r for r in federation.records if r.error == "CardUntrusted"]
    assert [r.seam for r in refused] == ["fraud_screening"]


# Capability inflation -----------------------------------------------------------------

async def test_capability_inflation_keeps_the_seam_and_holds_the_skill() -> None:
    federation = Federation(A2APartners(northgate=NorthgatePartner(books=True)))
    graph = await federation.graph()
    seam = federation.connections["repair_estimate"]
    assert seam.trust is not None and seam.trust.inflated == ("book_repair",)
    out = await graph.ainvoke(clean_claim(), thread("inflated"))
    assert out["decision"] == "paid"  # estimate_repair still works
    with pytest.raises(CardUntrusted, match="book_repair is not approved"):
        remote_skill_tool(seam, "book_repair", "Northgate")


async def test_a_digest_pin_on_the_inflated_card_takes_the_whole_seam_down() -> None:
    by_digest = replace(NORTHGATE, keys={},
                        digests=frozenset({card_digest(NorthgatePartner().card)}))
    trust = {**MERIDIAN_TRUST, "Northgate": by_digest}
    partners = A2APartners(northgate=NorthgatePartner(books=True))
    conns = await connect_partners(partners.apps(), trust=trust)
    graph = build_a2a_graph(conns)
    with pytest.raises(CardUntrusted, match="is not a card pinned for Northgate"):
        await graph.ainvoke(clean_claim(), thread("digest pin"))
    assert partners.northgate.executor.received == []  # every motor claim waits


# The trust gradient -------------------------------------------------------------------

async def test_an_unproven_skill_may_not_decide_over_its_ceiling() -> None:
    federation = Federation()
    graph = await federation.graph()
    with pytest.raises(CardUntrusted,
                       match="estimate 6,000 is over the 5,000 an unproven "
                             "estimate_repair may decide"):
        await graph.ainvoke(sideswipe(), thread("sideswipe, unproven"))
    for n in range(19):  # with the sideswipe's clean answer, 20 under this card
        await graph.ainvoke(clean_claim(), thread(f"clean-{n}"))
    out = await graph.ainvoke(sideswipe(), thread("sideswipe, proven"))
    assert (out["repair_estimate"], out["decision"]) == (6_000.0, "paid")


async def test_a_new_card_starts_unproven_even_when_validly_signed() -> None:
    partners = A2APartners()
    url, app = partners.apps()["repair_estimate"]
    http = in_process_http(url, app, "claims")
    seam = await connect(url, http, ESTIMATE, MERIDIAN_POLICY, trust=NORTHGATE)
    record = seam.record("estimate_repair")
    record.clean.update(f"t-{n}" for n in range(20))
    assert seam.gradient("estimate_repair").ceiling(record) is None  # proven
    same = await refresh(seam, url, http, ESTIMATE, MERIDIAN_POLICY)
    assert same is seam  # the same digest: the same card, and its track record
    partners.northgate = NorthgatePartner(books=True)  # Northgate ships a new card
    http = in_process_http(url, partners.northgate.app, "claims")
    fresh = await refresh(seam, url, http, ESTIMATE, MERIDIAN_POLICY)
    assert fresh.trust is not None and seam.trust is not None
    assert fresh.trust.trusted and fresh.trust.digest != seam.trust.digest
    assert fresh.gradient("estimate_repair").ceiling(
        fresh.record("estimate_repair")) == 5_000.0  # unproven again


async def test_one_malformed_answer_suspends_the_skill_under_this_card() -> None:
    conns = await connect_partners(A2APartners().apps())
    node = RemoteAgentNode(SEAMS["repair_estimate"], conns["repair_estimate"],
                           log=lambda r: None)
    with pytest.raises(MalformedAnswer):
        node._outcome(task(TaskState.TASK_STATE_COMPLETED, {"estimate": "3000"}))
    with pytest.raises(CardUntrusted, match="suspended under this card"):
        await node.call_once({"claim_id": "CLM-1001", "damaged_parts": ["door"]})


# Where the decision is read -----------------------------------------------------------

async def test_the_inventory_authority_column_starts_with_the_card_decision() -> None:
    federation = Federation(A2APartners(northgate=NorthgatePartner(books=True)))
    graph = await federation.graph()
    inventory = with_trust(with_authority(inventory_from_graph(graph, OWNERS_A2A)),
                           federation.connections)
    answers = {e.callee: e.answers["authority"] for e in inventory.entries}
    assert answers["repair_estimate"].startswith(
        "card trusted by key (unapproved: book_repair); chain for estimate_repair")
    assert answers["fraud_screening"].startswith("card trusted by digest; ")


async def test_authority_goes_only_to_the_card_the_policy_decided_on() -> None:
    conns = await connect_partners(A2APartners().apps())
    decision = conns["repair_estimate"].trust
    assert decision is not None
    assert decision.covers(conns["repair_estimate"].card)
    assert not decision.covers(LookAlike().card)
    assert not decide(LookAlike().card, NORTHGATE_URL, NORTHGATE).trusted


async def test_a_registry_result_is_a_candidate_never_a_decision() -> None:
    lookalike = LookAlike()
    northgate = NorthgatePartner()
    deployed: dict[str, ASGIApp] = {LOOKALIKE_URL: lookalike.app,
                                    NORTHGATE_URL: northgate.app}
    for url in registry_with_lookalike().lookup(PROVIDER):  # try each, newest first
        http = in_process_http(url, deployed[url], "claims")
        seam = await connect(url, http, ESTIMATE, MERIDIAN_POLICY, trust=NORTHGATE)
        if seam.trust is not None and seam.trust.trusted:
            break
    assert seam.card.supported_interfaces[0].url == NORTHGATE_URL  # Northgate's card
    assert lookalike.executor.received == []
