"""Seam contract tests, both ways: the incident no stub of Claims' could have seen, the
same suite run by Northgate against its release candidate, the recording and the suite
kept current, conformance, a partner with paper terms, and hop attribution.

Here the companion's Northgate plays one role only: Northgate's own deployments, its
staging and its release candidate, on Northgate's side of the line.
"""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import seamkit.ledger
from meridian.attribution import Hop, attribute, hop_on
from meridian.containment import CONTAINED_EXPECTATIONS, DESK, OWNERS, with_breaker
from meridian.drill import T0
from meridian.federated import EXPECTATIONS
from meridian.incident import fail_unpriced, swap_model
from meridian.saga import build_saga
from meridian.samples import clean_claim, suspect_claim
from meridian.stubs import (
    FRAME,
    PHOTO,
    PRICED,
    RECORDING,
    SUITE,
    estimate_suite,
    northgate_stub,
    record,
    with_tests,
)
from northgate.server import NORTHGATE_URL, NORTHGATE_V7_4, NorthgatePartner
from seamkit.contracttest import Failure, conformance, load_suite, verify
from seamkit.drill import Clock
from seamkit.inventory import inventory_from_graph
from seamkit.remote import connect
from seamkit.server import a2a_app
from stub.recording import load, stale
from stub.server import StubOutOfContract, StubPartner
from tests.test_ch12_retries import no_wait
from tests.test_ch14_containment import Contained, asks_history
from tests.wire import thread

pytestmark = pytest.mark.anyio

SEAM = "repair_estimate"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


def candidate(**changes: Any) -> tuple[NorthgatePartner, httpx.AsyncClient]:
    """A Northgate deployment on Northgate's side: staging, or a release candidate."""
    northgate = NorthgatePartner(release="v7.4", **changes)
    app = a2a_app(northgate.card, northgate.executor)
    return northgate, httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                        base_url=NORTHGATE_URL)


def on_stub(stub: StubPartner) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=stub.app),
                             base_url=stub.url)


def unpriced() -> dict[str, Any]:
    """CLM-1001 as a claim that also lists a part Northgate does not price."""
    return {**clean_claim(), "damaged_parts": ["bumper", "mirror"]}


# The incident ------------------------------------------------------------------------

async def test_the_incident_a_failed_unpriced_part_opens_the_estimate_breaker(
        clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    for n in range(3):  # before the release: Claims' own request, refused
        out = await graph.ainvoke(unpriced(), thread(f"before-{n}"))
        assert out["aborted"].startswith("repair_estimate: PartnerRejected")
    assert fed.room.breakers[SEAM].state == "closed"
    fail_unpriced(fed.northgate)  # the code release: same card, same terms
    for n in range(3, 6):
        out = await graph.ainvoke(unpriced(), thread(f"after-{n}"))
        assert out["aborted"].startswith("repair_estimate: PartnerFailed")
    assert fed.room.breakers[SEAM].state == "open"
    out = await graph.ainvoke(clean_claim(), thread("clean"))
    [asked] = out["__interrupt__"]  # a priced claim: Northgate is never asked
    assert (asked.value["to"], asked.value["needs"]) == (DESK, SEAM)
    # Claims' CI ran its suite against its stub, and every test passed.
    stub = on_stub(northgate_stub())
    assert await verify(stub, NORTHGATE_URL, estimate_suite()) == []


async def test_fixed_northgates_ci_fails_the_candidate_before_release() -> None:
    tests = load_suite(SUITE)  # the file Claims publishes to Northgate
    northgate, http = candidate()
    assert await verify(http, NORTHGATE_URL, tests) == []  # the release in force
    fail_unpriced(northgate)  # the release candidate
    assert await verify(http, NORTHGATE_URL, tests) == [
        Failure("CLM-R03: rejected", "expected rejected, got failed")]
    assert await verify(on_stub(northgate_stub()), NORTHGATE_URL, tests) == []


async def test_the_suite_asserts_what_claims_reads_never_the_value() -> None:
    northgate, http = candidate()
    swap_model(northgate)  # Chapter 15's door, 1,800 to 1,250
    assert await verify(http, NORTHGATE_URL, load_suite(SUITE)) == []  # a canary's job


# Keeping the stub true ---------------------------------------------------------------

async def test_the_committed_recording_and_suite_match_northgates_staging() -> None:
    northgate, _ = candidate()
    app = a2a_app(northgate.card, northgate.executor)
    made = await record(httpx.ASGITransport(app=app), NORTHGATE_URL)
    committed = load(RECORDING)
    assert (made.card_digest, made.exchanges) == (committed.card_digest,
                                                  committed.exchanges)
    assert estimate_suite([made]) == load_suite(SUITE)


async def test_conformance_holds_on_the_stub_and_on_northgates_staging() -> None:
    _, staging = candidate()
    for http in (on_stub(northgate_stub()), staging):
        assert await conformance(http, NORTHGATE_URL, "estimate_repair", PRICED,
                                 (FRAME, PHOTO)) == []


async def test_a_recording_goes_stale_when_the_partner_moves() -> None:
    recording = load(RECORDING)
    stub = northgate_stub()
    connection = await connect(stub.url, on_stub(stub), EXPECTATIONS[SEAM])
    assert stale(recording, connection) == []
    _, staging = candidate(announces=None)
    v8 = NorthgatePartner(release="v8")
    moved = await connect(NORTHGATE_URL, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=a2a_app(v8.card, v8.executor)),
        base_url=NORTHGATE_URL), EXPECTATIONS[SEAM])
    assert [s.split(" ")[0] + " " + s.split(" ")[1] for s in stale(recording, moved)
            ] == ["card digest", "terms revision"]
    connection.behaviour["estimate_repair"] = "bv:000000000000"  # a person accepted
    assert stale(recording, connection) == [
        "behaviour version bv:3ec17ceb844a is now bv:000000000000"]


async def test_a_partner_with_paper_terms_gets_a_stub_and_a_suite() -> None:
    northgate, _ = candidate(publishes=False, signs=False)
    app = a2a_app(northgate.card, northgate.executor)
    paper = [NORTHGATE_V7_4]  # agreed in a document; nothing on the card
    made = await record(httpx.ASGITransport(app=app), NORTHGATE_URL, terms=paper)
    with pytest.raises(StubOutOfContract, match="no terms"):
        StubPartner(made)
    stub = StubPartner(made, terms=paper)
    tests = estimate_suite([made])
    assert await verify(on_stub(stub), NORTHGATE_URL, tests) == []
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url=NORTHGATE_URL)
    assert await verify(http, NORTHGATE_URL, tests) == []  # Northgate runs it too


# Hop attribution ---------------------------------------------------------------------

async def test_chapter_14s_incident_attributed_from_each_owners_side(
        clock: Clock) -> None:
    fed = Contained(clock, slow=12.0, risk_release="2026.12", risk_breaker=False,
                    assistant=asks_history)
    graph, nodes, _ = await build_saga(fed.partners, fed.http(), fed.saver,
                                       fed.issuer, fed.records.append)
    for node in nodes.values():
        node.sleep = no_wait
    await graph.ainvoke(suspect_claim(), thread("i16"))
    state = (await graph.aget_state(thread("i16"))).values
    claims = attribute("fraud_score", state, fed.records, CONTAINED_EXPECTATIONS)
    assert (claims.owner, claims.seam) == ("Risk", "fraud_screening")
    assert claims.why.endswith("on the send")
    risk = hop_on("estimate_history", "Northgate", fed.risk.records)
    assert risk is not None and (risk.owner, risk.seam) == ("Northgate",
                                                            "estimate_history")


async def test_a_wrong_answer_every_hop_passed_is_attributed_to_claims(
        clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    swap_model(fed.northgate)
    await graph.ainvoke(clean_claim(), thread("w16"))
    state = (await graph.aget_state(thread("w16"))).values
    assert state["repair_estimate"] == 2_450.0  # Chapter 15's door, unseen here
    assert attribute("repair_estimate", state, fed.records,
                     CONTAINED_EXPECTATIONS) == Hop(
        "Claims", SEAM, "Northgate's answer met Claims' expectations")
    assert attribute("payment_memo", state, fed.records,
                     CONTAINED_EXPECTATIONS).owner == "Claims"


# The inventory -----------------------------------------------------------------------

async def test_the_inventory_says_who_runs_the_contract_tests(clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    inventory = with_tests(with_breaker(inventory_from_graph(graph, OWNERS)),
                           {SEAM: True, "repair_booking": False})
    failure = {e.callee: e.answers["failure"] for e in inventory.entries}
    assert failure[SEAM].endswith("; substitute; stubbed; contract tests run by "
                                  "Northgate: yes")
    assert failure["repair_booking"].endswith("contract tests run by Northgate: no")
