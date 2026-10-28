"""Who answers: the ownership map on two owners' sides, every earlier chapter's signals
routed through it, and seam SLOs with error budgets measured from crossing records.
"""

from collections.abc import Iterator
from dataclasses import replace

import pytest

import seamkit.ledger
from meridian.drill import T0
from meridian.federated import EXPECTATIONS
from meridian.receipts import OWNERSHIP, RISK_OWNERSHIP, with_receipts
from meridian.saga import SAGA_SEAMS, UNDO_EXPECTATIONS
from meridian.stubs import PRICED, northgate_stub
from seamkit.breaker import SeamBreaker
from seamkit.drill import Clock
from seamkit.errors import BreakerOpen
from seamkit.inventory import SeamEntry, SeamInventory
from seamkit.ownership import (
    Signal,
    UnmappedSeam,
    from_breaker,
    from_budget,
    from_finding,
)
from seamkit.reconcile import Finding
from seamkit.remote import CrossingRecord
from seamkit.slo import burn
from stub.faults import Fault, FaultScript
from tests.test_ch14_containment import FLOOD
from tests.test_ch16_stub import drive, estimate_breaker, node_on

pytestmark = pytest.mark.anyio

NORTHGATE_CONTACT = "northgate-support, via the partner manager"
SLO = EXPECTATIONS["repair_estimate"].slo


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


def opened(seam: str, owner: str, skill: str, at: float) -> SeamBreaker:
    breaker = SeamBreaker(seam, owner, skill, FLOOD)
    for n in range(FLOOD.failures):  # failures to answer, inside one window
        breaker.observe(True, at + n)
    assert breaker.state == "open"
    return breaker


def test_chapter_14s_incident_is_routed_on_two_maps(clock: Clock) -> None:
    screening = opened("fraud_screening", "Risk", "screen_fraud", clock())
    claims = [p for s in from_breaker(screening) for p in OWNERSHIP.route(s)]
    assert {(p.to, p.side) for p in claims} == {("claims-oncall", "caller"),
                                                ("risk-oncall", "callee")}
    assert not any("northgate" in p.to for p in claims)  # not Claims' partner
    history = opened("estimate_history", "Northgate", "estimate_history", clock())
    risk = [p for s in from_breaker(history) for p in RISK_OWNERSHIP.route(s)]
    assert {(p.to, p.side) for p in risk} == {("risk-oncall", "caller"),
                                              (NORTHGATE_CONTACT, "callee")}


def test_a_drift_finding_reaches_both_sides_of_the_seam() -> None:
    drift = Finding("repair_estimate", "canary drift", "", "",
                    detail="estimate_repair: CLM-C02, CLM-C03 against bv:3ec17ceb844a")
    pages = OWNERSHIP.route(from_finding(drift))
    assert [(p.to, p.signal.kind) for p in pages] == [
        ("claims-oncall", "drift"), (NORTHGATE_CONTACT, "drift")]


def test_a_residue_goes_to_a_queue_a_person_works() -> None:
    residue = Finding("repair_booking", "residue", "k", "t",
                      detail="owed 6,500; frame ordered by SHOP-7")
    pages = OWNERSHIP.route(from_finding(residue))
    assert [(p.to, p.side) for p in pages] == [("claims-oncall", "caller"),
                                               ("claims-recoveries", "queue")]


def test_every_seam_claims_runs_is_on_its_map() -> None:
    seams = [*SAGA_SEAMS, *(e.seam for e in UNDO_EXPECTATIONS.values())]
    OWNERSHIP.require(seams)  # startup passes
    short = replace(OWNERSHIP, seams={s: o for s, o in OWNERSHIP.seams.items()
                                      if s != "payments.undo"})
    with pytest.raises(UnmappedSeam, match="payments.undo is not on the Claims map"):
        short.require(seams)
    with pytest.raises(UnmappedSeam, match="sends breaker on estimate_history nowhere"):
        OWNERSHIP.route(Signal("estimate_history", "breaker", "open"))


def records(now: float, late: int, refused: int) -> list[CrossingRecord]:
    """A window of estimate sends: `late` past the SLO's seconds, `refused` by
    Claims' own breaker, and one from before the window."""
    sent = [CrossingRecord("repair_estimate", "Northgate", "v7", "send", (),
                           state="TASK_STATE_COMPLETED", at=now - n * 600,
                           seconds=5.5 if n < late else 2.0) for n in range(100)]
    held = [CrossingRecord("repair_estimate", "Northgate", "v7", "send", (),
                           error="BreakerOpen", at=now - 60) for _ in range(refused)]
    old = CrossingRecord("repair_estimate", "Northgate", "v7", "send", (),
                         error="PartnerUnavailable", at=now - 40 * 86_400)
    return [*sent, *held, old]


def test_the_slo_burns_on_late_answers_and_pages_both_owners(clock: Clock) -> None:
    assert SLO is not None and (SLO.target, SLO.seconds) == (0.95, 4.0)
    budget = burn("repair_estimate", SLO, records(clock(), late=3, refused=10), clock())
    assert (budget.sends, budget.bad, round(budget.spent, 2)) == (100, 3, 0.6)
    [signal] = from_budget(budget, SLO.alert, clock())
    assert [p.to for p in OWNERSHIP.route(signal)] == ["claims-oncall",
                                                       NORTHGATE_CONTACT]
    calm = burn("repair_estimate", SLO, records(clock(), late=2, refused=0), clock())
    assert from_budget(calm, SLO.alert) == []  # 40% spent: under the alert


async def test_a_breaker_opening_spends_no_budget(clock: Clock) -> None:
    assert SLO is not None
    stub = northgate_stub(script=FaultScript().add(Fault("fail")))
    crossed: list[CrossingRecord] = []
    breaker = estimate_breaker()
    node = await node_on(stub, breaker=breaker, records=crossed)
    for _ in range(3):  # three failures open it
        await drive(node, PRICED)
    for _ in range(5):  # five sends refused by Claims' own breaker
        assert isinstance((await drive(node, PRICED))["error"], BreakerOpen)
    budget = burn("repair_estimate", SLO, crossed, clock())
    assert (budget.sends, budget.bad) == (3, 3)  # only what reached Northgate


def test_the_inventory_accountability_column_is_filled() -> None:
    inventory = SeamInventory([SeamEntry("claims_intake", "repair_estimate", "Claims",
                                         "Northgate")])
    with_receipts(inventory, OWNERSHIP, {"repair_estimate": "countersigned"},
                  {"repair_estimate": SLO})
    [entry] = inventory.entries
    assert entry.answers["accountability"] == (
        "receipts: countersigned; owners: claims-oncall / northgate-support, via the "
        "partner manager; SLO 95% in 4 s over 28 days")
    assert "accountability" not in entry.open_properties()
