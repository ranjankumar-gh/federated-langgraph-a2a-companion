"""Who pays, across a company line: Northgate's statement checked against the receipt
log, the residue on Claims' compensation line, and the credit a breached SLO earns."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

import seamkit.ledger
from meridian.costs import NORTHGATE_PRICES, OWNERSHIP, costed, paging
from meridian.drill import T0
from meridian.samples import clean_claim
from seamkit.costs import (
    Charge,
    CostLedger,
    check_invoice,
    credit,
    purpose_of,
)
from seamkit.drill import Clock
from seamkit.ownership import Page
from seamkit.receipt import SeamReceipt
from seamkit.slo import Budget
from tests.spans import install
from tests.test_ch13_compensation import UNDO_BOOKING
from tests.test_ch17_receipts import NORTHGATE_SEAMS, Receipted, saga_with_residue
from tests.wire import thread

pytestmark = pytest.mark.anyio

SEAM = "repair_estimate"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    install()
    yield shared


@dataclass
class Costed(Receipted):
    """Chapter 17's receipted saga, with every budgeted seam pricing its tasks."""

    charges: CostLedger = field(default_factory=CostLedger)
    pages: list[Page] = field(default_factory=list)

    async def graph(self) -> Any:
        graph = await super().graph()
        costed([*self.nodes.values(), *self.undoer.undo.values()], self.charges,
               paging(OWNERSHIP, self.pages, self.clock))
        return graph

    def northgate_receipts(self) -> list[SeamReceipt]:
        return list(self.log.receipts(NORTHGATE_SEAMS))


async def two_claims(clock: Clock) -> Costed:
    """CLM-1002's saga with its residue, then CLM-1001 estimated and waiting to pay."""
    saga = Costed(clock)
    graph = await saga_with_residue(saga, "a18")
    await graph.ainvoke(clean_claim(), thread("b18"))  # pauses for approval to pay
    return saga


async def test_northgates_statement_matches_the_receipts(clock: Clock) -> None:
    saga = await two_claims(clock)
    lines = saga.northgate.statement()
    assert [c.price for c in lines] == [3.5, 3.5]  # two estimates, at v7.5's price
    assert check_invoice(lines, saga.charges.entries([SEAM]), saga.northgate_receipts(),
                         NORTHGATE_PRICES) == []
    estimates = saga.charges.entries([SEAM])
    assert {(e.purpose, e.price, e.terms) for e in estimates} == {
        ("claim", 3.5, "v7.5")}
    assert all(e.receipt for e in estimates)  # each charge names its receipt


async def test_four_wrong_lines_are_four_findings(clock: Clock) -> None:
    saga = await two_claims(clock)
    first, second = saga.northgate.statement()
    revoked = canceled_receipt(saga)  # Chapter 7's revoked estimate, as Claims holds it
    lines = [first, first,  # billed twice under one honoured key
             replace(second, price=4.0),  # not v7.5's price
             Charge("task-nobody-sent", "k-unknown", "estimate_repair", 3.5),
             Charge(revoked.ours.task_id, revoked.idempotency_key, "estimate_repair",
                    3.5)]
    found = check_invoice(lines, saga.charges.entries([SEAM]),
                          [*saga.northgate_receipts(), revoked], NORTHGATE_PRICES)
    assert [f.kind for f in found] == ["twice", "price", "no receipt", "canceled"]
    assert found[1].detail == "billed 4; terms v7.5 say 3.5"


def canceled_receipt(saga: Costed) -> SeamReceipt:
    """A receipt for an estimate that ended canceled: a send under a chain revoked
    before Northgate's commit act point (Chapter 7). Never answered, never billed."""
    [estimate, *_] = [r for r in saga.northgate_receipts() if r.seam == SEAM]
    return replace(estimate, ours=replace(estimate.ours, task_id="task-revoked",
                                          state="canceled"),
                   idempotency_key="k-revoked")


async def test_the_residue_is_a_charge_on_claims_compensation_line(
        clock: Clock) -> None:
    saga = Costed(clock)
    await saga_with_residue(saga, "a18")
    [undo] = saga.charges.entries([UNDO_BOOKING])
    assert (undo.purpose, undo.price, undo.partner) == ("compensation", 6_500.0,
                                                        "Northgate")
    [cancel] = [r for r in saga.northgate_receipts() if r.seam == UNDO_BOOKING]
    assert undo.receipt == cancel.digest  # the owed amount Northgate signed for
    assert saga.northgate.statement()[0].price == 3.5  # the residue is not invoiced


async def test_an_unpriced_seam_is_on_the_ledger_with_no_price(clock: Clock) -> None:
    saga = Costed(clock)
    await saga_with_residue(saga, "a18")
    [screening, *_] = saga.charges.entries(["fraud_screening"])
    assert screening.price is None  # Risk 2026.12: no price, no report
    assert screening.usage == "" and screening.receipt  # receipted, not priced


def test_a_breached_slo_earns_the_agreed_credit() -> None:
    window = [Charge(f"t-{n}", f"k-{n}", "estimate_repair", 3.5) for n in range(100)]
    inside = Budget(SEAM, 100, 3, 5.0)  # Chapter 17's window: 60% spent
    breached = Budget(SEAM, 100, 6, 5.0)  # 120%: the promise is broken
    assert credit(window, inside.spent, 0.10) == 0.0
    assert credit(window, breached.spent, 0.10) == 35.0  # 10% of 350, a design value


def test_what_each_send_is_for() -> None:
    assert purpose_of({"claim_id": "CLM-C01"}, False, False, False) == "canary"
    assert purpose_of({"claim_id": "CLM-R02"}, False, False, False) == "recording"
    assert purpose_of({"claim_id": "CLM-1002"}, False, False, True) == "compensation"
    assert purpose_of({"claim_id": "CLM-1002"}, False, True, False) == "trial"
    assert purpose_of({"claim_id": "CLM-1002"}, True, False, False) == "retry"
    assert purpose_of({"claim_id": "CLM-1002"}, False, False, False) == "claim"
