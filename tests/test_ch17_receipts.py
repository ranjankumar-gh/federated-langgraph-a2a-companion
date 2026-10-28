"""Trace what you own, receipt what you don't: the audit nobody could answer, spans that
end at the seam, the seam receipt both owners sign, and what it carries for the earlier
chapters' records.
"""

import asyncio
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
import pytest
from a2a.helpers import new_data_part
from a2a.types import Task

import seamkit.ledger
from meridian.attribution import attribute
from meridian.data_classes import MERIDIAN_POLICY
from meridian.drill import T0
from meridian.federated import A2APartners
from meridian.finance_server import FinancePartner
from meridian.identity import ISSUER, KEYS, MERIDIAN, revocation_sources
from meridian.receipts import CLAIMS_KID, RECEIPT_SECRET, RECEIPTED, pinned, receipted
from meridian.risk_server import RiskPartner
from meridian.saga import SAGA_EXPECTATIONS, SAGA_SEAMS, reopen
from meridian.stubs import CI_TRUST, PRICED, northgate_stub, receipted_stub
from northgate.server import NORTHGATE_V7_4, NORTHGATE_V7_5, NorthgatePartner
from seamkit.approval import decode_approval, refuse_approval
from seamkit.contract import check
from seamkit.drill import Clock
from seamkit.errors import PartnerFailed, SeamError, TaskUnfinished
from seamkit.push import PushRecord, log_push
from seamkit.receipt import (
    ReceiptHalf,
    ReceiptLog,
    ReceiptMismatch,
    Receipts,
    SeamReceipt,
    nonce_for,
    outcome_signed,
)
from seamkit.remote import RemoteAgentNode, connect
from seamkit.server import Arrived, Countersigner
from seamkit.tracing import trace_id
from stub.faults import Fault, FaultScript
from stub.server import STUB_KEY, STUB_KID, StubPartner
from tests.spans import ended, install
from tests.test_ch12_retries import Wire
from tests.test_ch13_compensation import BOOKING, UNDO_BOOKING, Saga
from tests.test_ch16_stub import drive
from tests.wire import as_run, thread

pytestmark = pytest.mark.anyio

DAY = 86_400.0
SEAM = "repair_estimate"
NORTHGATE_SEAMS = (SEAM, BOOKING, UNDO_BOOKING)
PINNED = pinned()


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    install()  # OpenTelemetry's SDK, exporting to memory
    yield shared


@dataclass
class Receipted(Saga):
    """Chapter 13's saga on Northgate terms v7.5, with Risk and Finance countersigning
    and every Claims node cutting receipts into one log."""

    log: ReceiptLog = field(default_factory=ReceiptLog)
    mismatches: list[ReceiptMismatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        lists = revocation_sources(self.issuer)
        assessors, records = asyncio.Event(), asyncio.Event()
        assessors.set()
        records.set()
        self.northgate = NorthgatePartner(release="v7.5", assessors=assessors,
                                          revocations=lists, now=self.clock)
        self.finance = FinancePartner(release="2026.15", revocations=lists,
                                      now=self.clock, countersigns=True)
        self.risk = RiskPartner(release="2026.12", lookups=records, revocations=lists,
                                now=self.clock, countersigns=True)
        self.partners = A2APartners(risk=self.risk, northgate=self.northgate,
                                    finance=self.finance)
        self.undo_wire = Wire(self.northgate.app)

    async def graph(self) -> Any:
        graph = await super().graph()
        receipted([*self.nodes.values(), *self.undoer.undo.values()], self.log,
                  self.mismatches.append)
        return graph

    def receipts(self, tid: str, seams: tuple[str, ...] = ()) -> list[SeamReceipt]:
        return [r for r in self.log.receipts(seams) if r.thread == tid]


async def saga_with_residue(saga: Saga, tid: str) -> Any:
    """Chapter 13's incident, fixed: booked, the frame ordered, reversed, cancelled
    with a residue of 6,500. Then two weeks pass, and Northgate forgets."""
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, tid)
    saga.northgate.executor.shop_orders("CLM-1002", "frame")
    out = await reopen(graph, thread(tid), prior_claims=3)
    assert out["undone"]["repair_booking"] == "residue"
    saga.clock.advance(14 * DAY)
    await saga.northgate.forget()
    return graph


# The incident ------------------------------------------------------------------------

async def test_the_incident_only_claims_word_remains(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga_with_residue(saga, "a17")
    ledger = await saga.ledger(graph, "a17")
    sent = {f for r in saga.records if r.partner == "Northgate" and r.call == "send"
            for f in r.sent}
    assert sent == {"claim_id", "damaged_parts", "task"}  # Claims' log, Claims' word
    assert saga.findings[0].detail.startswith("owed 6,500")  # Claims' finding
    with as_run(thread("a17")), pytest.raises(PartnerFailed):  # Northgate has none
        await saga.nodes[BOOKING].read(ledger[BOOKING]["task_id"])
    assert not any("receipt" in entry for entry in ledger.values())


# Receipt what you don't ----------------------------------------------------------------

async def test_fixed_the_receipt_log_answers_the_auditor(clock: Clock) -> None:
    saga = Receipted(clock)
    await saga_with_residue(saga, "a17")
    crossed = saga.receipts("a17", NORTHGATE_SEAMS)  # after Northgate forgot
    assert {r.kind for r in crossed} == {"countersigned"}
    assert all(r.theirs and r.theirs.verifies(PINNED[r.theirs.signer])
               for r in crossed)
    names = {f for r in crossed if r.theirs for f in r.theirs.fields}
    assert names == {"claim_id", "damaged_parts", "task"}  # Northgate signed for these
    [cancel] = [r for r in crossed if r.seam == UNDO_BOOKING]
    assert cancel.outcome["owed"] == 6_500 and cancel.theirs is not None
    assert outcome_signed(cancel, nonce_for(RECEIPT_SECRET, cancel.idempotency_key))
    assert saga.findings[0].receipt == cancel.digest  # the finding cites it


async def test_no_claim_history_or_fraud_field_ever_crossed_to_northgate(
        clock: Clock) -> None:
    saga = Receipted(clock)
    await saga_with_residue(saga, "f17")
    crossed = {f for r in saga.receipts("f17", NORTHGATE_SEAMS) if r.theirs
               for f in r.theirs.fields}
    classes = {MERIDIAN_POLICY.classes.get(f, "unclassified") for f in crossed}
    assert classes.isdisjoint({"claim_history", "fraud_assessment"})
    screened = {f for r in saga.receipts("f17", ("fraud_screening",)) if r.theirs
                for f in r.theirs.fields}
    assert "prior_claims" in screened  # the history went to Risk, and Risk signed it


async def test_every_receipt_carries_the_card_the_chain_and_the_behaviour(
        clock: Clock) -> None:
    saga = Receipted(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "c17")
    ledger = await saga.ledger(graph, "c17")
    [estimate] = saga.receipts("c17", (SEAM,))
    entry = ledger[SEAM]
    assert (estimate.card_digest, estimate.authority) == (entry["card_digest"],
                                                          entry["authority"])
    assert estimate.behaviour == entry.get("behaviour", "")
    assert entry["receipt"] == estimate.digest and estimate.terms == "v7.5"
    assert "token" not in str(estimate.plain()).lower()


async def test_the_approval_rides_in_the_payments_receipt_and_still_verifies(
        clock: Clock) -> None:
    saga = Receipted(clock)
    graph = await saga.graph()
    await saga.paid(graph, "p17")
    [carried] = [r for r in saga.receipts("p17", ("payments",)) if r.approval]
    approval = decode_approval(carried.approval)
    assert approval is not None and approval.task_id == carried.ours.task_id
    assert refuse_approval({"approval": carried.approval}, {ISSUER: MERIDIAN.keys[ISSUER]},
                           approval.role, approval.task_id, approval.claim,
                           approval.amount, approval.asked) == ""
    assert carried.kind == "countersigned"  # Finance signed for the reply it received


async def test_hop_attribution_cites_the_receipt(clock: Clock) -> None:
    saga = Receipted(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "h17")
    state = (await graph.aget_state(thread("h17"))).values
    hop = attribute("repair_estimate", state, saga.records, SAGA_EXPECTATIONS)
    [estimate] = saga.receipts("h17", (SEAM,))
    assert hop.owner == "Claims" and hop.receipt == estimate.digest
    assert estimate.theirs is not None  # what Northgate signed, past Claims' records


# A partner that lies or slips ---------------------------------------------------------

class SignsAnotherAnswer(Countersigner):
    """A partner whose countersigner hashes an answer it did not send."""

    def stamp(self, task: Task) -> Task:
        other = Task()
        other.CopyFrom(task)
        other.status.message.parts.append(new_data_part({"estimate": 2_450.0}))
        out = Task()
        out.CopyFrom(task)
        out.metadata.CopyFrom(super().stamp(other).metadata)
        return out


class SignsAnotherRequest(Countersigner):
    """A partner whose countersigner hashes a request Claims never sent."""

    def stamp(self, task: Task) -> Task:
        came = self.arrived[task.id]
        self.arrived[task.id] = Arrived(came.skill, came.nonce, "sha256:" + "0" * 64,
                                        came.fields)
        return super().stamp(task)


async def receipted_node(stub: StubPartner, countersigned: bool = True,
                         trust: Mapping[str, Any] = CI_TRUST,
                         log: ReceiptLog | None = None,
                         mismatches: list[ReceiptMismatch] | None = None,
                         ) -> RemoteAgentNode:
    """Claims' estimate node on the stub, cutting receipts under Claims' CI pins."""
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=stub.app),
                             base_url=stub.url)
    connection = await connect(stub.url, http, SAGA_EXPECTATIONS[SEAM], MERIDIAN_POLICY,
                               terms=stub.terms, trust=CI_TRUST["Northgate"])
    receipts = Receipts(KEYS["claims"], CLAIMS_KID, RECEIPT_SECRET, pinned(trust),
                        log or ReceiptLog(), countersigned,
                        (mismatches if mismatches is not None else []).append)
    return RemoteAgentNode(SAGA_SEAMS[SEAM], connection, lambda record: None,
                           receipts=receipts)


@pytest.mark.parametrize(("signer", "why"), [
    (SignsAnotherAnswer, "answered something other than what it signed"),
    (SignsAnotherRequest, "signed for a request Claims never sent"),
])
async def test_a_half_that_does_not_match_is_a_mismatch_for_both_owners(
        signer: type[Countersigner], why: str) -> None:
    stub = receipted_stub()
    stub.handler.countersigner = signer(STUB_KEY, STUB_KID)
    mismatches: list[ReceiptMismatch] = []
    out = await drive(await receipted_node(stub, mismatches=mismatches), PRICED)
    assert isinstance(out["error"], ReceiptMismatch) and why in str(out["error"])
    assert "repair_estimate" not in out  # the answer is not taken
    assert [m.detail for m in mismatches] == [out["error"].detail]


async def test_a_partner_that_promised_a_half_and_sent_none(clock: Clock) -> None:
    stub = receipted_stub()
    stub.handler.countersigner = None  # v7.5 on the card, and a release that forgot
    out = await drive(await receipted_node(stub), PRICED)
    assert isinstance(out["error"], ReceiptMismatch)
    assert "carries no countersignature" in str(out["error"])


async def test_terms_without_receipts_leave_a_one_sided_receipt() -> None:
    log = ReceiptLog()
    out = await drive(await receipted_node(northgate_stub(), countersigned=False,
                                           log=log), PRICED)
    assert out["repair_estimate"] == 3_000.0
    [receipt] = log.receipts()
    assert receipt.kind == "ours" and receipt.theirs is None  # Claims' word only


async def test_the_stubs_ci_key_signs_nothing_production_trusts() -> None:
    from meridian.trust import SAGA_TRUST
    out = await drive(await receipted_node(receipted_stub(), trust=SAGA_TRUST), PRICED)
    assert isinstance(out["error"], ReceiptMismatch) and STUB_KID in str(out["error"])


def test_checks_rule_refuses_countersigned_expectations_against_terms_without() -> None:
    expects = RECEIPTED[SEAM]
    [breach] = check(NORTHGATE_V7_4, expects, MERIDIAN_POLICY)
    assert breach.detail == ("estimate_repair countersigns no receipts, and this seam "
                             "expects every answer countersigned")
    assert check(NORTHGATE_V7_5, expects, MERIDIAN_POLICY) == []
    assert NORTHGATE_V7_5.effective == date(2027, 2, 15)


def test_a_deleted_receipt_breaks_the_seams_chain(tmp_path: Any) -> None:
    log = ReceiptLog(tmp_path)
    for n in range(3):
        half = ReceiptHalf("estimate_repair", f"t{n}", "c", "completed", "s", "a",
                           ("claim_id",), "claims", "2027-02-20T10:00:00+00:00")
        log.append(SeamReceipt(SEAM, "x17", half, None, "", "", "", "v7", "v7.5",
                               f"k{n}", ""))
    assert log.broken(SEAM) == ""
    path = tmp_path / f"{SEAM}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")  # the middle one, gone
    assert log.broken(SEAM) == f"{SEAM}: receipt 1 does not follow the one before it"


# Trace what you own --------------------------------------------------------------------

async def test_spans_end_at_the_seam_and_a_later_read_links_to_the_send(
        clock: Clock) -> None:
    stub = receipted_stub(script=FaultScript().add(Fault("hang")))
    node = await receipted_node(stub)
    out = await drive(node, PRICED)
    assert isinstance(out["error"], TaskUnfinished)  # Northgate's assessor desk
    entry = out["ledger"][SEAM]
    clock.advance(21 * 60)
    await stub.finish(entry["task_id"])
    done = await drive(node, out)
    assert done["repair_estimate"] == 3_000.0
    send, *reads = ended(SEAM)
    assert send.name == f"{SEAM} send" and {r.name for r in reads} == {f"{SEAM} read"}
    assert trace_id(entry["traceparent"]) == format(send.context.trace_id, "032x")
    assert all(r.links[0].context.span_id == send.context.span_id for r in reads)
    values = {str(v) for s in ended() for v in (s.attributes or {}).values()}
    assert not values & {"CLM-R01", "3000.0", "3000", "bumper"}  # names, never values


async def test_the_partner_starts_its_own_trace_linked_to_claims(clock: Clock) -> None:
    node = await receipted_node(receipted_stub())
    await drive(node, PRICED)
    send = ended(f"{SEAM} send")[0]
    [served] = ended("serve estimate_repair")
    assert served.parent is None  # its own trace, though it ran in the same process
    assert served.context.trace_id != send.context.trace_id
    assert served.links[0].context.span_id == send.context.span_id  # linked, by metadata


def test_each_push_verdict_is_a_crossing_record_with_no_body(
        caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="seamkit.crossing"):
        log_push(PushRecord(SEAM, "r01", "task-9", 401, "token"))
    [record] = [r for r in caplog.records if r.name == "seamkit.crossing"]
    crossing = record.__dict__["crossing"]
    assert (crossing["call"], crossing["sent"], crossing["state"]) == (
        "push", (), "401 token")
    assert crossing["task_id"] == "task-9"


def test_a_seam_error_is_what_a_mismatch_is() -> None:
    assert issubclass(ReceiptMismatch, SeamError)  # a run ends on it, like any other
