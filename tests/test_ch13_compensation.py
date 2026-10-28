"""Compensation across owners: the repair nobody cancelled, the saga plan, the
compensator and its three outcomes, authority for an undo, and why the hub orchestrates.
"""

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import northgate.server
import seamkit.ledger
from meridian.drill import T0
from meridian.federated import GRANTS, A2APartners
from meridian.finance_server import FinancePartner
from meridian.identity import (
    ADJUSTER,
    COMPENSATOR,
    KEYS,
    LEAD_SESSION,
    MANAGER_SESSION,
    RECONCILER_SESSION,
    MeridianIssuer,
    revocation_sources,
)
from meridian.risk_server import RiskPartner
from meridian.saga import MERIDIAN_PLAN, SagaGraph, adopt, build_saga, reopen
from meridian.samples import suspect_claim
from northgate.server import NORTHGATE_URL, NORTHGATE_V7_4, NorthgatePartner
from seamkit.approval import encode_approval
from seamkit.delegation import (
    HEADER,
    Chain,
    Grant,
    as_peer,
    encode,
    mint,
    run_values,
)
from seamkit.drill import Clock, probe
from seamkit.errors import (
    ContractBreach,
    DelegationRefused,
    PartnerRejected,
    TaskUnfinished,
)
from seamkit.ledger import LedgerEntry
from seamkit.push import WakeQueue
from seamkit.reconcile import Finding, Reconciler
from seamkit.remote import CrossingRecord, RemoteAgentNode
from seamkit.resume import wake
from seamkit.saga import check_plan
from tests.test_ch12_retries import Crash, Wire, no_wait
from tests.wire import as_run, thread

pytestmark = pytest.mark.anyio

HOUR = 3600.0
BOOKING, PAY, UNDO_BOOKING = "repair_booking", "payments", "repair_booking.undo"
# CLM-1002 before the policy merge: its three earlier claims sit under an old policy
# number, so Risk sees none. Two signals fire, and 68 is under the threshold of 70.
BEFORE_MERGE = {**suspect_claim(), "prior_claims": 0}


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    """One clock for every owner: Meridian's issuer, the partners, and the ledger."""
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


@dataclass
class Saga:
    """Claims, Risk, Northgate v7.4 and Finance 2026.15, on one clock."""

    clock: Clock
    undoes: bool = True  # False: the incident's graph, which refers and undoes nothing
    saver: InMemorySaver = field(default_factory=InMemorySaver)
    records: list[CrossingRecord] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        lists = revocation_sources(self.issuer)
        assessors = asyncio.Event()
        assessors.set()  # the assessor has the shop's photo of CLM-1002's frame
        self.northgate = NorthgatePartner(release="v7.4", assessors=assessors,
                                          revocations=lists, now=self.clock)
        self.finance = FinancePartner(release="2026.15", revocations=lists,
                                      now=self.clock)
        records = asyncio.Event()
        records.set()  # Risk 2026.12 looks the police report up; nobody is asked
        self.risk = RiskPartner(release="2026.12", lookups=records, revocations=lists,
                                now=self.clock)
        self.partners = A2APartners(risk=self.risk, northgate=self.northgate,
                                    finance=self.finance)
        self.undo_wire = Wire(self.northgate.app)  # cancel_booking's own wire

    async def graph(self) -> SagaGraph:
        graph, self.nodes, self.undoer = await build_saga(
            self.partners, {UNDO_BOOKING: self.undo_wire.http(NORTHGATE_URL)},
            self.saver, self.issuer, self.records.append, self.findings.append,
            undoes=self.undoes)
        for node in [*self.nodes.values(), *self.undoer.undo.values()]:
            node.sleep = no_wait
        # Northgate's estimates are proven under this card (Chapter 5's gradient), so
        # its unproven ceiling does not stop CLM-1002's 8,300.
        record = self.nodes["repair_estimate"].partner.record("estimate_repair")
        record.clean.update(f"earlier-{n}" for n in range(20))
        return graph

    async def booked_and_waiting(self, graph: SagaGraph, tid: str) -> dict[str, Any]:
        """Day one: cleared at 68, estimated at 8,300, booked; the payment asks Claims."""
        out: dict[str, Any] = await graph.ainvoke(BEFORE_MERGE, thread(tid))
        assert (out["fraud_score"], out["repair_estimate"]) == (68, 8_300)
        assert out["booking_ref"] == "BK-CLM-1002-SHOP-7"
        asked: dict[str, Any] = out["__interrupt__"][0].value
        return asked

    def approval(self, asked: dict[str, Any], session: str = MANAGER_SESSION,
                 ) -> dict[str, Any]:
        signed = self.issuer.approve(session, str(asked["task_id"]),
                                     str(asked["claim_id"]), float(asked["amount"]),
                                     str(asked["asked"]))
        return {"approved": True, "approval": encode_approval(signed)}

    async def paid(self, graph: SagaGraph, tid: str) -> None:
        """Both approvers approve, and Finance pays 8,300."""
        asked = await self.booked_and_waiting(graph, tid)
        with pytest.raises(TaskUnfinished):  # Finance's own approver is next
            await graph.ainvoke(Command(resume=self.approval(asked)), thread(tid))
        await self.finance.approve(str(asked["task_id"]), "finance:fa-12")
        assert await wake(graph, thread(tid), set()) == "done"

    async def ledger(self, graph: SagaGraph, tid: str) -> dict[str, LedgerEntry]:
        values = (await graph.aget_state(thread(tid))).values
        found: dict[str, LedgerEntry] = values["ledger"]
        return found

    def booking(self) -> northgate.server.Booking:
        [booking] = self.northgate.executor.booked.values()
        return booking

    async def finance_state(self, task_id: str) -> str:
        from a2a.server.context import ServerCallContext
        task = await self.finance.executor.store.get(task_id, ServerCallContext())
        assert task is not None
        return str(seamkit.ledger.state_name(task.status.state))

    def cancel_bookings(self) -> int:
        """How many cancel_booking tasks Northgate's executor actually ran."""
        return sum(1 for r in self.northgate.executor.received
                   if r["skill"] == "cancel_booking")


# The incident ------------------------------------------------------------------------

async def test_the_incident_a_referral_that_undoes_nothing(clock: Clock) -> None:
    saga = Saga(clock, undoes=False)
    graph = await saga.graph()
    asked = await saga.booked_and_waiting(graph, "i13")
    clock.advance(16 * HOUR)  # the next morning, the policy merge lands
    out = await reopen(graph, thread("i13"), prior_claims=3)
    assert (out["fraud_score"], out["decision"]) == (82, "refer")  # Claims is right
    assert saga.booking().cancelled is None  # the shop still holds the bay
    assert await saga.finance_state(str(asked["task_id"])) == "input-required"


# The compensator ---------------------------------------------------------------------

async def test_the_compensator_cancels_the_payment_then_the_booking(
        clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    asked = await saga.booked_and_waiting(graph, "c13")
    clock.advance(16 * HOUR)
    sent = len(saga.records)
    out = await reopen(graph, thread("c13"), prior_claims=3)
    assert out["undone"] == {"payments": "cancelled", "repair_booking": "compensated",
                             "repair_estimate": "cost accepted",
                             "fraud_screening": "nothing to undo"}
    ledger = await saga.ledger(graph, "c13")
    assert ledger[UNDO_BOOKING]["compensates"] == ledger[BOOKING]["task_id"]
    assert ledger[BOOKING]["state"] == "completed"  # never rewritten
    assert saga.booking().cancelled == {"outcome": "cancelled", "owed": 0.0,
                                        "residue": ""}
    assert out["decision"] == "refer"
    undo = [(r.seam, r.call) for r in saga.records[sent:]
            if r.call in ("cancel", "send") and r.seam != "fraud_screening"]
    assert undo == [("payments.undo", "cancel"), (UNDO_BOOKING, "send")]  # newest first
    assert await saga.finance_state(str(asked["task_id"])) == "canceled"
    assert saga.findings == []


async def test_a_refused_approval_after_the_booking_undoes_the_booking(
        clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    asked = await saga.booked_and_waiting(graph, "r13")
    wrong = saga.approval(asked, LEAD_SESSION)  # signed, but by the team lead
    out = await graph.ainvoke(Command(resume=wrong), thread("r13"))
    assert out["aborted"].startswith("payments: PartnerFailed")
    assert out["undone"]["payments"] == "not committed"
    assert out["undone"]["repair_booking"] == "compensated"
    assert saga.finance.paid == []  # the refusal authorized nothing


# The ask can fail --------------------------------------------------------------------

async def test_parts_already_ordered_are_cancelled_with_a_residue(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "p13")
    saga.northgate.executor.shop_orders("CLM-1002", "frame")
    out = await reopen(graph, thread("p13"), prior_claims=3)
    assert out["undone"]["repair_booking"] == "residue"
    [finding] = saga.findings
    ledger = await saga.ledger(graph, "p13")
    assert (finding.seam, finding.verdict) == (BOOKING, "residue")
    assert finding.task_id == ledger[BOOKING]["task_id"]
    assert finding.compensation == ledger[UNDO_BOOKING]["task_id"]
    assert finding.authority == ledger[BOOKING]["authority"]  # the forward chain
    assert finding.detail.startswith("owed 6,500; frame ordered by SHOP-7")


@pytest.mark.parametrize(("why", "late"), [
    ("has started", lambda saga: saga.northgate.executor.shop_starts("CLM-1002")),
    ("72-hour", lambda saga: saga.clock.advance(73 * HOUR)),
])
async def test_a_refusal_goes_to_people_and_is_never_asked_again(
        clock: Clock, why: str, late: Callable[[Saga], None]) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "x13")
    late(saga)
    out = await reopen(graph, thread("x13"), prior_claims=3)
    assert out["undone"]["repair_booking"] == "refused"
    [finding] = saga.findings
    assert finding.verdict == "refused" and why in finding.detail
    assert "settled by Claims' adjuster with Northgate's network desk" in finding.detail
    assert saga.cancel_bookings() == 1  # no retry loop


async def test_a_payment_is_voided_before_the_payment_run(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.paid(graph, "v13")
    out = await reopen(graph, thread("v13"), prior_claims=3)
    assert out["undone"]["payments"] == "compensated"
    [payment] = saga.finance.paid
    assert payment["voided"] and not payment["sent"]
    assert saga.finance.payment_run() == []  # nothing left to send


async def test_after_the_payment_run_a_void_is_refused(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.paid(graph, "s13")
    clock.advance(12 * HOUR)
    assert saga.finance.payment_run() == ["PAY-CLM-1002-8300"]  # the money leaves
    out = await reopen(graph, thread("s13"), prior_claims=3)
    assert out["undone"]["payments"] == "refused"
    assert out["undone"]["repair_booking"] == "compensated"
    [finding] = saga.findings
    assert "left in the payment run" in finding.detail
    assert "Finance's recovery team, from the payee" in finding.detail


async def test_a_lost_payment_is_never_compensated(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    night = saga.finance.backup()  # last night's backup, before CLM-1002 reached Finance
    await saga.paid(graph, "l13")
    saga.finance.restore(night)  # the records go back; the payment does not
    out = await reopen(graph, thread("l13"), prior_claims=3)
    assert out["undone"]["payments"] == "lost"
    assert out["undone"]["repair_booking"] == "compensated"
    assert [f.verdict for f in saga.findings] == ["lost"]
    assert not any(r.seam == "payments.undo" and r.call == "send"
                   for r in saga.records)  # no void names a task Finance lost
    assert not saga.finance.paid[0]["voided"]


# A compensation is a remote task -----------------------------------------------------

async def test_a_lost_answer_is_resent_under_the_same_derived_key(
        clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "k13")
    saga.undo_wire.drops = 1  # Northgate cancels; the answer never reaches Claims
    out = await reopen(graph, thread("k13"), prior_claims=3)
    assert out["undone"]["repair_booking"] == "compensated"
    assert saga.cancel_bookings() == 1  # the resend got the first task back
    sends = [r for r in saga.records if r.seam == UNDO_BOOKING and r.call == "send"]
    assert len({r.idempotency_key for r in sends}) == 1 and len(sends) == 2


async def test_a_crash_after_the_send_sends_nothing_twice(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "z13")
    saga.undo_wire.crash = True  # Claims dies once Northgate has the cancel_booking
    with pytest.raises(Crash):
        await reopen(graph, thread("z13"), prior_claims=3)
    ledger = await saga.ledger(graph, "z13")
    assert ledger[UNDO_BOOKING]["state"] == "intended"  # the intent was checkpointed
    out = await graph.ainvoke(None, thread("z13"))  # the re-run
    assert out["undone"]["repair_booking"] == "compensated"
    assert saga.cancel_bookings() == 1


# Authority for an undo ---------------------------------------------------------------

async def test_a_revoked_adjuster_leaves_the_compensation_its_own_link(
        clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    asked = await saga.booked_and_waiting(graph, "a13")
    clock.advance(HOUR)
    saga.issuer.revoke(ADJUSTER, clock())  # suspended, over this very claim
    clock.advance(61)  # every partner has pulled the new list
    out = await graph.ainvoke(Command(resume=saga.approval(asked)), thread("a13"))
    assert out["aborted"].startswith("payments: DelegationRevoked")
    assert out["undone"]["repair_booking"] == "compensated"
    accepted = [e for e in saga.northgate.guard.log if e["skill"] == "cancel_booking"]
    assert [(e["subject"], e["refused"]) for e in accepted] == [(COMPENSATOR, "")]
    # Her chain can still do one thing, cancel a task bound to her; a read is refused,
    # and it could never have asked for an undo.
    dead = saga.issuer.cancel_root(ADJUSTER)
    link = mint(dead, KEYS["claims"], "claims", "northgate", GRANTS[BOOKING],
                frozenset({"CLM-1002"}))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(
        app=as_peer(saga.northgate.app, "claims")), base_url=NORTHGATE_URL)
    assert "revoked" in await probe(http, (*dead, link))
    undo = Grant(frozenset({"cancel_booking"}), frozenset({"claim_id", "task"}), 0)
    with pytest.raises(DelegationRefused, match="widens the skills"):
        mint(dead, KEYS["claims"], "claims", "northgate", undo, frozenset({"CLM-1002"}))


async def test_a_compensation_link_names_one_task(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "n13")
    booking = (await saga.ledger(graph, "n13"))[BOOKING]["task_id"]
    node = saga.undoer.undo[BOOKING]
    with as_run(thread("n13")), run_values(compensating="CLM-1002", undoes=booking):
        with pytest.raises(PartnerRejected, match="task some-other-task not granted"):
            await node._send({"claim_id": "CLM-1002", "task": "some-other-task"})
    assert saga.booking().cancelled is None


# Orchestrated, not choreographed -----------------------------------------------------

def cancel_booking_rpc(booking: str) -> bytes:
    """The cancel_booking a partner other than Claims would send Northgate."""
    return json.dumps({"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": {
        "message": {"messageId": uuid.uuid4().hex, "role": "ROLE_USER",
                    "parts": [{"data": {"claim_id": "CLM-1002", "task": booking}}],
                    "metadata": {"skill": "cancel_booking"}}}}).encode()


async def test_risk_cannot_undo_a_booking_claims_made(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.booked_and_waiting(graph, "o13")
    booking = (await saga.ledger(graph, "o13"))[BOOKING]["task_id"]
    with as_run(thread("o13")):  # the chain Claims really gives Risk
        to_risk: Chain = (*saga.issuer.run_chain(),)
    risk_link = mint(to_risk, KEYS["claims"], "claims", "risk", GRANTS["fraud_screening"],
                     frozenset({"CLM-1002"}))
    undo = Grant(frozenset({"cancel_booking"}), frozenset({"claim_id", "task"}), 0,
                 frozenset({booking}))
    with pytest.raises(DelegationRefused, match="widens the skills"):
        mint((*to_risk, risk_link), KEYS["risk"], "risk", "northgate", undo,
             frozenset({"CLM-1002"}))
    # The compensation root has one hop, so Claims cannot pass an undo on either.
    with as_run(thread("o13")), run_values(compensating="CLM-1002"):
        root = saga.issuer.compensation_chain()
    with pytest.raises(DelegationRefused, match="does not spend a hop"):
        mint(root, KEYS["claims"], "claims", "risk", Grant(undo.skills, undo.fields, 1),
             frozenset({"CLM-1002"}))
    # Even a root wide enough to reach Risk is refused at Northgate: the booking is
    # Claims', and only the workload that sent a task may undo it.
    root = saga.issuer._root(COMPENSATOR, frozenset({"CLM-1002"}), clock() + 600,
                             Grant(undo.skills, undo.fields, 2))
    wide = mint(root, KEYS["claims"], "claims", "risk",
                Grant(undo.skills, undo.fields, 1), frozenset({"CLM-1002"}))
    chain = (*root, wide, mint((*root, wide), KEYS["risk"], "risk", "northgate", undo,
                               frozenset({"CLM-1002"})))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(
        app=as_peer(saga.northgate.app, "risk")), base_url=NORTHGATE_URL)
    response = await http.post("/", content=cancel_booking_rpc(booking), headers={
        HEADER: encode(chain), "A2A-Version": "1.0", "content-type": "application/json"})
    assert response.status_code == 403
    assert f"task {booking} was not sent by risk" in response.json()["error"]["message"]
    assert saga.booking().cancelled is None
    # The same news, rung to Claims as a doorbell: Claims re-screens and orchestrates.
    out = await reopen(graph, thread("o13"), prior_claims=3)
    assert out["undone"]["repair_booking"] == "compensated"


# The plan is checked against the cards -----------------------------------------------

async def test_a_booking_without_a_declared_undo_is_never_made(
        clock: Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    bare = replace(NORTHGATE_V7_4, skills={
        **NORTHGATE_V7_4.skills,
        "book_repair": replace(NORTHGATE_V7_4.skills["book_repair"],
                               compensated_by="", compensate_seconds=None)})
    monkeypatch.setitem(northgate.server.TERMS, "v7.4", bare)
    saga = Saga(clock)
    graph = await saga.graph()
    [breach] = saga.nodes[BOOKING].partner.breaches
    assert breach.detail == "book_repair is not undone by cancel_booking"
    with pytest.raises(ContractBreach):
        await graph.ainvoke(BEFORE_MERGE, thread("b13"))
    assert saga.northgate.executor.booked == {}  # held before anything crossed


def test_the_plan_needs_the_window_the_partner_promises() -> None:
    @dataclass
    class Held:
        terms: tuple[Any, ...]

    short = replace(NORTHGATE_V7_4, skills={
        **NORTHGATE_V7_4.skills,
        "book_repair": replace(NORTHGATE_V7_4.skills["book_repair"],
                               compensate_seconds=3600)})
    [breach] = check_plan(MERIDIAN_PLAN, {BOOKING: Held((short,))})  # type: ignore
    assert breach.detail == ("cancel_booking is honoured 3600 s, and the plan "
                             "needs 259200 s")
    assert check_plan(MERIDIAN_PLAN, {BOOKING: Held((NORTHGATE_V7_4,))}) == []  # type: ignore


# Chapter 12's orphan -----------------------------------------------------------------

async def test_an_orphan_is_undone_by_the_claims_own_thread(clock: Clock) -> None:
    saga = Saga(clock)
    graph = await saga.graph()
    await saga.paid(graph, "g13")
    payments: RemoteAgentNode = saga.nodes[PAY]
    with as_run(thread("g13")):  # a second payment task no entry names
        orphan = await payments._send({"claim_id": "CLM-1002", "amount": 14_500.0,
                                       "repair_estimate": 8_300.0},
                                      key=uuid.uuid4().hex)
    job = reconciler(graph, saga, "g13")
    [finding] = await job.run()
    assert (finding.verdict, finding.task_id) == ("orphan", orphan.id)
    out = await adopt(graph, thread("g13"), finding)  # the thread undoes it, not the job
    assert out["undone"][f"payments#{orphan.id[:8]}"] == "cancelled"
    assert out["decision"] == "paid"  # the claim itself stands
    assert await saga.finance_state(orphan.id) == "canceled"
    assert [p["voided"] for p in saga.finance.paid] == [False]
    assert await job.run() == []  # named now: the job finds nothing


def reconciler(graph: SagaGraph, saga: Saga, *tids: str) -> Reconciler:
    configs = {tid: thread(tid) for tid in tids}
    queue = WakeQueue(graph, lambda tid: configs[tid])
    own: RunnableConfig = {"configurable": {"thread_id": "reconcile",
                                            "session": RECONCILER_SESSION}}
    return Reconciler(queue, lambda: configs, {PAY: saga.nodes[PAY]}, own,
                      log=lambda f: None)


def test_the_plan_has_a_row_for_every_step_that_commits() -> None:
    assert [(s.seam, s.compensated_by or s.settles) for s in MERIDIAN_PLAN.steps] == [
        ("fraud_screening", "nothing to undo"), ("repair_estimate", "cost accepted"),
        ("repair_booking", "cancel_booking"), ("payments", "void_payment")]
