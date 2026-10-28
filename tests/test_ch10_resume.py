"""Interrupt and resume across owners: the incident, the routing, the approval, the
cross-owner resume state machine, six days on one clock, and a fork that pays twice."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from a2a.helpers import new_data_part
from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, Task, TaskState, TaskStatus
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import seamkit.ledger
from meridian.drill import T0
from meridian.federated import EXPECTATIONS, SEAMS, A2APartners, a2a_workers
from meridian.finance_server import FINANCE_TERMS_2026_12, FinancePartner
from meridian.identity import (
    LEAD_SESSION,
    MANAGER,
    MANAGER_SESSION,
    MeridianIssuer,
    revocation_sources,
)
from meridian.inprocess import ClaimGraph
from meridian.risk_server import RiskPartner
from meridian.samples import clean_claim
from northgate.server import NorthgatePartner
from seamkit.approval import encode_approval
from seamkit.contract import NO_CROSSING, check
from seamkit.drill import Clock
from seamkit.errors import (
    ContractBreach,
    PartnerFailed,
    PartnerNeedsInput,
    TaskUnfinished,
    WaitExpired,
)
from seamkit.ledger import TERMINAL, LedgerEntry, intended, question_digest
from seamkit.resume import PAUSED, TRANSITIONS, Held, fork, resume_step, sweep, wake
from seamkit.server import StoreWriter
from tests.wire import Federation, thread

pytestmark = pytest.mark.anyio

PAY = "payments"
DAY = 24 * 3600.0

# The incident's Claims: the payments seam answers "approval", and names nobody.
NAMES_NOBODY = {**SEAMS, PAY: replace(SEAMS[PAY], roles=())}


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    """One clock for every owner: Meridian's issuer, Finance, and Claims' ledger."""
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


@dataclass
class TwoParty:
    """Meridian on one clock, with Finance on its two-party release."""

    clock: Clock
    addressed: bool = True
    seams: dict[str, Any] = field(default_factory=lambda: dict(SEAMS))

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        self.finance = FinancePartner(release="2026.12", addressed=self.addressed,
                                      revocations=revocation_sources(self.issuer),
                                      now=self.clock)
        lists = revocation_sources(self.issuer)
        partners = A2APartners(risk=RiskPartner(revocations=lists, now=self.clock),
                               northgate=NorthgatePartner(revocations=lists,
                                                          now=self.clock),
                               finance=self.finance)
        self.fed = Federation(partners, seams=self.seams, issuer=self.issuer)
        self.saver = InMemorySaver()

    async def graph(self) -> ClaimGraph:
        return await self.fed.graph(checkpointer=self.saver)

    async def entry(self, graph: ClaimGraph, tid: str) -> LedgerEntry:
        entry: LedgerEntry = (await graph.aget_state(thread(tid))).values["ledger"][PAY]
        return entry

    def approval(self, session: str, asked: dict[str, Any]) -> dict[str, Any]:
        """What the approver's screen sends back: a yes, and the issuer's signature."""
        signed = self.issuer.approve(session, str(asked["task_id"]),
                                     str(asked["claim_id"]), float(asked["amount"]),
                                     str(asked["asked"]))
        return {"approved": True, "approval": encode_approval(signed)}


def asked_of(out: dict[str, Any]) -> dict[str, Any]:
    question: dict[str, Any] = out["__interrupt__"][0].value
    return question


# The incident -------------------------------------------------------------------------

async def test_the_incident_two_claims_people_approve_and_nobody_at_finance(
        clock: Clock) -> None:
    meridian = TwoParty(clock, addressed=False, seams=NAMES_NOBODY)
    graph = await meridian.graph()
    out = await graph.ainvoke(clean_claim(), thread("i10"))
    first = asked_of(out)  # Finance's question for a Claims approver
    out = await graph.ainvoke(Command(resume={"approved": True}), thread("i10"))
    second = asked_of(out)  # Finance's question for its own approver, in Claims' queue
    assert (first["needs"], second["needs"]) == ("approval", "approval")
    assert "to" not in first and "to" not in second  # neither names who must answer
    clock.advance(2 * DAY)  # day two: a Claims team lead clears the backlog
    out = await graph.ainvoke(Command(resume={"approved": True}), thread("i10"))
    assert out["decision"] == "paid" and out["payment_ref"] == "PAY-CLM-1001-3000"
    [payment] = meridian.finance.paid
    assert payment["approvers"] == ["claims (unverified)", "finance (unverified)"]
    assert meridian.finance.executor.waiting == {}  # Finance's approver: never asked


# Addressed interrupts and routing by addressee ----------------------------------------

async def test_a_question_for_finances_people_never_reaches_claims(
        clock: Clock) -> None:
    meridian = TwoParty(clock)
    graph = await meridian.graph()
    out = await graph.ainvoke(clean_claim(), thread("a1"))
    asked = asked_of(out)
    assert asked["to"] == "claims/claims_manager"
    with pytest.raises(TaskUnfinished):  # Finance asks its own approver, inside
        await graph.ainvoke(Command(resume=meridian.approval(MANAGER_SESSION, asked)),
                            thread("a1"))
    entry = await meridian.entry(graph, "a1")
    assert (entry["state"], entry["waiting_on"]) == ("working",
                                                     "finance/payments_approver")
    assert not (await graph.aget_state(thread("a1"))).interrupts  # no Claims queue
    await meridian.finance.approve(entry["task_id"], "finance:fa-12")
    assert await wake(graph, thread("a1"), set()) == "done"
    out = (await graph.aget_state(thread("a1"))).values
    assert out["decision"] == "paid"
    assert meridian.finance.paid[0]["approvers"] == [MANAGER, "finance:fa-12"]


async def test_claims_refuses_a_question_addressed_to_someone_it_does_not_answer_for(
        clock: Clock) -> None:
    meridian = TwoParty(clock)
    meridian.finance.executor.own = ""  # Finance addresses, but every question crosses
    graph = await meridian.graph()
    out = await graph.ainvoke(clean_claim(), thread("r1"))
    asked = asked_of(out)
    with pytest.raises(PartnerNeedsInput):  # finance/payments_approver: not ours
        await graph.ainvoke(Command(resume=meridian.approval(MANAGER_SESSION, asked)),
                            thread("r1"))
    entry = await meridian.entry(graph, "r1")
    assert entry["waiting_on"] == "finance/payments_approver"
    assert not (await graph.aget_state(thread("r1"))).interrupts


async def test_a_seam_that_names_roles_refuses_a_question_that_names_nobody(
        clock: Clock) -> None:
    meridian = TwoParty(clock, addressed=False)  # bare interrupts, HEAD's Claims
    graph = await meridian.graph()
    with pytest.raises(PartnerNeedsInput):
        await graph.ainvoke(clean_claim(), thread("r2"))
    assert not (await graph.aget_state(thread("r2"))).interrupts


# An approval is a credential ----------------------------------------------------------

@pytest.mark.parametrize("who, change", [
    ("bare", {}),  # Chapter 3's answer: a value anyone can type
    ("lead", {}),  # signed, but by a team lead: the wrong role
    ("manager", {"amount": 2_800.0}),  # signed for another amount
])
async def test_finance_refuses_an_answer_that_does_not_prove_who_approved(
        clock: Clock, who: str, change: dict[str, Any]) -> None:
    meridian = TwoParty(clock)
    graph = await meridian.graph()
    asked = {**asked_of(await graph.ainvoke(clean_claim(), thread("c1"))), **change}
    answer = ({"approved": True} if who == "bare" else meridian.approval(
        LEAD_SESSION if who == "lead" else MANAGER_SESSION, asked))
    with pytest.raises(PartnerFailed):
        await graph.ainvoke(Command(resume=answer), thread("c1"))
    assert meridian.finance.paid == []


# The cross-owner resume state machine -------------------------------------------------

def paused_task(state: int, said: dict[str, Any] | None = None) -> Task:
    message = Message(message_id="m", role=Role.ROLE_AGENT,
                      parts=[new_data_part(said)] if said else [])
    return Task(id="t-1", context_id="c-1",
                status=TaskStatus(state=state,  # type: ignore[arg-type]
                                  message=message))


def recorded(task: Task, **changes: Any) -> LedgerEntry:
    entry = intended(PAY, "pay_claim", "sha256:card", "chain:x", "2026.12")
    base = {**entry, "task_id": task.id, "state": "working", "terms": "sha256:terms",
            "since": T0}
    return {**base, **changes}  # type: ignore[typeddict-item]


HELD = Held("sha256:card", "sha256:terms", "", ("claims/claims_manager",), True,
            72 * 3600)
FOR_US = {"needs": "approval", "to": "claims/claims_manager"}
FOR_THEM = {"needs": "approval", "to": "finance/payments_approver"}
WAITING = {"waiting_on": "finance/payments_approver"}
S = TaskState


@pytest.mark.parametrize("state, said, step", [
    (S.TASK_STATE_INPUT_REQUIRED, FOR_US, "ask"),
    (S.TASK_STATE_INPUT_REQUIRED, FOR_THEM, "refuse"),
    (S.TASK_STATE_INPUT_REQUIRED, {"needs": "approval"}, "refuse"),
    (S.TASK_STATE_WORKING, WAITING, "wait"),
    (S.TASK_STATE_WORKING, None, "wait"),
    (S.TASK_STATE_SUBMITTED, None, "wait"),
    (S.TASK_STATE_AUTH_REQUIRED, {"needs": "delegation"}, "renew"),
    (S.TASK_STATE_COMPLETED, None, "read"),
])
def test_every_paused_case_has_one_step(state: int, said: dict[str, Any] | None,
                                        step: str) -> None:
    task = paused_task(state, said)
    asks = said and state == S.TASK_STATE_INPUT_REQUIRED
    asked = question_digest(said) if said and asks else ""
    assert resume_step(recorded(task, asked=asked), task, HELD, T0) == step


def test_the_rechecks_run_in_order_and_each_one_wins() -> None:
    task = paused_task(S.TASK_STATE_INPUT_REQUIRED, FOR_US)
    now = recorded(task, asked=question_digest(FOR_US))
    revoked = replace(HELD, revoked="adjuster:adj-117 revoked", card="sha256:new")
    assert resume_step(now, task, revoked, T0) == "cancel"  # authority before card
    new_card = replace(HELD, card="sha256:new")
    assert resume_step(now, task, new_card, T0) == "ask"  # same terms: continue
    assert resume_step(now, task, replace(new_card, terms="sha256:other"),
                       T0) == "hold"
    assert resume_step({**now, "asked": "sha256:old"}, task, HELD, T0) == "reask"
    waiting = paused_task(S.TASK_STATE_WORKING, WAITING)
    assert resume_step(recorded(waiting), waiting, HELD, T0 + 4 * DAY) == "expire"
    before = {k: v for k, v in recorded(waiting).items()
              if k not in ("waiting_on", "since", "asked", "terms")}
    # An entry an earlier release wrote: same card, so it waits; a new card holds it,
    # because nothing recorded proves the terms were the same.
    assert resume_step(before, waiting, HELD, T0) == "wait"  # type: ignore[arg-type]
    assert resume_step(before, waiting, new_card, T0) == "hold"  # type: ignore


def test_the_table_names_every_paused_state_and_only_paused_ones() -> None:
    assert {state for state, _ in TRANSITIONS} == PAUSED
    assert not PAUSED & TERMINAL


# The wait horizon, and six days on one clock ------------------------------------------

def test_the_contract_holds_the_wait_horizon() -> None:
    claims = EXPECTATIONS[PAY]
    assert claims.wait_seconds == 72 * 3600
    assert not [b for b in check(FINANCE_TERMS_2026_12, claims, NO_CROSSING)
                if "wait" in b.detail]  # 48 h at Finance, inside Claims' 72 h
    slow = replace(FINANCE_TERMS_2026_12, skills={"pay_claim": replace(
        FINANCE_TERMS_2026_12.skills["pay_claim"], wait_seconds=96 * 3600)})
    found = check(slow, claims, NO_CROSSING)
    assert any("may wait 345600 s" in b.detail for b in found)
    silent = replace(FINANCE_TERMS_2026_12, skills={"pay_claim": replace(
        FINANCE_TERMS_2026_12.skills["pay_claim"], wait_seconds=None)})
    assert any("no wait horizon" in b.detail
               for b in check(silent, claims, NO_CROSSING))


async def start_and_approve(meridian: TwoParty, tid: str) -> ClaimGraph:
    """Day 0: Claims sends the payment, and its claims manager approves."""
    graph = await meridian.graph()
    asked = asked_of(await graph.ainvoke(clean_claim(), thread(tid)))
    with pytest.raises(TaskUnfinished):
        await graph.ainvoke(Command(resume=meridian.approval(MANAGER_SESSION, asked)),
                            thread(tid))
    return graph


async def test_six_days_the_chain_expires_and_the_resume_renews_it(
        clock: Clock) -> None:
    meridian = TwoParty(clock)
    graph = await start_and_approve(meridian, "d6")
    entry = await meridian.entry(graph, "d6")
    threads = {"d6": thread("d6")}
    nodes = a2a_workers(meridian.fed.connections, issuer=meridian.issuer)
    assert await sweep(graph, threads, nodes, set()) == {}  # nothing moved: no run
    clock.advance(2 * DAY)  # 48 h: Finance reassigns; its alternate approver says yes
    # Finance's commit act point: the adjuster's 15-minute chain expired on day 0.
    why = await meridian.finance.approve(entry["task_id"], "finance:fa-31")
    assert why == "expired"  # auth-required, the alternate's answer held
    woken = await sweep(graph, threads, nodes, set())
    assert woken == {"d6": "done"}  # read auth-required, renewed, read completed
    out = (await graph.aget_state(thread("d6"))).values
    assert (out["decision"], out["payment_ref"]) == ("paid", "PAY-CLM-1001-3000")
    assert meridian.finance.paid[0]["approvers"] == [MANAGER, "finance:fa-31"]


async def test_a_wait_past_claims_horizon_cancels_the_task(clock: Clock) -> None:
    meridian = TwoParty(clock)
    graph = await start_and_approve(meridian, "h1")
    entry = await meridian.entry(graph, "h1")
    clock.advance(4 * DAY)  # nobody at Finance answered; Claims waits 72 h
    assert await wake(graph, thread("h1"), set()) == WaitExpired.__name__
    task = await meridian.finance.executor.store.get(entry["task_id"], context())
    assert task is not None and task.status.state == S.TASK_STATE_CANCELED
    assert meridian.finance.paid == []


def context() -> ServerCallContext:
    return ServerCallContext()


# What changed while the person was away -----------------------------------------------

@pytest.mark.parametrize("release, ends", [("2026.12.1", TaskUnfinished),
                                           ("2026.13", ContractBreach)])
async def test_a_card_shipped_while_the_question_waited(
        clock: Clock, release: str, ends: type[Exception]) -> None:
    meridian = TwoParty(clock)
    graph = await meridian.graph()
    asked = asked_of(await graph.ainvoke(clean_claim(), thread("k1")))
    meridian.finance.ship(release)  # while the claims manager has the question open
    graph = await meridian.graph()  # Claims reconnects and holds the new card
    with pytest.raises(ends):
        await graph.ainvoke(Command(resume=meridian.approval(MANAGER_SESSION, asked)),
                            thread("k1"))
    answered = [r for r in meridian.finance.executor.received if r["resumed"]]
    assert len(answered) == (1 if release == "2026.12.1" else 0)


async def test_an_answer_to_the_old_question_is_never_sent(clock: Clock) -> None:
    meridian = TwoParty(clock)
    graph = await meridian.graph()
    asked = asked_of(await graph.ainvoke(clean_claim(), thread("q1")))
    answer = meridian.approval(MANAGER_SESSION, asked)  # approved, not yet sent
    revised = {"needs": "approval", "claim_id": "CLM-1001", "amount": 2_800.0,
               "to": "claims/claims_manager"}  # Finance re-asks after a new estimate
    task_id = str(asked["task_id"])
    task = await meridian.finance.executor.store.get(task_id, context())
    assert task is not None
    writer = StoreWriter(meridian.finance.executor.store, task_id, task.context_id)
    updater = TaskUpdater(writer, task_id, task.context_id)
    await updater.requires_input(updater.new_agent_message([new_data_part(revised)]))
    out = await graph.ainvoke(Command(resume=answer), thread("q1"))
    assert asked_of(out)["amount"] == 2_800.0  # asked again, with the new amount
    sent = [r for r in meridian.finance.executor.received if r["resumed"]]
    assert sent == []  # the approval of 3,000 never left Claims


# Time travel stops at the seam --------------------------------------------------------

async def paid_thread(fed: Federation, tid: str) -> tuple[ClaimGraph, str]:
    """CLM-1001 settled on Finance 2026.11, and the checkpoint before its payment."""
    graph = await fed.graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(clean_claim(), thread(tid))
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    async for snapshot in graph.aget_state_history(thread(tid)):
        if PAY not in snapshot.values.get("ledger", {}) and snapshot.next == (PAY,):
            return graph, str(snapshot.config["configurable"]["checkpoint_id"])
    raise AssertionError("no checkpoint before the payment")


def started(fed: Federation) -> int:
    return len([r for r in fed.partners.finance.executor.received if not r["resumed"]])


async def test_a_fork_from_before_the_payment_pays_again() -> None:
    fed = Federation()
    graph, before = await paid_thread(fed, "f1")
    past: RunnableConfig = {"configurable": {**thread("f1")["configurable"],
                                             "checkpoint_ns": "",
                                             "checkpoint_id": before}}
    out = await graph.ainvoke(None, past)  # an audit replay from before the payment
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    assert started(fed) == 2  # Finance was sent pay_claim twice, and paid twice


async def test_the_fork_guard_carries_the_ledger_and_reads_the_task() -> None:
    fed = Federation()
    graph, before = await paid_thread(fed, "f2")
    forked = await fork(graph, thread("f2"), before)
    out = await graph.ainvoke(None, forked)
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    assert started(fed) == 1  # the fork read the completed task instead


async def test_a_wake_never_writes_into_a_thread_with_a_run_in_flight(
        clock: Clock) -> None:
    meridian = TwoParty(clock)
    graph = await start_and_approve(meridian, "w1")
    before = await meridian.entry(graph, "w1")
    assert await wake(graph, thread("w1"), {"w1"}) == "busy"
    assert await meridian.entry(graph, "w1") == before


def test_an_entry_from_before_this_chapter_still_loads() -> None:
    old = {k: v for k, v in intended(PAY, "pay_claim", "sha256:card", "chain:x",
                                     "2026.11").items()}
    assert set(LedgerEntry.__required_keys__) <= set(old)
    assert {"waiting_on", "asked", "since", "terms"} <= set(
        LedgerEntry.__optional_keys__)


def test_a_reassignment_at_the_partner_does_not_restart_the_wait() -> None:
    first = paused_task(S.TASK_STATE_WORKING, WAITING)
    entry = seamkit.ledger.observe(recorded(first), first, "2026.12")
    other = paused_task(S.TASK_STATE_WORKING,
                        {"waiting_on": "finance/alternate_approver"})
    later = seamkit.ledger.observe(entry, other, "2026.12")
    assert later["waiting_on"] == "finance/alternate_approver"
    assert later["since"] == entry["since"] == T0
