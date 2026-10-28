"""Retries, idempotency, and reconciliation: the payment Finance made twice, the
partner's half of a key, the seam retry policy, onward keys and the loop rule, and the
reconciliation job in both directions."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field, fields, replace
from itertools import pairwise
from typing import Any, TypedDict

import httpx
import pytest
from a2a.auth.user import User
from a2a.helpers import new_data_message
from a2a.server.context import ServerCallContext
from a2a.types import ListTasksResponse, Role, SendMessageRequest, Task, TaskState
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.types import RetryPolicy as NodeRetryPolicy
from starlette.types import ASGIApp

import seamkit.ledger
from meridian.drill import T0
from meridian.federated import (
    EXPECTATIONS,
    A2APartners,
    a2a_workers,
    connect_partners,
    with_review,
)
from meridian.finance_server import (
    FINANCE_TERMS_2026_13,
    FINANCE_TERMS_2026_14,
    FINANCE_URL,
    FinancePartner,
)
from meridian.identity import (
    ISSUER,
    KEYS,
    MANAGER_SESSION,
    MERIDIAN,
    RECONCILER_SESSION,
    MeridianIssuer,
    revocation_sources,
)
from meridian.incident import ResendOnUnknown
from meridian.inprocess import ClaimGraph, build_graph
from meridian.review import keyword_reader, reviewer
from meridian.risk_server import RISK_URL, RiskPartner
from meridian.samples import clean_claim
from northgate.server import NorthgatePartner
from seamkit.approval import encode_approval
from seamkit.contract import NO_CROSSING, check
from seamkit.delegation import Link, Trust, as_peer, digest, narrows, sign, verify
from seamkit.drill import Clock
from seamkit.errors import (
    ContractBreach,
    DelegationRefused,
    DelegationRevoked,
    MalformedAnswer,
    PartnerRejected,
    PartnerUnavailable,
    TaskUnfinished,
)
from seamkit.ledger import LedgerEntry
from seamkit.push import WakeQueue
from seamkit.resume import wake
from seamkit.reconcile import Finding, Reconciler
from seamkit.remote import RemoteAgentNode
from seamkit.retry import RetryPolicy, backoff, onward_key, refuse_retry
from seamkit.server import KeyedHandler
from tests.wire import as_run, new_tasks, thread

pytestmark = pytest.mark.anyio

PAY = "payments"
DAY = 24 * 3600.0
POLICY = EXPECTATIONS[PAY].retry  # three attempts, a ten-minute deadline


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    """One clock for every owner: Meridian's issuer, the partners, and the ledger."""
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


class Crash(BaseException):
    """Claims' process dies: nothing it was doing gets to write a checkpoint."""


@dataclass
class Wire(httpx.AsyncBaseTransport):
    """An in-process transport: a send reaches the partner, and its answer may be lost.

    `drops` loses that many answers to a send that opens a task, as a deploy that
    drains connections does; `down` refuses every call; `crash` kills Claims once the
    partner has the send.
    """

    app: ASGIApp
    drops: int = 0
    down: bool = False
    crash: bool = False
    down_after_drop: bool = False  # the deploy outlasts the lost answer

    def __post_init__(self) -> None:
        self.inner = httpx.ASGITransport(app=as_peer(self.app, "claims"))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        response = await self.inner.handle_async_request(request)
        body = request.content
        opens = b'"SendMessage"' in body and b'"taskId"' not in body
        if opens and self.crash:
            self.crash = False
            raise Crash("Claims died after the send, before its checkpoint")
        if opens and self.drops:
            self.drops -= 1
            self.down = self.down_after_drop
            raise httpx.ReadTimeout("the answer was lost", request=request)
        return response

    def http(self, url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self, base_url=url)


async def no_wait(seconds: float) -> None:
    """A retry's backoff, in a test: nothing waits, and the clock is untouched."""


@dataclass
class Meridian:
    """Claims, Risk, Northgate and Finance on one clock, Finance on `release`."""

    clock: Clock
    release: str = "2026.14"
    northgate_release: str = "v7"
    assistant: Any = None
    saver: InMemorySaver = field(default_factory=InMemorySaver)

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        lists = revocation_sources(self.issuer)
        self.finance = FinancePartner(release=self.release, now=self.clock,
                                      revocations=revocation_sources(self.issuer))
        self.northgate = NorthgatePartner(release=self.northgate_release,
                                          revocations=lists, now=self.clock)
        self.risk = RiskPartner(revocations=lists, now=self.clock,
                                assistant=self.assistant)
        self.partners = A2APartners(risk=self.risk, northgate=self.northgate,
                                    finance=self.finance)
        self.pay = Wire(self.finance.app)
        self.screen = Wire(self.risk.app)

    async def graph(self, kind: type[RemoteAgentNode] | None = None) -> ClaimGraph:
        """HEAD's Claims graph; `kind` replaces the payments node (the incident's)."""
        self.connections = await connect_partners(
            self.partners.apps(), {PAY: self.pay.http(FINANCE_URL),
                                   "fraud_screening": self.screen.http(RISK_URL)},
            issuer=self.issuer)
        self.nodes = a2a_workers(self.connections, issuer=self.issuer)
        for node in self.nodes.values():
            node.sleep = no_wait
        if kind is not None:
            gate = self.nodes[PAY]
            self.nodes[PAY] = kind(**{f.name: getattr(gate, f.name)
                                      for f in fields(gate)})
        return build_graph(with_review(self.nodes, reviewer(keyword_reader)),
                           self.saver)

    async def entry(self, graph: ClaimGraph, tid: str, seam: str = PAY) -> LedgerEntry:
        values = (await graph.aget_state(thread(tid))).values
        entry: LedgerEntry = values["ledger"][seam]
        return entry

    def approval(self, asked: dict[str, Any]) -> dict[str, Any]:
        signed = self.issuer.approve(MANAGER_SESSION, str(asked["task_id"]),
                                     str(asked["claim_id"]), float(asked["amount"]),
                                     str(asked["asked"]))
        return {"approved": True, "approval": encode_approval(signed)}

    async def approve_at_claims(self, graph: ClaimGraph, tid: str) -> None:
        """Finance asks Claims' claims manager, she approves, and Finance's is next."""
        out = await graph.ainvoke(clean_claim(), thread(tid))
        asked = out["__interrupt__"][0].value
        with pytest.raises(TaskUnfinished):
            await graph.ainvoke(Command(resume=self.approval(asked)), thread(tid))

    def payment_tasks(self) -> list[str]:
        received = self.finance.executor.received
        return [r["task_id"] for r in received if not r["resumed"]]

    def reconciler(self, graph: ClaimGraph, *tids: str) -> Reconciler:
        configs = {tid: thread(tid) for tid in tids}
        queue = WakeQueue(graph, lambda tid: configs[tid])
        own: RunnableConfig = {"configurable": {"thread_id": "reconcile",
                                                "session": RECONCILER_SESSION}}
        self.findings: list[Finding] = []
        return Reconciler(queue, lambda: configs, self.nodes, own,
                          log=self.findings.append)


# The incident ------------------------------------------------------------------------

async def test_the_incident_a_resend_under_an_unhonoured_key_pays_twice(
        clock: Clock) -> None:
    meridian = Meridian(clock, release="2026.11")  # before two-party approval
    graph = await meridian.graph(kind=ResendOnUnknown)
    meridian.pay.drops = 1  # 02:10, Finance's deploy: the answer to the send is lost
    out = await graph.ainvoke(clean_claim(), thread("i12"))
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    first, second = meridian.payment_tasks()  # task A, then task B
    assert [p["payment_ref"] for p in meridian.finance.paid] == [
        "PAY-CLM-1001-3000", "PAY-CLM-1001-3000"]  # Finance paid 3,000 twice
    entry = await meridian.entry(graph, "i12")
    assert (entry["task_id"], entry["state"]) == (second, "completed")  # one entry: B


async def test_on_the_two_party_release_the_second_task_waits_unseen(
        clock: Clock) -> None:
    meridian = Meridian(clock, release="2026.13")
    graph = await meridian.graph(kind=ResendOnUnknown)
    meridian.pay.drops = 1
    await meridian.approve_at_claims(graph, "w12")  # Claims approves task B
    first, second = meridian.payment_tasks()
    assert (await meridian.entry(graph, "w12"))["task_id"] == second
    task = await meridian.finance.executor.store.get(first, ServerCallContext())
    assert task is not None  # A waits for a Claims approval no Claims person can see
    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED


# The partner's half ------------------------------------------------------------------

async def test_finance_honours_the_key_and_the_resend_gets_task_a(
        clock: Clock) -> None:
    meridian = Meridian(clock)  # Finance 2026.14: a 24-hour key window on pay_claim
    graph = await meridian.graph(kind=ResendOnUnknown)  # even the incident's resend
    meridian.pay.drops = 1
    await meridian.approve_at_claims(graph, "k12")
    [only] = meridian.payment_tasks()  # one task: the repeat ran nothing
    assert (await meridian.entry(graph, "k12"))["task_id"] == only


class Named(User):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.name


def payment(key: str, amount: float = 2_000.0) -> SendMessageRequest:
    message = new_data_message({"claim_id": "CLM-1001", "amount": amount,
                                "repair_estimate": amount}, role=Role.ROLE_USER)
    message.message_id = key
    message.metadata.update({"skill": "pay_claim"})
    return SendMessageRequest(message=message)


def handler_of(finance: FinancePartner) -> KeyedHandler:
    app: Any = finance.guard.app
    handler: KeyedHandler = app.state.handler
    return handler


async def test_a_key_is_scoped_to_its_caller_and_to_its_request(clock: Clock) -> None:
    finance = FinancePartner(release="2026.14", now=clock)
    handler, claims = handler_of(finance), ServerCallContext(user=Named("claims"))
    first = await handler.on_message_send(payment("k-1"), claims)
    again = await handler.on_message_send(payment("k-1"), claims)
    assert isinstance(first, Task) and isinstance(again, Task) and again.id == first.id
    other = await handler.on_message_send(payment("k-1"),
                                          ServerCallContext(user=Named("risk")))
    assert isinstance(other, Task) and other.id != first.id  # another caller's key
    with pytest.raises(Exception, match="different request"):  # corrected amount
        await handler.on_message_send(payment("k-1", 1_800.0), claims)
    clock.advance(DAY)  # past the window: a repeat is a first send again
    later = await handler.on_message_send(payment("k-1"), claims)
    assert isinstance(later, Task) and later.id != first.id
    assert [p["payment_ref"] for p in finance.paid] == ["PAY-CLM-1001-2000"] * 3


async def test_a_concurrent_repeat_waits_for_the_original(clock: Clock) -> None:
    finance = FinancePartner(release="2026.14", now=clock)
    handler, claims = handler_of(finance), ServerCallContext(user=Named("claims"))
    one, two = await asyncio.gather(handler.on_message_send(payment("k-2"), claims),
                                    handler.on_message_send(payment("k-2"), claims))
    assert isinstance(one, Task) and isinstance(two, Task) and one.id == two.id
    assert len(finance.paid) == 1


async def test_the_caller_reads_a_mismatch_as_a_rejection(clock: Clock) -> None:
    meridian = Meridian(clock)
    graph = await meridian.graph()
    await meridian.approve_at_claims(graph, "m12")
    node = meridian.nodes[PAY]
    entry = await meridian.entry(graph, "m12")
    with as_run(thread("m12")), pytest.raises(PartnerRejected):
        await node._send({"claim_id": "CLM-1001", "amount": 2_800.0,
                          "repair_estimate": 2_800.0}, key=entry["idempotency_key"])


# The caller's half: the seam retry policy ---------------------------------------------

def test_the_contract_holds_the_retry_deadline_inside_the_key_window() -> None:
    claims = EXPECTATIONS[PAY]
    assert (claims.retry.attempts, claims.retry.deadline_seconds) == (3, 600)
    assert not [b for b in check(FINANCE_TERMS_2026_14, claims, NO_CROSSING)
                if "key" in b.detail]
    short = replace(FINANCE_TERMS_2026_14, skills={"pay_claim": replace(
        FINANCE_TERMS_2026_14.skills["pay_claim"], key_seconds=300)})
    [breach] = [b for b in check(short, claims, NO_CROSSING) if "key" in b.detail]
    assert "keeps a key 300 s" in breach.detail
    assert not [b for b in check(FINANCE_TERMS_2026_13, claims, NO_CROSSING)
                if "key" in b.detail]  # no window: the policy resends nothing


UNAVAILABLE = PartnerUnavailable(PAY, "timed out")


@pytest.mark.parametrize("error", [
    PartnerRejected(PAY, "key reused for a different request"),
    DelegationRevoked(PAY, "adjuster:adj-117 revoked"),
    ContractBreach(PAY, "pay_claim requires vin"),
    MalformedAnswer(PAY, "artifact lacks payment_ref"),
])
def test_an_answer_is_never_retried(error: Exception) -> None:
    assert refuse_retry(POLICY, "get", error, 1, 86_400, 30, T0, T0)  # type: ignore
    assert refuse_retry(POLICY, "send", error, 1, 86_400, 30, T0, T0)  # type: ignore


def test_reads_retry_and_sends_retry_only_under_an_honoured_key() -> None:
    assert refuse_retry(POLICY, "get", UNAVAILABLE, 1, None, 30, T0, T0) == ""
    assert refuse_retry(POLICY, "send", UNAVAILABLE, 1, 86_400, 30, T0, T0 + 60) == ""
    assert "no key window" in refuse_retry(POLICY, "send", UNAVAILABLE, 1, None, 30,
                                           T0, T0)
    assert "past 600 s" in refuse_retry(POLICY, "send", UNAVAILABLE, 1, 86_400, 30,
                                        T0, T0 + 601)
    assert "past 270 s" in refuse_retry(POLICY, "send", UNAVAILABLE, 1, 300, 30,
                                        T0, T0 + 280)  # the window, less the skew
    assert "3 attempts" in refuse_retry(POLICY, "send", UNAVAILABLE, 3, 86_400, 30,
                                        T0, T0)


def test_backoff_is_jittered_and_honours_retry_after() -> None:
    assert backoff(POLICY, 3, UNAVAILABLE, draw=lambda low, high: high) == 4.0
    response = httpx.Response(503, headers={"retry-after": "30"},
                              request=httpx.Request("POST", FINANCE_URL))
    busy = PartnerUnavailable(PAY, "busy")
    busy.__cause__ = httpx.HTTPStatusError("503", request=response.request,
                                           response=response)
    assert backoff(POLICY, 1, busy, draw=lambda low, high: low) == 30.0


async def test_fixed_the_resend_returns_task_a_and_finance_pays_once(
        clock: Clock) -> None:
    meridian = Meridian(clock)  # Finance 2026.14, Claims' payments retry policy
    graph = await meridian.graph()
    meridian.pay.drops = 2  # the send and the first resend both lose their answer
    await meridian.approve_at_claims(graph, "r12")
    [task_a] = meridian.payment_tasks()
    await meridian.finance.approve(task_a, "finance:fa-12")
    assert await wake(graph, thread("r12"), set()) == "done"
    assert (await meridian.entry(graph, "r12"))["task_id"] == task_a
    assert [p["payment_ref"] for p in meridian.finance.paid] == ["PAY-CLM-1001-3000"]


class Sent(TypedDict):
    claim_id: str


async def test_langgraphs_node_retry_retries_a_revoked_chain() -> None:
    calls: list[int] = []

    def send(state: Sent) -> Sent:
        calls.append(1)
        raise DelegationRevoked(PAY, "adjuster:adj-117 revoked")

    graph = StateGraph(Sent)
    graph.add_node("payments", send,
                   retry_policy=NodeRetryPolicy(initial_interval=0.01))
    graph.add_edge(START, "payments")
    graph.add_edge("payments", END)
    with pytest.raises(DelegationRevoked):
        await graph.compile().ainvoke({"claim_id": "CLM-1001"})
    assert len(calls) == 3  # retry_on's default: any exception outside a short list


# Retries across hops -----------------------------------------------------------------

def once(claim: dict[str, Any], seen: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Risk's assistant: the claim's shop history, once."""
    return None if seen else {"claim_id": claim["claim_id"]}


@pytest.mark.parametrize("release, northgate_tasks", [("v7", 2), ("v7.3", 1)])
async def test_an_onward_key_makes_a_rerun_at_risk_one_task_at_northgate(
        clock: Clock, release: str, northgate_tasks: int) -> None:
    meridian = Meridian(clock, northgate_release=release, assistant=once)
    await meridian.graph()
    node = meridian.nodes["fraud_screening"]
    claim = {k: v for k, v in clean_claim().items() if k in node.seam.sends}
    with as_run(thread("h12")):
        for _ in range(2):  # Risk honours no key: one screening task per send
            await node._send(claim, key="k-h12")
    assert len(new_tasks(meridian.risk)) == 2
    asked = [r for r in meridian.northgate.executor.received
             if r["skill"] == "estimate_history"]
    assert len(asked) == northgate_tasks  # v7.3 honours the onward key
    assert onward_key("k-h12", "estimate_history") == onward_key("k-h12",
                                                                 "estimate_history")


def test_a_chain_that_comes_back_is_refused_whatever_its_hops(clock: Clock) -> None:
    northgate_key = Ed25519PrivateKey.generate()
    pinned = {**MERIDIAN.keys, "northgate": northgate_key.public_key()}
    trusts = {ISSUER: Trust("meridian", ISSUER, pinned)}
    signers = {**KEYS, "northgate": northgate_key}
    skills = frozenset({"screen_fraud", "estimate_history"})
    root = sign(Link(ISSUER, "claims", "adjuster:adj-117", frozenset({"CLM-1001"}),
                     skills, frozenset({"claim_id"}), 3, T0 + 900), KEYS[ISSUER])
    chain: tuple[Link, ...] = (root,)
    for issuer, audience in [("claims", "risk"), ("risk", "northgate"),
                             ("northgate", "risk")]:
        above = chain[-1]
        link = Link(issuer, audience, above.subject, above.claims, above.skills,
                    above.fields, above.hops - 1, above.expires, parent=digest(above))
        chain = (*chain, sign(link, signers[issuer]))
    assert not any(narrows(p, c) for p, c in pairwise(chain))  # the hops allow it
    with pytest.raises(DelegationRefused, match="a loop: risk"):
        verify(chain, "northgate", "risk", trusts, T0)
    out = replace(chain[-1], audience="finance")  # the same depth, no owner twice
    verify((*chain[:-1], sign(out, northgate_key)), "northgate", "finance", trusts, T0)


# Reconciliation inward ---------------------------------------------------------------

async def test_the_crash_window_on_risk_is_closed_by_a_lookup(clock: Clock) -> None:
    meridian = Meridian(clock)
    graph = await meridian.graph()
    meridian.screen.crash = True  # Risk has the task; Claims never records it
    with pytest.raises(Crash):
        await graph.ainvoke(clean_claim(), thread("c12"))
    entry = await meridian.entry(graph, "c12", "fraud_screening")
    assert (entry["state"], entry["task_id"]) == ("intended", "")
    await meridian.graph()  # Claims restarts
    graph = build_graph(with_review(meridian.nodes, reviewer(keyword_reader)),
                        meridian.saver)
    await graph.ainvoke(None, thread("c12"))  # the re-run looks the key up first
    [only] = new_tasks(meridian.risk)
    assert (await meridian.entry(graph, "c12", "fraud_screening"))["task_id"] == (
        only["task_id"])


async def test_the_crash_window_on_finance_is_closed_by_the_key(clock: Clock) -> None:
    meridian = Meridian(clock)
    graph = await meridian.graph()
    meridian.pay.crash = True
    with pytest.raises(Crash):
        await graph.ainvoke(clean_claim(), thread("f12"))
    out = await graph.ainvoke(None, thread("f12"))  # the re-run sends under the key
    assert out["__interrupt__"]  # Finance's question, on task A
    assert len(meridian.payment_tasks()) == 1


async def test_a_task_found_by_its_key_is_adopted_by_the_threads_own_run(
        clock: Clock) -> None:
    meridian = Meridian(clock, release="2026.13")  # no key window: never resend
    graph = await meridian.graph()
    meridian.pay.drops, meridian.pay.down_after_drop = 1, True  # a long deploy
    with pytest.raises(PartnerUnavailable, match="no key window"):
        await graph.ainvoke(clean_claim(), thread("a12"))  # the lookup failed too
    meridian.pay.down = False  # Finance is back
    job = meridian.reconciler(graph, "a12")
    [found] = await job.run()
    assert found.verdict == "found" and found.task_id == meridian.payment_tasks()[0]
    assert await job.queue.drain() == {"a12": "asking"}  # adopted: Finance's question
    assert len(meridian.payment_tasks()) == 1


async def test_a_list_that_skips_a_task_never_yields_a_resend(clock: Clock) -> None:
    meridian = Meridian(clock, release="2026.13")
    handler = handler_of(meridian.finance)

    async def skips(*args: Any, **kwargs: Any) -> ListTasksResponse:
        return ListTasksResponse()  # a task updated mid-listing, skipped

    handler.on_list_tasks = skips  # type: ignore[method-assign]
    graph = await meridian.graph()
    meridian.pay.drops = 1
    with pytest.raises(PartnerUnavailable, match="outcome unknown"):
        await graph.ainvoke(clean_claim(), thread("s12"))
    job = meridian.reconciler(graph, "s12")
    [absent] = await job.run()
    assert absent.verdict == "absent" and meridian.findings == [absent]
    assert len(meridian.payment_tasks()) == 1  # nobody sent it again


# Reconciliation outward, and a restore from backup ---------------------------------

async def test_the_job_finds_task_a_as_an_orphan(clock: Clock) -> None:
    meridian = Meridian(clock, release="2026.13")
    graph = await meridian.graph(kind=ResendOnUnknown)
    meridian.pay.drops = 1
    await meridian.approve_at_claims(graph, "o12")
    first, second = meridian.payment_tasks()
    job = meridian.reconciler(graph, "o12")
    findings = await job.run()
    [orphan] = [f for f in findings if f.verdict == "orphan"]
    entry = await meridian.entry(graph, "o12")
    assert (orphan.task_id, orphan.idempotency_key) == (first,
                                                        entry["idempotency_key"])
    assert orphan.detail == f"input-required; the entry with its key names {second}"
    assert meridian.findings[-1] == orphan and meridian.finance.paid == []


async def test_a_restore_from_backup_is_lost_and_never_resent(clock: Clock) -> None:
    meridian = Meridian(clock)
    last_night = meridian.finance.backup()
    graph = await meridian.graph()
    await meridian.approve_at_claims(graph, "b12")
    [task_id] = meridian.payment_tasks()
    await meridian.finance.approve(task_id, "finance:fa-12")  # paid; no push reached
    meridian.finance.restore(last_night)  # Finance restores last night's backup
    job = meridian.reconciler(graph, "b12")
    [lost] = await job.run()
    assert (lost.verdict, lost.task_id) == ("lost", task_id)
    assert lost.detail == "the entry says working" and job.queue.pending == {}
    assert len(meridian.finance.paid) == 1  # the record went back; the money did not
    assert len(meridian.payment_tasks()) == 1


async def test_a_partner_that_stays_down_pages_a_person_after_three_runs(
        clock: Clock) -> None:
    meridian = Meridian(clock)
    graph = await meridian.graph()
    await meridian.approve_at_claims(graph, "d12")
    meridian.pay.down = True
    job = meridian.reconciler(graph, "d12")
    for run in range(3):
        verdicts = {f.verdict for f in await job.run()}
        assert verdicts == {"unreadable"}
        paged = [f for f in meridian.findings if f.task_id]
        assert len(paged) == (1 if run == 2 else 0)
