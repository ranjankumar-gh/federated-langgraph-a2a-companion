"""Failure containment: the outage everyone blamed on Risk, the seam breaker, degraded
modes and the hold horizon, containment on the peer seam, the partner stop and its
drill, and a push flood behind the receiver's rate limit.
"""

import asyncio
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import seamkit.ledger
from meridian.containment import (
    CONTAINED_EXPECTATIONS,
    DESK,
    OWNERS,
    Containment,
    build_contained,
    with_breaker,
)
from meridian.data_classes import MERIDIAN_POLICY
from meridian.drill import T0
from meridian.federated import EXPECTATIONS, A2APartners, a2a_workers
from meridian.finance_server import FinancePartner
from meridian.identity import MANAGER_SESSION, MeridianIssuer, revocation_sources
from meridian.incident import Clocked, SlowNorthgate
from meridian.risk_server import (
    BASIS_DEGRADED,
    HISTORY_BREAKER,
    RISK_TERMS,
    RISK_TERMS_2027_02,
    RiskPartner,
)
from meridian.saga import SagaGraph, build_saga
from meridian.samples import clean_claim, suspect_claim
from meridian.stop_drill import forward_sends, stop_report
from northgate.server import NORTHGATE_V7_4, NorthgatePartner
from seamkit.approval import encode_approval
from seamkit.breaker import BreakerPolicy, SeamBreaker
from seamkit.contract import check
from seamkit.delegation import as_peer
from seamkit.drill import Clock
from seamkit.errors import (
    BreakerOpen,
    CardUntrusted,
    ContractBreach,
    MalformedAnswer,
    PartnerFailed,
    PartnerRejected,
    PartnerUnavailable,
    TaskUnfinished,
)
from seamkit.inventory import inventory_from_graph
from seamkit.push import Backstop, RateLimit, push_token, receiver
from seamkit.remote import CrossingRecord, RemoteAgentNode
from seamkit.retry import NO_RETRY, refuse_retry
from seamkit.stop import Stop
from tests.test_ch11_delivery import KEY, MINUTE, PUSHED, Pushed, forged, post
from tests.test_ch12_retries import no_wait
from tests.wire import thread

pytestmark = pytest.mark.anyio

HOUR = 3600.0
# CLM-1002 before the policy merge, as in Chapter 13: it scores 68 and clears.
BEFORE_MERGE = {**suspect_claim(), "prior_claims": 0}


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    """One clock for every owner, every breaker, and every wire."""
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


def asks_history(claim: dict[str, Any], seen: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Risk's fraud assistant, scripted: the repairing shop's history, once."""
    return None if seen else {"claim_id": claim["claim_id"]}


@dataclass
class Contained:
    """Claims, Risk, Northgate v7.4 and Finance 2026.15 on one clock and its wires."""

    clock: Clock
    slow: float = 0.0  # seconds each send to Northgate takes, from anyone
    risk_release: str = "2027.02"
    risk_breaker: bool = True  # Risk's own breaker on the peer seam
    assistant: Any = None
    frames_wait: bool = False  # Northgate's assessor desk has not finished
    undo_door: bool = True
    saver: InMemorySaver = field(default_factory=InMemorySaver)
    records: list[CrossingRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        lists = revocation_sources(self.issuer)
        self.assessors = asyncio.Event()
        if not self.frames_wait:
            self.assessors.set()
        self.northgate = NorthgatePartner(release="v7.4", assessors=self.assessors,
                                          revocations=lists, now=self.clock)
        self.finance = FinancePartner(release="2026.15", revocations=lists,
                                      now=self.clock)
        lookups = asyncio.Event()
        lookups.set()
        self.risk = RiskPartner(
            release=self.risk_release, lookups=lookups, revocations=lists,
            now=self.clock, assistant=self.assistant,
            breaker=HISTORY_BREAKER if self.risk_breaker else None,
            wire=lambda app: self.wire(app, SlowNorthgate(self.slow)),
            terms=RISK_TERMS_2027_02 if self.risk_release == "2027.02" else RISK_TERMS)
        self.partners = A2APartners(risk=self.risk, northgate=self.northgate,
                                    finance=self.finance)

    def wire(self, app: Any, latency: Callable[[httpx.Request], float] | None = None,
             ) -> httpx.AsyncBaseTransport:
        return Clocked(httpx.ASGITransport(app=app), self.clock, self.clock.advance,
                       latency=latency or (lambda request: 0.0))

    def http(self) -> dict[str, httpx.AsyncClient]:
        """Claims' wires: to Risk at Risk's own speed, to Northgate at Northgate's."""
        slow = SlowNorthgate(self.slow)

        def client(app: Any, url: str, latency: Any = None) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=self.wire(as_peer(app, "claims"), latency), base_url=url)

        from meridian.risk_server import RISK_URL
        from northgate.server import NORTHGATE_URL
        return {"fraud_screening": client(self.risk.app, RISK_URL),
                "repair_estimate": client(self.northgate.app, NORTHGATE_URL, slow),
                "repair_booking": client(self.northgate.app, NORTHGATE_URL, slow)}

    async def graph(self) -> SagaGraph:
        graph, self.nodes, self.undoer, self.room = await build_contained(
            self.partners, self.http(), self.saver, self.issuer, self.records.append,
            lambda finding: None, undo_door=self.undo_door)
        self.proven()
        return graph

    def proven(self) -> None:
        for node in [*self.nodes.values(), *self.undoer.undo.values()]:
            node.sleep = no_wait
        # Northgate's estimates are proven under this card (Chapter 5's gradient).
        record = self.nodes["repair_estimate"].partner.record("estimate_repair")
        record.clean.update(f"earlier-{n}" for n in range(20))

    def received(self, skill: str) -> int:
        return sum(1 for r in self.northgate.executor.received if r["skill"] == skill)

    def approval(self, asked: Mapping[str, Any]) -> dict[str, Any]:
        signed = self.issuer.approve(MANAGER_SESSION, str(asked["task_id"]),
                                     str(asked["claim_id"]), float(asked["amount"]),
                                     str(asked["asked"]))
        return {"approved": True, "approval": encode_approval(signed)}


def opened(breaker: SeamBreaker, now: float) -> None:
    for _ in range(breaker.policy.failures):
        assert breaker.allow(now) == ""
        breaker.record(PartnerUnavailable, 10.0, now)
    assert breaker.state == "open"


# The incident ------------------------------------------------------------------------

async def test_the_incident_a_slow_northgate_times_out_claims_risk_seam(
        clock: Clock) -> None:
    fed = Contained(clock, slow=12.0, risk_release="2026.12", risk_breaker=False,
                    assistant=asks_history)
    graph, nodes, _ = await build_saga(fed.partners, fed.http(), fed.saver,
                                       fed.issuer, fed.records.append)
    for node in nodes.values():
        node.sleep = no_wait
    started = clock()
    out = await graph.ainvoke(suspect_claim(), thread("i14"))
    assert (out["fraud_score"], out["decision"]) == (82, "refer")  # never needs Northgate
    assert clock() - started > 10  # past the ten seconds Claims waits on Risk
    timed_out = [r for r in fed.records if r.seam == "fraud_screening" and r.error]
    assert timed_out and timed_out[0].call == "send"  # on Claims' dashboard: Risk
    assert not [r for r in fed.records if r.partner == "Northgate"]  # Claims never called
    assert fed.received("estimate_history") == 1  # but Risk did, for this claim


async def test_fixed_risk_contains_its_own_seam_and_claims_breaker_stays_closed(
        clock: Clock) -> None:
    fed = Contained(clock, slow=12.0, assistant=asks_history)
    graph = await fed.graph()
    for n in range(3):  # three screenings pay the timeout; Risk's breaker counts them
        out = await graph.ainvoke(suspect_claim(), thread(f"f14-{n}"))
        assert out["decision"] == "refer"
    assert fed.risk.history_breaker is not None
    assert fed.risk.history_breaker.state == "open"
    assert fed.room.breakers["fraud_screening"].state == "closed"  # 3 of its 5
    started = clock()
    out = await graph.ainvoke(clean_claim(), thread("f14-clean"))
    assert clock() - started < 1  # no wait on Northgate at Risk now
    assert out["screening_basis"] == BASIS_DEGRADED
    assert out["decision"] == "refer"  # a degraded screening never clears a claim
    assert out["degraded"]["fraud_screening"].startswith("referred")
    out = await graph.ainvoke(suspect_claim(), thread("f14-suspect"))
    assert out["decision"] == "refer" and out["degraded"] == {}  # 82 refers anyway
    assert fed.received("estimate_history") == 3  # Risk sent nothing while open


# The seam breaker --------------------------------------------------------------------

def breaker(trial: str = "estimate_repair", skill: str = "estimate_repair",
            **changes: Any) -> SeamBreaker:
    policy = CONTAINED_EXPECTATIONS["repair_estimate"].breaker
    assert policy is not None
    return SeamBreaker("repair_estimate", "Northgate", skill,
                       replace(policy, trial=trial, **changes))


@pytest.mark.parametrize("outcome", [
    (PartnerUnavailable, 10.0), (PartnerFailed, 1.0), (None, 7.0)])
def test_the_breaker_opens_on_failures_to_answer(
        outcome: tuple[type[PartnerFailed] | None, float]) -> None:
    seam = breaker()
    for _ in range(3):
        assert seam.allow(T0) == ""
        seam.record(outcome[0], outcome[1], T0)
    assert seam.state == "open" and seam.refuses(T0).startswith("open for 300")


@pytest.mark.parametrize("answer", [PartnerRejected, ContractBreach, CardUntrusted,
                                    MalformedAnswer])
def test_an_answer_never_counts(answer: type[PartnerRejected]) -> None:
    seam = breaker()
    for _ in range(10):  # Chapter 4's v8 rejections: Claims' own requests were wrong
        assert seam.allow(T0) == ""
        seam.record(answer, 1.0, T0)
    assert seam.state == "closed"


def test_half_open_admits_one_trial_and_a_booking_never_trials() -> None:
    estimate = breaker()
    booking = breaker(skill="book_repair")
    booking.follows = estimate
    opened(booking, T0)
    opened(estimate, T0)
    later = T0 + 301
    assert booking.refuses(later) == "waits on estimate_repair to close"
    assert estimate.allow(later) == ""  # the trial: somebody's real estimate
    assert estimate.allow(later) == "half-open, its one trial in flight"
    estimate.record(None, 1.0, later)  # it answered in time
    assert estimate.state == "closed"
    assert booking.allow(later) == "" and booking.state == "closed"


def test_the_in_flight_cap_refuses_the_fifth_send() -> None:
    seam = breaker()
    assert [seam.allow(T0) for _ in range(5)][-1] == "4 tasks already in flight"
    seam.record(None, 1.0, T0)  # one answers: there is room again
    assert seam.allow(T0) == ""


def test_an_open_breaker_retries_nothing() -> None:
    lost = PartnerUnavailable("payments", "no answer")
    why = refuse_retry(replace(NO_RETRY, attempts=3, deadline_seconds=600), "send",
                       lost, 1, 24 * 3600, 30, T0, T0 + 5, held="open for 590 s more")
    assert why == "the seam's breaker holds: open for 590 s more"


def test_check_refuses_a_late_threshold_under_the_partners_p95() -> None:
    expects = CONTAINED_EXPECTATIONS["repair_estimate"]
    assert expects.breaker is not None
    assert check(NORTHGATE_V7_4, expects, MERIDIAN_POLICY) == []
    eager = replace(expects, breaker=replace(expects.breaker, late_seconds=3.0))
    [breach] = check(NORTHGATE_V7_4, eager, MERIDIAN_POLICY)
    assert breach.detail == ("estimate_repair promises 4 s at p95, and this seam's "
                             "breaker counts an answer after 3 s against it")


# Degraded modes ----------------------------------------------------------------------

async def test_an_open_estimate_breaker_sends_nothing_and_the_desk_substitutes(
        clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    opened(fed.room.breakers["repair_estimate"], clock())
    out = await graph.ainvoke(clean_claim(), thread("d14"))
    [asked] = out["__interrupt__"]
    assert (asked.value["to"], asked.value["needs"]) == (DESK, "repair_estimate")
    assert fed.received("estimate_repair") == 0  # nothing crossed
    refused = [r for r in fed.records if r.seam == "repair_estimate"]
    assert [r.error for r in refused] == ["BreakerOpen"]
    with pytest.raises(BreakerOpen, match="waits on estimate_repair"):  # booking holds
        await graph.ainvoke(Command(resume={"estimate": 2_900.0, "assessor": "ad-4"}),
                            thread("d14"))
    values = (await graph.aget_state(thread("d14"))).values
    assert values["repair_estimate"] == 2_900.0
    assert values["provenance"]["repair_estimate"]["substitute"] == f"{DESK}:ad-4"
    assert values["provenance"]["repair_estimate"]["task_id"] == ""  # not Northgate's
    assert "repair estimate 2,900" in values["payment_memo"]  # what Finance will print
    assert fed.room.held == {"d14": ("repair_booking", None)}  # nothing at stake yet


async def test_a_held_claim_resumes_when_the_breaker_closes(clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    pay = fed.room.breakers["payments"]
    opened(pay, clock())
    with pytest.raises(BreakerOpen, match="held"):
        await graph.ainvoke(clean_claim(), thread("h14"))
    assert fed.room.due(clock()) == []  # open for ten minutes
    clock.advance(601)
    assert fed.room.due(clock()) == ["h14"]  # its trial is due: ring it
    from seamkit.push import WakeQueue
    queue = WakeQueue(graph, thread)
    queue.ring("h14")
    assert await queue.drain() == {"h14": "asking"}  # Finance asks its approver
    assert pay.state == "closed" and fed.room.held == {}


async def test_a_hold_aborts_at_the_horizon_and_the_booking_is_undone(
        clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    pay = fed.room.breakers["payments"]
    pay.policy = replace(pay.policy, open_seconds=30 * 24 * HOUR)  # Finance stays out
    opened(pay, clock())
    with pytest.raises(BreakerOpen):
        await graph.ainvoke(clean_claim(), thread("z14"))
    [(seam, horizon)] = fed.room.held.values()
    assert seam == "payments" and horizon is not None
    assert horizon - clock() == pytest.approx(71 * HOUR, abs=5)  # 72 h less the margin
    clock.advance(70 * HOUR)
    assert fed.room.due(clock()) == []
    clock.advance(HOUR + 5)
    assert fed.room.due(clock()) == ["z14"]
    out = await graph.ainvoke(None, thread("z14"))
    assert out["aborted"] == "payments: held to the hold horizon"
    assert out["undone"]["repair_booking"] == "compensated"  # inside its 72 hours
    [booking] = fed.northgate.executor.booked.values()
    assert booking.cancelled == {"outcome": "cancelled", "owed": 0.0, "residue": ""}


# The partner stop --------------------------------------------------------------------

async def stopped_federation(clock: Clock, undo_door: bool = True,
                             ) -> tuple[Contained, SagaGraph, Any]:
    """Two claims at Northgate: a frame estimate in flight, and a repair booked whose
    payment waits on Claims' claims manager."""
    from seamkit.push import WakeQueue
    fed = Contained(clock, frames_wait=True, undo_door=undo_door)
    graph = await fed.graph()
    queue = WakeQueue(graph, thread)
    with pytest.raises(TaskUnfinished):  # the assessor desk is on CLM-1002's frame
        await queue.run(BEFORE_MERGE, thread("s14-frame"))
    out = await queue.run(clean_claim(), thread("s14-booked"))
    assert out["booking_ref"] == "BK-CLM-1001-SHOP-3" and out["__interrupt__"]
    return fed, graph, queue


async def test_the_stop_drill(clock: Clock) -> None:
    fed, graph, queue = await stopped_federation(clock)
    threads = {tid: thread(tid) for tid in ("s14-frame", "s14-booked")}
    before = {tid: (await graph.aget_state(c)).values["ledger"]
              for tid, c in threads.items()}
    sent = len(fed.records)
    stop = Stop("Northgate", "the estimate desk is under investigation",
                "claims/on-call", clock(), unwind=True)
    rung = await fed.room.stops.stop(stop, queue, threads)
    assert rung == ["s14-frame", "s14-booked"]
    for tid, config in threads.items():  # the stop rang; it wrote nothing
        assert (await graph.aget_state(config)).values["ledger"] == before[tid]
    assert await queue.drain() == {"s14-frame": "asking", "s14-booked": "done"}
    report = {row.thread: row for row in await stop_report(
        graph, threads, ("repair_estimate", "repair_booking"))}
    frame, booked = report["s14-frame"], report["s14-booked"]
    assert frame.tasks == {"repair_estimate": "canceled (by the stop)"}
    assert frame.modes == {}  # it asks the desk now; the desk's answer records it
    assert booked.undone == {"payments": "cancelled", "repair_booking": "compensated",
                             "repair_estimate": "cost accepted",
                             "fraud_screening": "nothing to undo"}
    assert forward_sends(fed.records[sent:], "Northgate") == []  # the forward door
    assert any(r.seam == "repair_booking.undo" and r.call == "send"
               for r in fed.records[sent:])  # the undo door
    assert fed.northgate.executor.canceled  # the estimate, by the frame claim's run
    [booking] = fed.northgate.executor.booked.values()
    assert booking.cancelled is not None


async def test_a_stop_that_closes_the_undo_door_leaves_the_booking_standing(
        clock: Clock) -> None:
    fed, graph, queue = await stopped_federation(clock, undo_door=False)
    stop = Stop("Northgate", "the estimate desk is under investigation",
                "claims/on-call", clock(), unwind=True)
    await fed.room.stops.stop(stop, queue, ["s14-booked"])
    await queue.drain()
    out = (await graph.aget_state(thread("s14-booked"))).values
    assert out["undone"]["repair_booking"] == "uncompensated"
    [booking] = fed.northgate.executor.booked.values()
    assert booking.cancelled is None  # the shop keeps the bay


async def test_a_released_stop_sends_a_new_task_under_a_new_key(clock: Clock) -> None:
    fed, graph, queue = await stopped_federation(clock)
    stop = Stop("Northgate", "maintenance", "claims/on-call", clock())  # no unwind
    await fed.room.stops.stop(stop, queue, ["s14-frame"])
    await queue.drain()
    old = (await queue.entry("s14-frame", "repair_estimate"))
    assert old is not None and old["stopped_by"] == stop.name
    fed.room.stops.release("Northgate", clock())
    fed.assessors.set()
    out = await graph.ainvoke(Command(resume={"estimate": 8_100.0, "assessor": "ad-4"}),
                              thread("s14-frame"))
    new = (await queue.entry("s14-frame", "repair_estimate"))
    assert new is not None and new["earlier"] == [old["task_id"]]
    assert new["idempotency_key"] != old["idempotency_key"]
    assert out["repair_estimate"] == 8_300  # Northgate's own, after the release


# The receiver ------------------------------------------------------------------------

FLOOD = BreakerPolicy(late_seconds=60.0, failures=10, window_seconds=180,
                      open_seconds=600, trial="estimate_repair", max_in_flight=1_000,
                      mode="hold")


async def test_a_push_flood_opens_the_receivers_breaker_and_the_backstop_finds_it(
        clock: Clock) -> None:
    meridian = Pushed(clock)
    graph, queue = await meridian.start()
    flood = SeamBreaker("repair_estimate", "Northgate", "estimate_repair", FLOOD)
    meridian.late.app = receiver(KEY, queue, RateLimit(20, MINUTE, clock),
                                 flood={"repair_estimate": flood})
    entry = await meridian.order("clm-1002")
    nodes = a2a_workers(meridian.fed.connections, seams=PUSHED, issuer=meridian.issuer)
    every = float(EXPECTATIONS["repair_estimate"].backstop_seconds or 0)
    backstop = Backstop(queue, lambda: {"clm-1002": thread("clm-1002")},
                        nodes["repair_estimate"], every)
    token = push_token(KEY, "repair_estimate", entry["idempotency_key"])
    while clock() < T0 + 18 * MINUTE:  # ticks at 0, 5, 10, 15: nothing moved
        await backstop.tick(clock())
        clock.advance(MINUTE)
    statuses = []
    for _ in range(3 * 25):  # 25 a minute for three minutes, past the limit of 20
        statuses.append((await post(meridian, token, forged(entry))).status_code)
        clock.advance(MINUTE / 25)
        await backstop.tick(clock())
    assert flood.state == "open" and statuses[-1] == 503
    queue.pending.clear()  # the flood's rings: each run would only read a working task
    await meridian.assessor_finishes()  # Northgate's real push, into the open breaker
    assert meridian.bell.attempts[-1][1] == 503
    found: dict[float, dict[str, str]] = {}
    while not found:
        if woken := await backstop.tick(clock()):
            found[clock()] = woken
        clock.advance(MINUTE)
    [(when, woken)] = found.items()
    assert woken == {"clm-1002": "done"}
    assert when - (T0 + 21 * MINUTE) <= every  # at most one cadence late


# The inventory -----------------------------------------------------------------------

async def test_the_inventory_fills_the_failure_column(clock: Clock) -> None:
    fed = Contained(clock)
    graph = await fed.graph()
    inventory = with_breaker(inventory_from_graph(graph, OWNERS))
    failure = {e.callee: e.answers["failure"] for e in inventory.entries}
    assert set(failure) == {"fraud_screening", "repair_estimate", "repair_booking",
                            "payments"}
    assert failure["repair_estimate"].endswith("; substitute")
    assert failure["fraud_screening"].endswith("; hold")
    assert isinstance(fed.room, Containment)
