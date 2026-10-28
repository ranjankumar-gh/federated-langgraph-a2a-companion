"""Claims' CI, with Northgate replaced by the partner stub: the stub's card and its key,
the build-time check against Northgate's terms, the stub answering nothing it was not
given, and the fault catalogue: every failure path Chapters 11 to 15 built, run against
a partner Claims cannot deploy.
"""

import asyncio
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

import seamkit.ledger
from meridian.canaries import ESTIMATE_BASELINE, ESTIMATE_GOLDENS
from meridian.data_classes import MERIDIAN_POLICY
from meridian.drill import T0
from meridian.federated import EXPECTATIONS, SEAMS, A2APartners
from meridian.finance_server import FinancePartner
from meridian.identity import (
    CANARY_SESSION,
    CI_SESSION,
    MeridianIssuer,
    revocation_sources,
)
from meridian.incident import AFTERMARKET_PRICES, Clocked
from meridian.risk_server import RiskPartner
from meridian.saga import MERIDIAN_PLAN, SAGA_EXPECTATIONS, UNDO_EXPECTATIONS
from meridian.samples import clean_claim
from meridian.stubs import CI_TRUST, PRICED, RECORDING, UNPRICED, northgate_stub
from meridian.trust import MERIDIAN_TRUST, SAGA_TRUST
from northgate.server import NorthgatePartner
from seamkit.breaker import SeamBreaker
from seamkit.canary import Baseline, CanarySuite
from seamkit.contract import seam_spec
from seamkit.drill import Clock
from seamkit.errors import (
    BreakerOpen,
    PartnerFailed,
    PartnerRejected,
    SeamError,
    TaskUnfinished,
)
from seamkit.push import Backstop, PushTo, RateLimit, WakeQueue, receiver
from seamkit.reconcile import Finding, not_found
from seamkit.remote import CrossingRecord, RemoteAgentNode, connect
from seamkit.saga import Compensator
from stub.faults import Fault, FaultScript
from stub.recording import Exchange, from_baseline, load
from stub.server import STUB_KID, StubOutOfContract, StubPartner
from tests.test_ch11_delivery import KEY, MINUTE, PUSHED, RECEIVER, Bell, Late
from tests.test_ch14_containment import FLOOD
from tests.wire import Federation


def thread(tid: str) -> RunnableConfig:
    """A run of Claims' CI pipeline, under its own session."""
    return {"configurable": {"thread_id": tid, "session": CI_SESSION}}

pytestmark = pytest.mark.anyio

SEAM = "repair_estimate"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


async def node_on(stub: StubPartner, seam: str = SEAM,
                  wire: httpx.AsyncBaseTransport | None = None,
                  breaker: SeamBreaker | None = None,
                  records: list[CrossingRecord] | None = None) -> RemoteAgentNode:
    """A Claims node on one seam, connected to the stub under Claims' CI policy."""
    http = httpx.AsyncClient(transport=wire or httpx.ASGITransport(app=stub.app),
                             base_url=stub.url)
    expects = {e.seam: e for e in (*SAGA_EXPECTATIONS.values(),
                                   *UNDO_EXPECTATIONS.values())}[seam]
    connection = await connect(stub.url, http, expects, MERIDIAN_POLICY,
                               trust=CI_TRUST["Northgate"])
    spec = SEAMS.get(seam) or seam_spec(expects)
    log = records.append if records is not None else (lambda record: None)
    return RemoteAgentNode(spec, connection, log, breaker=breaker)


async def drive(node: RemoteAgentNode, state: Mapping[str, Any]) -> dict[str, Any]:
    """Enter the node until its seam completes, as the graph's supersteps would.
    A SeamError ends the run, as it ends a claim's; it is returned, not raised."""
    held: dict[str, Any] = {**state, "ledger": dict(state.get("ledger", {}))}
    for _ in range(4):
        try:
            update = await node(held)
        except SeamError as exc:
            held["error"] = exc
            return held
        held["ledger"] = {**held["ledger"], **update.pop("ledger", {})}
        held.update(update)
        if node.seam.name in update.get("completed", []):
            break
    return held


# The stub and its card ---------------------------------------------------------------

async def test_the_stub_serves_northgates_card_under_a_ci_key() -> None:
    stub = northgate_stub()
    recording = load(RECORDING)
    assert stub.digest == recording.card_digest  # the bytes a signature covers
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=stub.app),
                             base_url=stub.url)
    ci = await connect(stub.url, http, EXPECTATIONS[SEAM], MERIDIAN_POLICY,
                       trust=CI_TRUST["Northgate"])
    assert ci.trust is not None and ci.trust.pinned_by == "key"
    assert ci.breaches == () and ci.terms[0].revision == "v7.4"
    production = await connect(stub.url, http, EXPECTATIONS[SEAM], MERIDIAN_POLICY,
                               trust=MERIDIAN_TRUST["Northgate"])
    assert production.trust is not None and not production.trust.trusted


def test_no_production_trust_policy_pins_the_stub_key() -> None:
    for policy in (MERIDIAN_TRUST, SAGA_TRUST):
        assert all(STUB_KID not in trust.keys for trust in policy.values())
    assert STUB_KID in CI_TRUST["Northgate"].keys


def test_the_stub_package_imports_nothing_of_northgates() -> None:
    probe = ("import sys, stub.server, stub.recording, stub.faults; "
             "print(sorted(m for m in sys.modules if m.split('.')[0] == 'northgate'))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"


# Held to the terms at build time ------------------------------------------------------

OUT_OF_CONTRACT = [
    (Exchange("estimate_repair", PRICED, "completed",
              {"estimate": "3,000", "notes": ""}, "scripted"),
     "estimate is '3,000', not estimate as a required number in claim currency"),
    (Exchange("estimate_repair", PRICED, "completed", {"estimate": 3_000.0},
              "scripted"), "the answer lacks notes"),
    (Exchange("estimate_repair", PRICED, "input-required", {}, "scripted",
              {"needs": "vin"}), "asks for 'vin', which it never declares"),
    (Exchange("estimate_repair", UNPRICED, "failed", {}, "scripted"),
     "a failed task is a fault, not an answer: script it as one"),
    (Exchange("estimate_valuation", PRICED, "completed", {"value": 1.0}, "scripted"),
     "estimate_valuation is not a skill on this card under terms v7.4"),
]


@pytest.mark.parametrize(("exchange", "why"), OUT_OF_CONTRACT)
def test_an_answer_the_terms_do_not_allow_refuses_to_build(exchange: Exchange,
                                                           why: str) -> None:
    with pytest.raises(StubOutOfContract, match=why.replace("(", r"\(")):
        northgate_stub(scripted=[exchange])


async def test_the_stub_answers_only_what_it_was_given() -> None:
    stub = northgate_stub()
    node = await node_on(stub)
    out = await drive(node, PRICED)
    assert out["repair_estimate"] == 3_000.0
    out = await drive(node, {"claim_id": "CLM-R09", "damaged_parts": ["bumper"]})
    assert isinstance(out["error"], PartnerRejected)  # never an invented estimate
    task = await node.read(out["ledger"][SEAM]["task_id"])
    assert "no recording for estimate_repair" in str(task.status.message)


# The fault catalogue: Chapter 12 ------------------------------------------------------

async def test_ch12_a_lost_answer_is_found_by_its_key_and_estimated_once() -> None:
    stub = northgate_stub(script=FaultScript().add(Fault("lose_answer"), nth=1))
    node = await node_on(stub)
    out = await drive(node, PRICED)
    assert out["repair_estimate"] == 3_000.0  # adopted by its key, never resent
    assert stub.sent("estimate_repair") == 1


async def test_ch12_a_repeated_key_gets_the_first_task() -> None:
    stub = northgate_stub()
    node = await node_on(stub)
    first = await node.call_once(PRICED, key="claims-key-1")
    again = await node.call_once(PRICED, key="claims-key-1")
    assert first == again and stub.sent("estimate_repair") == 1


async def test_ch12_a_restore_from_backup_loses_the_task_and_nothing_resends() -> None:
    stub = northgate_stub(script=FaultScript().add(Fault("hang")))
    node = await node_on(stub)
    out = await drive(node, PRICED)
    assert isinstance(out["error"], TaskUnfinished)  # Northgate is on it
    await stub.restore()
    lost = await drive(node, out)
    assert isinstance(lost["error"], PartnerFailed) and not_found(lost["error"])
    assert stub.sent("estimate_repair") == 1


# The fault catalogue: Chapter 14 ------------------------------------------------------

def estimate_breaker() -> SeamBreaker:
    policy = EXPECTATIONS[SEAM].breaker
    assert policy is not None
    return SeamBreaker(SEAM, "Northgate", "estimate_repair", policy)


@pytest.mark.parametrize("fault", [Fault("slow", seconds=12.0), Fault("fail"),
                                   Fault("down")])
async def test_ch14_a_slow_failing_or_down_partner_opens_the_breaker(
        clock: Clock, fault: Fault) -> None:
    stub = northgate_stub(script=FaultScript().add(fault), advance=clock.advance)
    breaker = estimate_breaker()
    wire = Clocked(httpx.ASGITransport(app=stub.app), clock, clock.advance)
    node = await node_on(stub, wire=wire, breaker=breaker)
    for _ in range(3):  # three counted outcomes inside a minute
        await drive(node, PRICED)
    assert breaker.state == "open"
    refused = await drive(node, PRICED)
    assert isinstance(refused["error"], BreakerOpen)
    assert stub.sent("estimate_repair") == (0 if fault.kind == "down" else 3)


async def test_ch14_a_failure_after_the_send_returned_still_counts(
        clock: Clock) -> None:
    stub = northgate_stub(script=FaultScript().add(Fault("fail")))
    breaker = estimate_breaker()
    node = await node_on(stub, breaker=breaker)
    node.seam = replace(node.seam, return_immediately=True)  # the push seam's send
    for _ in range(3):  # each send returns at once; each task fails a moment later
        out = await drive(node, PRICED)
        while isinstance(out["error"], TaskUnfinished):  # read again, as a poll would
            await asyncio.sleep(0.01)
            out = await drive(node, out)
        assert isinstance(out["error"], PartnerFailed)
    assert breaker.state == "open"


async def test_ch14_a_refusal_never_opens_the_breaker(clock: Clock) -> None:
    stub = northgate_stub()
    breaker = estimate_breaker()
    node = await node_on(stub, breaker=breaker)
    for _ in range(10):  # Claims' own unpriced requests, refused as recorded
        assert isinstance((await drive(node, UNPRICED))["error"], PartnerRejected)
    assert breaker.state == "closed"


# The fault catalogue: Chapter 13 ------------------------------------------------------

@dataclass
class Undo:
    """A booking at the stub, and Chapter 13's compensator to undo it."""

    stub: StubPartner
    findings: list[Finding]

    async def undo(self, claim_id: str) -> tuple[str, list[Finding]]:
        booking = await node_on(self.stub, "repair_booking")
        booked = await drive(booking, {"claim_id": claim_id,
                                       "damaged_parts": ["frame"]})
        undo = await node_on(self.stub, "repair_booking.undo")
        compensator = Compensator(MERIDIAN_PLAN, {"repair_booking": undo},
                                  lambda: "", self.findings.append)
        state: dict[str, Any] = {"claim_id": claim_id, "decision": "refer",
                                 "ledger": booked["ledger"], "undone": {}}
        while "repair_booking" not in state["undone"]:
            update = await compensator(state)
            state["ledger"] = {**state["ledger"], **update.get("ledger", {})}
            state["undone"] = {**state["undone"], **update.get("undone", {})}
        return state["undone"]["repair_booking"], self.findings


@pytest.mark.parametrize(("claim_id", "verdict", "detail"), [
    ("CLM-R05", "compensated", ""),
    ("CLM-R06", "residue", "owed 6,500; frame ordered by SHOP-7 for CLM-R06"),
    ("CLM-R07", "refused", "the repair of CLM-R07 has started at SHOP-7"),
])
async def test_ch13_each_answer_to_a_compensation(claim_id: str, verdict: str,
                                                  detail: str) -> None:
    got, findings = await Undo(northgate_stub(), []).undo(claim_id)
    assert got == verdict
    assert [f.detail.split("; settled")[0] for f in findings] == (
        [detail] if detail else [])


async def test_ch13_a_late_compensation_reaches_a_person(clock: Clock) -> None:
    script = FaultScript().add(Fault("slow", seconds=12.0), skill="cancel_booking")
    stub = northgate_stub(script=script, advance=clock.advance)
    undo = Undo(stub, [])
    wire = Clocked(httpx.ASGITransport(app=stub.app), clock, clock.advance)
    booking = await node_on(stub, "repair_booking")
    booked = await drive(booking, {"claim_id": "CLM-R05", "damaged_parts": ["frame"]})
    node = await node_on(stub, "repair_booking.undo", wire=wire)
    compensator = Compensator(MERIDIAN_PLAN, {"repair_booking": node}, lambda: "",
                              undo.findings.append)
    state: dict[str, Any] = {"claim_id": "CLM-R05", "decision": "refer",
                             "ledger": booked["ledger"], "undone": {}}
    while "repair_booking" not in state["undone"]:
        update = await compensator(state)
        state["ledger"] = {**state["ledger"], **update.get("ledger", {})}
        state["undone"] = {**state["undone"], **update.get("undone", {})}
    assert state["undone"]["repair_booking"] == "uncompensated"
    [finding] = undo.findings
    assert finding.verdict == "uncompensated"  # a person, never a retry


# The fault catalogue: Chapter 11 ------------------------------------------------------

@dataclass
class OnStub:
    """Claims' push seam on one clock, with Northgate replaced by the stub."""

    clock: Clock
    fault: Fault

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        self.late = Late()
        self.bell = Bell(self.late, self.clock)
        self.stub = northgate_stub(
            script=FaultScript().add(self.fault, claim_id="CLM-R01"), now=self.clock,
            push=httpx.AsyncClient(transport=self.bell, base_url=RECEIVER))
        lists = revocation_sources(self.issuer)
        partners = A2APartners(RiskPartner(revocations=lists, now=self.clock),
                               NorthgatePartner(revocations=lists, now=self.clock),
                               FinancePartner(revocations=lists, now=self.clock))
        self.fed = Federation(partners, seams=PUSHED, issuer=self.issuer,
                              trust=CI_TRUST)

    async def start(self) -> tuple[WakeQueue, str]:
        graph = await self.fed.graph(
            checkpointer=InMemorySaver(), bells={SEAM: PushTo(f"{RECEIVER}/push", KEY)},
            apps={SEAM: (self.stub.url, self.stub.app)})
        self.queue = WakeQueue(graph, thread)
        self.late.app = receiver(KEY, self.queue, RateLimit(20, MINUTE, self.clock))
        with pytest.raises(TaskUnfinished):  # the stub keeps CLM-R01 working
            await self.queue.run({**clean_claim(), **PRICED}, thread("r01"))
        await self.queue.drain()
        entry = await self.queue.entry("r01", SEAM)
        assert entry is not None and entry["state"] == "working"
        return self.queue, entry["task_id"]


async def test_ch11_a_lost_push_is_found_within_one_backstop_cadence(
        clock: Clock) -> None:
    claims = OnStub(clock, Fault("push_lost"))
    queue, task_id = await claims.start()
    every = float(EXPECTATIONS[SEAM].backstop_seconds or 0)
    node = await node_on(claims.stub)
    backstop = Backstop(queue, lambda: {"r01": thread("r01")}, node, every)
    await backstop.tick(clock())
    clock.advance(2 * MINUTE)
    await claims.stub.finish(task_id)  # answered, and nobody is rung
    finished = clock()
    found: dict[str, str] = {}
    while not found:
        clock.advance(MINUTE)
        found = await backstop.tick(clock())
    assert found == {"r01": "done"} and clock() - finished <= every
    assert not claims.bell.attempts[-1:] or claims.bell.attempts[-1][0] < finished


async def test_ch11_a_forged_push_writes_nothing(clock: Clock) -> None:
    claims = OnStub(clock, Fault("hang"))
    queue, task_id = await claims.start()
    before = await queue.entry("r01", SEAM)
    assert await claims.stub.forge(task_id) == 202  # a real token, a state never sent
    assert await queue.drain() == {"r01": "TaskUnfinished"}  # it read: still working
    assert await queue.entry("r01", SEAM) == before
    await claims.stub.finish(task_id)  # the real push rings, and the run reads
    assert await queue.drain() == {"r01": "done"}


async def test_ch14_a_push_flood_opens_the_receivers_breaker(clock: Clock) -> None:
    claims = OnStub(clock, Fault("hang"))
    queue, task_id = await claims.start()
    flood = SeamBreaker(SEAM, "Northgate", "estimate_repair", FLOOD)
    claims.late.app = receiver(KEY, queue, RateLimit(20, MINUTE, clock),
                               flood={SEAM: flood})
    statuses = []
    for _ in range(3 * 25):  # 25 a minute for three minutes, past the limit of 20
        statuses.append(await claims.stub.forge(task_id))
        clock.advance(MINUTE / 25)
    assert flood.state == "open" and statuses[-1] == 503


# The fault catalogue: Chapter 15 ------------------------------------------------------

def suite_on(node: RemoteAgentNode, findings: list[Finding]) -> CanarySuite:
    policy = EXPECTATIONS[SEAM].canary
    assert policy is not None
    return CanarySuite(SEAM, policy, ESTIMATE_GOLDENS, ESTIMATE_BASELINE, node,
                       {"session": CANARY_SESSION}, findings.append)


async def test_ch15_the_judge_holds_the_accepted_answers_in_band() -> None:
    findings: list[Finding] = []
    verdict = await suite_on(await node_on(northgate_stub()), findings).go(T0)
    assert verdict.kind == "in band" and findings == []


async def test_ch15_the_judge_finds_drift_in_a_drifted_recording() -> None:
    recording = load(RECORDING)
    drifted = Baseline({g.id: {"repair_estimate": sum(AFTERMARKET_PRICES[p] for p in
                                                      g.request["damaged_parts"])}
                        for g in ESTIMATE_GOLDENS}, "ci", T0)
    canaries = from_baseline(ESTIMATE_GOLDENS, drifted, EXPECTATIONS[SEAM],
                             answered={"notes": ""})
    kept = tuple(e for e in recording.exchanges if e.source != "canary")
    stub = northgate_stub(recording=replace(recording, exchanges=kept + canaries))
    findings: list[Finding] = []
    verdict = await suite_on(await node_on(stub), findings).go(T0)
    assert (verdict.kind, verdict.goldens) == ("drift", ("CLM-C02", "CLM-C03"))
    [finding] = findings
    assert finding.verdict == "canary drift"
