"""Meridian assembled (the closer): four owners, each behind its own federation gateway.

Claims, Risk and Finance each load their gateway from `meridian/gateway/<owner>/` and
build every seam they send on, and every skill they serve, through it. Northgate is
another company and runs its own server as it always has: Meridian's gateways end at
Meridian's side of each seam.

`hub=False` is the shape Chapter 2 chose: Risk's fraud assistant calls Northgate's
estimate_history as a tool when a claim's repair looks unusual (here: a claim over
10,000). `hub=True` is the shape the rerun found right: Claims reads the estimate
history of every claim it screens, before screening, and sends it to Risk as a fifth
field, under Risk's terms revision 2027.08.

`measure` reads only what the owners already record: the requests on each owner's
wire, the receipts, the cost entries, the ledger entries in Claims' checkpoint, and the
crossing records. Latency is wall time on the in-process wire: a floor, never a
production number.

Run `python -m meridian.assembled` to print both shapes' measurements.
"""

import asyncio
import json
import os
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import seamkit.ledger
from meridian.data_classes import HUB_POLICY, MERIDIAN_POLICY
from meridian.federated import A2APartners, in_process_http, run_chain_id, with_review
from meridian.finance_server import FinancePartner
from meridian.identity import (
    ADJUSTER,
    ADJUSTER_SESSION,
    KEYS,
    MANAGER_SESSION,
    MERIDIAN_TRUSTS,
    ROOT,
    SESSIONS,
    MeridianIssuer,
    revocation_sources,
)
from meridian.inprocess import ClaimGraph, ClaimState, Node, build_graph
from meridian.rates import RATES
from meridian.receipts import pinned
from meridian.resume import ResumeGate
from meridian.review import keyword_reader, reviewer
from meridian.risk_server import (
    RISK_TERMS_2027_06,
    RISK_TERMS_2027_08,
    SCREEN_PROMPT_TOKENS,
    RiskPartner,
    ScreeningModel,
)
from meridian.samples import clean_claim, suspect_claim
from meridian.trust import MERIDIAN_TRUST
import northgate.server as northgate_server
from northgate.server import NORTHGATE_URL, NorthgatePartner
from seamkit.approval import encode_approval
from seamkit.costs import CostLedger, tokens
from seamkit.delegation import HEADER, as_peer, chain_id, decode, inbound_chain
from seamkit.drill import Clock
from seamkit.errors import TaskUnfinished
from seamkit.gateway import Gateway, GatewayConfig, Runtime
from seamkit.receipt import ReceiptLog, Receipts
from seamkit.remote import CrossingRecord
from seamkit.resume import wake
from seamkit.revocation import checked, checker

HERE = Path(__file__).parent / "gateway"
AUGUST = datetime(2027, 8, 2, 9, 0, tzinfo=UTC).timestamp()

# CLM-1003, the claim Chapter 19's regulator asked about: 14,200, reported nine days
# after the accident, by a claimant with one earlier claim, for a bumper at SHOP-3.
CLM_1003: ClaimState = {"claim_id": "CLM-1003", "amount": 14_200.0, "days_to_report": 9,
                        "prior_claims": 1, "damaged_parts": ["bumper"]}
BOOKED = ("SHOP-3", ["bumper"])  # where CLM-1003's car is: a test books it on Northgate


def three_claims() -> list[ClaimState]:
    return [clean_claim(), suspect_claim(), dict(CLM_1003)]  # type: ignore[list-item]


class ScriptedModel(ScreeningModel):
    """Risk's screening model as Chapter 19's tests script it: 79 on CLM-1003 with three
    signals, and Chapter 1's rules on every other claim. No model key is needed."""

    def screen(self, claim: dict[str, Any], history: list[dict[str, Any]],
               config: RunnableConfig) -> tuple[float, list[str]]:
        if claim["claim_id"] != CLM_1003["claim_id"]:
            return super().screen(claim, history, config)
        task = str(config["configurable"].get("task_id", ""))
        reasons = ["reported_late", "high_amount", "repeat_claimant"]
        self.call(task, SCREEN_PROMPT_TOKENS, {"claim": claim},
                  {"score": 79.0, "reasons": reasons})
        return 79.0, reasons


def unusual_repair(claim: dict[str, Any],
                   seen: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Risk's fraud assistant, scripted: it asks about a claim over 10,000, once."""
    if seen or float(claim["amount"]) <= 10_000:
        return None
    return {"claim_id": claim["claim_id"]}


@dataclass(frozen=True)
class Sent:
    """One request on an owner's wire, as the client sent it."""

    seam: str
    caller: str
    method: str
    opens: bool  # a send that opens a task, not a read or a reply
    body: int  # bytes of the whole JSON-RPC request
    authority: int  # bytes of the delegation chain header
    links: int  # links in that chain
    data: Mapping[str, Any]  # the fields the message carried
    seconds: float = 0.0  # wall time on the in-process wire


@dataclass
class Wire:
    """Every request each owner's clients sent, timed on the wall clock."""

    sent: list[Sent] = field(default_factory=list)
    started: dict[int, float] = field(default_factory=dict)

    def watch(self, client: httpx.AsyncClient, seam: str, caller: str) -> None:
        async def request(req: httpx.Request) -> None:
            self.started[id(req)] = time.perf_counter()

        async def response(resp: httpx.Response) -> None:
            req = resp.request
            began = self.started.pop(id(req), None)
            if req.method != "POST" or began is None:
                return
            await resp.aread()
            body = json.loads(req.content)
            message = body.get("params", {}).get("message", {})
            data: dict[str, Any] = {}
            for part in message.get("parts", []):
                data.update(part.get("data", {}))
            header = req.headers.get(HEADER, "")
            self.sent.append(Sent(
                seam, caller, str(body.get("method", "")),
                body.get("method") == "SendMessage" and not message.get("taskId"),
                len(req.content), len(header), len(decode(header)), data,
                time.perf_counter() - began))

        client.event_hooks["request"].append(request)
        client.event_hooks["response"].append(response)


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid, "session": ADJUSTER_SESSION}}


@dataclass
class Assembled:
    """The federation, every owner behind its own gateway, on one clock."""

    hub: bool
    clock: Clock
    issuer: MeridianIssuer
    claims: Gateway
    risk: Gateway
    finance: Gateway
    partners: A2APartners
    graph: ClaimGraph
    wire: Wire
    records: list[CrossingRecord]  # Claims' crossing records
    risk_records: list[CrossingRecord]  # Risk's, on the peer seam
    runs: int = 0
    states: dict[str, dict[str, Any]] = field(default_factory=dict)  # by thread

    async def run(self, claims: Sequence[ClaimState] | None = None,
                  ) -> dict[str, dict[str, Any]]:
        """Each claim end to end: screened, estimated, approved by both approvers, paid
        or referred. Returns each claim's final state, by thread."""
        out: dict[str, dict[str, Any]] = {}
        for claim in claims or three_claims():
            self.runs += 1
            tid = f"{claim['claim_id']}-{self.runs}"
            state = await self.graph.ainvoke(claim, thread(tid))
            if "__interrupt__" in state:  # Finance asks Claims' claims manager first
                asked = state["__interrupt__"][0].value
                try:
                    await self.graph.ainvoke(Command(resume=self.approval(asked)),
                                             thread(tid))
                except TaskUnfinished:  # Finance's own approver is next
                    pass
                await self.finance_partner.approve(str(asked["task_id"]),
                                                   "finance:fa-12")
                assert await wake(self.graph, thread(tid), set()) == "done"
                state = (await self.graph.aget_state(thread(tid))).values
            out[tid] = dict(state)
        self.states |= out
        return out

    def approval(self, asked: Mapping[str, Any]) -> dict[str, Any]:
        signed = self.issuer.approve(MANAGER_SESSION, str(asked["task_id"]),
                                     str(asked["claim_id"]), float(asked["amount"]),
                                     str(asked["asked"]))
        return {"approved": True, "approval": encode_approval(signed)}

    @property
    def finance_partner(self) -> FinancePartner:
        return self.partners.finance

    @property
    def risk_partner(self) -> RiskPartner:
        return self.partners.risk

    def gateways(self) -> dict[str, Gateway]:
        return {"Claims": self.claims, "Risk": self.risk, "Finance": self.finance}


def traced() -> None:
    """Chapter 17's spans need a tracer. A process has one provider: if nobody set one,
    set OpenTelemetry's SDK, so every send carries its trace context in the message,
    as a traced deployment's does, and the bytes measured include it."""
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())


async def assemble(clock: Clock, hub: bool = False) -> Assembled:
    """Every owner behind its own gateway, its configuration read once."""
    traced()
    root = ROOT if not hub else ROOT.__class__(
        ROOT.skills, ROOT.fields | {"estimate_history"}, ROOT.hops)
    issuer = MeridianIssuer(now=clock, grant=root, sessions={
        **SESSIONS, ADJUSTER_SESSION: (ADJUSTER, frozenset(
            {"CLM-1001", "CLM-1002", "CLM-1003"}))})
    lists = revocation_sources(issuer)
    shape = "hub.toml" if hub else "gateway.toml"
    claims_cfg = GatewayConfig.load(HERE / "claims", shape)
    risk_cfg = GatewayConfig.load(HERE / "risk", shape)
    finance_cfg = GatewayConfig.load(HERE / "finance")
    keys = {**pinned(), "claims": KEYS["claims"].public_key()}

    # Risk: its screening behind its inbound gateway, and in the peer shape its tool
    # behind its outbound one. Its receipts on Northgate go into its own log.
    risk_log, risk_secret = ReceiptLog(), os.urandom(32)
    records = asyncio.Event()
    records.set()  # the police records service answers at once
    risk = RiskPartner(release="2027.08" if hub else "2027.06",
                       terms=RISK_TERMS_2027_08 if hub else RISK_TERMS_2027_06,
                       lookups=records, revocations=lists, now=clock,
                       assistant=None if hub else unusual_repair,
                       model=ScriptedModel(), meters=True,
                       onward=Receipts(KEYS["risk"], "risk", risk_secret, keys, risk_log,
                                       True))
    risk_records: list[CrossingRecord] = []
    risk_gw = Gateway(risk_cfg, Runtime(
        KEYS["risk"], "risk", checked(inbound_chain, checker(lists), clock),
        lambda: chain_id(inbound_chain()), MERIDIAN_TRUST, MERIDIAN_POLICY, keys,
        risk_secret, RATES, risk_log, CostLedger(), risk_records.append, rechecks=True,
        trusts=MERIDIAN_TRUSTS, revoked=checker(lists), now=clock))
    risk.app = risk_gw.inbound(risk.card, risk.executor, meter=risk.meter)

    finance = FinancePartner(release="2026.15", revocations=lists, now=clock)
    finance_gw = Gateway(finance_cfg, Runtime(
        KEYS["finance"], "finance", trusts=MERIDIAN_TRUSTS,
        revoked=checker(finance.sources), now=clock))
    finance.app = finance_gw.inbound(finance.card, finance.executor,
                                     store=finance.executor.store, keys=finance.keys,
                                     tasks=finance.bound, holders=finance.holders)
    finance.guard = finance.app  # type: ignore[assignment]

    assessors = asyncio.Event()
    assessors.set()
    northgate = NorthgatePartner(release="v7.6", assessors=assessors,
                                 revocations=lists, now=clock)
    partners = A2APartners(risk=risk, northgate=northgate, finance=finance)
    apps = {"Risk": risk.app, "Northgate": northgate.app, "Finance": finance.app}
    wire = Wire()

    if risk_cfg.seams:  # the peer shape: Risk's client to Northgate
        http = {s: httpx.AsyncClient(transport=httpx.ASGITransport(
            app=as_peer(northgate.app, "risk")), base_url=NORTHGATE_URL)
            for s in risk_cfg.seams}
        for seam, client in http.items():
            wire.watch(client, seam, "Risk")
        await risk_gw.connect(http)
        risk._tool = risk_gw.tool("estimate_history")

    claims_records: list[CrossingRecord] = []
    claims_gw = Gateway(claims_cfg, Runtime(
        KEYS["claims"], "claims", issuer.run_chain, partial(run_chain_id, issuer),
        MERIDIAN_TRUST, HUB_POLICY if hub else MERIDIAN_POLICY, keys, os.urandom(32),
        RATES, ReceiptLog(), CostLedger(), claims_records.append,
        gate=partial(ResumeGate, issuer=issuer)))
    clients = {seam: in_process_http(sc.url, apps[sc.expects.owner], "claims")
               for seam, sc in claims_cfg.seams.items()}
    for seam, client in clients.items():
        wire.watch(client, seam, "Claims")
    await claims_gw.connect(clients)
    nodes = {seam: claims_gw.outbound(seam) for seam in claims_cfg.seams}
    # Northgate's estimates are proven under this card (Chapter 5's gradient).
    proven = nodes["repair_estimate"].partner.record("estimate_repair")
    proven.clean.update(f"earlier-{n}" for n in range(20))
    for gw in (claims_gw, risk_gw):
        gw.check()
    graph = build_graph(with_review(cast(Mapping[str, Node], nodes),
                                    reviewer(keyword_reader)), InMemorySaver())
    return Assembled(hub, clock, issuer, claims_gw, risk_gw, finance_gw, partners,
                     graph, wire, claims_records, risk_records)


@dataclass(frozen=True)
class Measured:
    """One seam, measured from what the owners recorded. Sizes are medians per task."""

    seam: str
    caller: str  # the owner whose gateway sends on it
    tasks: int  # tasks opened on the seam
    fields: tuple[str, ...]  # the field names a new task sends
    request_bytes: int  # the whole JSON-RPC request that opens a task
    tokens: int  # the stand-in counter on the fields sent
    authority_bytes: int  # the delegation chain header
    links: int  # links in the chain the callee verifies
    median_ms: float
    p99_ms: float  # with fewer than 100 tasks, the slowest
    cost_per_task: float | None  # from the cost ledger; None: nothing priced it
    receipts: int
    ledger: str  # whose ledger holds the task: an owner, or "none"


def _p99(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]


def measure(a: Assembled) -> dict[str, Measured]:
    """Each seam's numbers, from the wire, the receipts, the costs and the ledgers."""
    states = a.states
    logs = {"Claims": a.claims.runtime.receipts, "Risk": a.risk.runtime.receipts}
    costs = {"Claims": a.claims.runtime.costs, "Risk": a.risk.runtime.costs}
    out: dict[str, Measured] = {}
    seams = {s: "Claims" for s in a.claims.config.seams}
    seams |= {s: "Risk" for s in a.risk.config.seams}
    for seam, caller in seams.items():
        opened = [s for s in a.wire.sent if s.seam == seam and s.caller == caller
                  and s.opens]
        if not opened:
            continue
        receipts = [r for r in logs[caller].receipts([seam])]
        gw = a.claims if caller == "Claims" else a.risk
        found = {f for s in opened for f in s.data}  # what each new task re-sends
        sends = gw.config.seams[seam].expects.sends
        named = [f for f in sends if f in found] + sorted(found - set(sends))
        priced = [e.price for e in costs[caller].entries([seam])
                  if e.price is not None]
        held = any(seam in s.get("ledger", {}) for s in states.values())
        ms = [s.seconds * 1000 for s in opened]
        out[seam] = Measured(
            seam, caller, len(opened), tuple(named),
            int(statistics.median(s.body for s in opened)),
            int(statistics.median(tokens(dict(s.data)) for s in opened)),
            int(statistics.median(s.authority for s in opened)),
            max(s.links for s in opened), statistics.median(ms), _p99(ms),
            statistics.mean(priced) if priced else None, len(receipts),
            caller if held else "none")
    return out


@contextmanager
def august(clock: Clock) -> Iterator[Clock]:
    """The run's world: one clock for every owner, and CLM-1003's car booked at SHOP-3.
    Both are put back afterwards, so nothing leaks into another federation."""
    held, booked = seamkit.ledger.CLOCK, northgate_server.BOOKINGS["meridian"]
    seamkit.ledger.CLOCK = clock
    booked[CLM_1003["claim_id"]] = BOOKED
    try:
        yield clock
    finally:
        seamkit.ledger.CLOCK = held
        booked.pop(CLM_1003["claim_id"], None)


async def main() -> None:  # pragma: no cover - prints the closer's numbers
    for hub in (False, True):
        with august(Clock(AUGUST)) as clock:
            a = await assemble(clock, hub)
            for _ in range(20):
                await a.run()
        print("hub" if hub else "peer")
        for m in measure(a).values():
            print(f"  {m.seam:16} {m.tasks:3} tasks  median {m.median_ms:5.1f} ms  "
                  f"p99 {m.p99_ms:5.1f} ms  {m.request_bytes} B  {m.links} links")


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["AUGUST", "BOOKED", "CLM_1003", "Assembled", "Measured", "ScriptedModel", "august",
           "Sent", "Wire", "assemble", "measure", "three_claims", "unusual_repair"]
