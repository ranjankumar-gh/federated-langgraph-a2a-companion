"""The delegation revocation drill on Meridian: Chapter 7's incident, replayed.

Every owner reads one clock, so twenty-two minutes pass inside a test. At 10:00 the
adjuster orders CLM-1002's frame-and-door estimate, which Northgate's assessor desk
takes about twenty minutes over. At 10:02:30 Claims reads the open task. At `at` the
adjuster is revoked. From then on, every ten seconds, the drill presents her chain to
each owner, the way a partner still holding it would, and records the first refusal.
At 10:06 a run queued under her session for CLM-1001 starts; at 10:21 the assessor
finishes; at 10:22 Claims reads the estimate task again.

`DrillFederation(fixed=False)` deploys Chapter 6's federation: an issuer whose cache
still hands out her root, partners with no revocation list, and a Northgate that checked
her chain once, when the estimate arrived.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from meridian.federated import (
    GRANTS,
    SEAMS,
    A2APartners,
    build_a2a_graph,
    connect_partners,
)
from meridian.finance_server import FinancePartner
from meridian.identity import (
    ADJUSTER_SESSION,
    KEYS,
    LIST_EVERY,
    LIST_GRACE,
    MeridianIssuer,
    revocation_sources,
)
from meridian.incident import CachingIssuer
from meridian.inprocess import ClaimGraph, ClaimState
from meridian.risk_server import RiskPartner
from meridian.samples import clean_claim, suspect_claim
from northgate.server import NorthgatePartner
from seamkit.delegation import Chain, as_peer, mint
from seamkit.drill import Clock, DrillRow, probe
from seamkit.errors import DelegationRefused, SeamError
from seamkit.remote import CrossingRecord, PartnerConnection
from langchain_core.runnables.config import var_child_runnable_config

T0 = datetime(2026, 10, 13, 10, 0, tzinfo=UTC).timestamp()  # 10:00
MINUTE = 60.0
PROBE_EVERY = 10.0  # seconds between presentations of her chain
OWNERS = {"Risk": "fraud_screening", "Northgate": "repair_estimate",
          "Finance": "payments"}
# Claims' own seam that returns as soon as the task exists: a twenty-minute
# estimate cannot be waited for inside a ten-second send.
ESTIMATE_ONLY = {"repair_estimate": replace(SEAMS["repair_estimate"],
                                            return_immediately=True)}


def adjusters_run(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid, "session": ADJUSTER_SESSION}}


@dataclass
class DrillFederation:
    """Meridian on one clock: the issuer, the three partners, and Claims' graphs."""

    fixed: bool = True  # False: Chapter 6's federation, before revocation existed
    clock: Clock = field(default_factory=lambda: Clock(T0))

    def __post_init__(self) -> None:
        self.issuer: MeridianIssuer = (MeridianIssuer(now=self.clock) if self.fixed
                                       else CachingIssuer(now=self.clock))
        self.assessors = asyncio.Event()  # set: the assessor has finished CLM-1002

        def lists() -> dict[str, Any]:  # each partner pulls its own copy
            return revocation_sources(self.issuer) if self.fixed else {}

        self.partners = A2APartners(
            risk=RiskPartner(revocations=lists(), now=self.clock),
            northgate=NorthgatePartner(assessors=self.assessors, revocations=lists(),
                                       now=self.clock, rechecks=self.fixed),
            finance=FinancePartner(revocations=lists(), now=self.clock))
        self.records: list[tuple[float, CrossingRecord]] = []  # Claims' own, timed
        self.outcomes: dict[str, str] = {}  # Claims' runs, by claim, as they ended
        self.connections: dict[str, PartnerConnection] = {}

    def log(self, record: CrossingRecord) -> None:
        self.records.append((self.clock(), record))

    async def connect(self) -> tuple[ClaimGraph, ClaimGraph]:
        self.connections = await connect_partners(self.partners.apps(),
                                                  issuer=self.issuer)
        common: dict[str, Any] = {"log": self.log, "issuer": self.issuer,
                                  "gates": self.fixed}
        estimate = build_a2a_graph(self.connections, seams=ESTIMATE_ONLY, **common)
        claims = build_a2a_graph(self.connections, **common)
        return estimate, claims

    async def run(self, graph: ClaimGraph, claim: ClaimState, tid: str) -> str:
        """One Claims run under the adjuster's session; how it ended, by name."""
        try:
            out = await graph.ainvoke(claim, adjusters_run(tid))
        except SeamError as exc:
            outcome = type(exc).__name__
        else:
            outcome = out.get("decision") or ("estimate written"
                                              if "repair_estimate" in out else "done")
        self.outcomes[str(claim["claim_id"])] = outcome
        return outcome

    def held_chains(self) -> dict[str, Chain]:
        """The chain each partner holds for her, as Claims minted it at 10:00."""
        token = var_child_runnable_config.set(adjusters_run("drill-probe"))
        try:
            root = self.issuer.run_chain()
        finally:
            var_child_runnable_config.reset(token)
        return {owner: (*root, mint(root, KEYS["claims"], "claims", owner.lower(),
                                    GRANTS[seam], frozenset({"CLM-1002"})))
                for owner, seam in OWNERS.items()}

    def probes(self) -> dict[str, httpx.AsyncClient]:
        apps = self.partners.apps()
        return {owner: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=as_peer(apps[seam][1], "claims")),
            base_url=apps[seam][0]) for owner, seam in OWNERS.items()}

    async def settle(self) -> None:
        """Let Northgate's assessor step finish whatever it does with the estimate."""
        executor = self.partners.northgate.executor
        for _ in range(200):
            if not executor.pending and executor.acts:
                return
            await asyncio.sleep(0.005)


async def revocation_drill(fed: DrillFederation, subject: str,
                           at: float) -> list[DrillRow]:
    """Revoke `subject` at `at` on the running federation; one row per owner."""
    estimate, claims = await fed.connect()
    await fed.run(estimate, suspect_claim(), "clm-1002")  # 10:00: Northgate queues it
    chains, probes = fed.held_chains(), fed.probes()
    fed.clock.t = T0 + 2.5 * MINUTE
    await fed.run(estimate, suspect_claim(), "clm-1002")  # 10:02:30: still working
    fed.clock.t = at
    fed.issuer.revoke(subject, at)
    first: dict[str, tuple[float, str]] = {}
    schedule = [(T0 + 6 * MINUTE, "CLM-1001 run"), (T0 + 21 * MINUTE, "assessor"),
                (T0 + 22 * MINUTE, "CLM-1002 read")]
    while fed.clock() <= T0 + 22 * MINUTE:
        while schedule and schedule[0][0] <= fed.clock():
            _, event = schedule.pop(0)
            if event == "CLM-1001 run":  # queued under her session before 10:03
                await fed.run(claims, clean_claim(), "clm-1001")
            elif event == "assessor":
                fed.assessors.set()
                await fed.settle()
            else:
                fed.outcomes["CLM-1002 at 10:22"] = await fed.run(
                    estimate, suspect_claim(), "clm-1002")
        if "Claims" not in first:
            try:
                with _as_run(adjusters_run("drill-probe")):
                    fed.issuer.run_chain()
            except DelegationRefused as exc:
                first["Claims"] = (fed.clock() - at, exc.detail)
        for owner, http in probes.items():
            if owner not in first and (why := await probe(http, chains[owner])):
                first[owner] = (fed.clock() - at, why)
        fed.clock.advance(PROBE_EVERY)
    return report(fed, first, at)


def report(fed: DrillFederation, first: Mapping[str, tuple[float, str]],
           at: float) -> list[DrillRow]:
    northgate = fed.partners.northgate
    term = fed.connections["repair_estimate"].terms[0].revocation_seconds
    agreed = {"Claims": 0.0, "Risk": LIST_EVERY + LIST_GRACE,
              "Northgate": float(term) if term is not None else None,
              "Finance": LIST_EVERY + LIST_GRACE}
    guards = {"Risk": fed.partners.risk.guard, "Northgate": northgate.guard,
              "Finance": fed.partners.finance.guard}
    rows = []
    for owner in ("Claims", "Risk", "Northgate", "Finance"):
        after = at + (agreed[owner] or 0.0)
        if owner == "Claims":
            acted = [f"{r.call} {r.seam}" for t, r in fed.records
                     if t > after and r.call == "send" and not r.error]
            tasks = {k: v for k, v in fed.outcomes.items() if k != "CLM-1002"}
        else:
            acted = [f"{e['skill'] or e['method']} accepted" for e in guards[owner].log
                     if e["at"] > after and not e["refused"]
                     and e["method"] == "SendMessage"]
            tasks = {}
        if owner == "Northgate":
            acted += [f"estimate committed, billed {cost}"
                      for task, cost in northgate.executor.billed.items()
                      if _committed_at(northgate, task) > after]
            tasks = {f"{claim} estimate": _ended(northgate, task)
                     for task, claim in _estimate_tasks(northgate)}
        when, why = first.get(owner, (None, ""))
        rows.append(DrillRow(owner, when, why, tuple(acted), tasks, agreed[owner]))
    return rows


def _committed_at(northgate: NorthgatePartner, task: str) -> float:
    return max((a["at"] for a in northgate.executor.acts
                if a["task_id"] == task and a["holds"] == "ok"), default=0.0)


def _estimate_tasks(northgate: NorthgatePartner) -> list[tuple[str, str]]:
    """Every estimate task that reached an act point, with its claim, in claim order."""
    return sorted({(a["task_id"], a["claim_id"]) for a in northgate.executor.acts},
                  key=lambda pair: pair[1])


def _ended(northgate: NorthgatePartner, task: str) -> str:
    executor = northgate.executor
    if task in executor.billed:
        return f"completed, billed {executor.billed[task]}"
    if task in executor.canceled:
        return "canceled, not billed"
    return "still open"


class _as_run:
    """A direct call outside a graph, seeing a run's config (the issuer reads it)."""

    def __init__(self, config: RunnableConfig) -> None:
        self.config = config

    def __enter__(self) -> None:
        self.token = var_child_runnable_config.set(self.config)

    def __exit__(self, *exc: object) -> None:
        var_child_runnable_config.reset(self.token)
