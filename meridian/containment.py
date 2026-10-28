"""Meridian's containment: a breaker on every seam, a degraded mode per seam, holds
bounded by the saga plan, and the partner stop.

Chapter 13's saga graph, with each seam's node wrapped in `Contained`. When the seam's
breaker refuses a send, the claim takes the mode the seam's expectations file names:

- screening (Risk) holds: a claim is never cleared unscreened. A screening Risk made
  without estimate history, as its release 2027.02 says in `basis`, may refer a claim
  and may not clear one: such a claim goes to an adjuster;
- estimate (Northgate) substitutes: a Claims desk assessor estimates, and the field is
  tagged as the desk's, never Northgate's;
- booking (Northgate) and payments (Finance) hold.

A held claim's run ends with BreakerOpen at the seam it waits on, and the claim joins
the hold list in Claims' process. `due` names the held threads to ring: those whose
breaker would admit a send now, and those past their hold horizon, which is the
earliest moment a committed step's compensation window closes, less a margin. A claim
woken past its horizon aborts, so the compensator runs while the partners still honour
the undo.

Under a partner stop each rung thread cancels its own open task at the stopped owner,
records the stop on the entry, and takes the seam's mode; a stop that unwinds aborts
every undecided claim with committed work at that owner.
"""

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Annotated, Any, cast

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.types import interrupt

import seamkit.ledger
from meridian.federated import OWNERS_A2A, A2APartners, a2a_workers, with_review
from meridian.identity import CLAIMS_ISSUER, MeridianIssuer
from meridian.inprocess import FRAUD_THRESHOLD, Node
from meridian.review import Reader, keyword_reader, reviewer
from meridian.risk_server import BASIS_DEGRADED
from meridian.saga import (
    EXPECTED,
    MERIDIAN_PLAN,
    SAGA_EXPECTATIONS,
    SAGA_SEAMS,
    SagaGraph,
    SagaState,
    build_saga_graph,
    compensator,
    connect_saga,
    saga_apps,
)
from meridian.trust import SAGA_TRUST
from seamkit.breaker import SeamBreaker, breakers_for
from seamkit.contract import Expectations, load_expectations, seam_spec
from seamkit.errors import BreakerOpen
from seamkit.ingress import Provenance
from seamkit.inventory import SeamInventory
from seamkit.ledger import TERMINAL, LedgerEntry, follow_on, observe
from seamkit.reconcile import Finding, log_finding
from seamkit.remote import CrossingRecord, RemoteAgentNode, SeamSpec, log_crossing
from seamkit.saga import HOUR, Compensator, SagaPlan
from seamkit.stop import PartnerStops

# Claims' expectations for the contained federation: the saga's seams, each with its
# [breaker] table, and the screening seam as Risk's release 2027.02 states it.
CONTAINED_EXPECTATIONS: dict[str, Expectations] = {
    **SAGA_EXPECTATIONS,
    "fraud_screening": load_expectations(EXPECTED / "fraud_screening.contained.toml")}
CONTAINED_SEAMS: dict[str, SeamSpec] = {  # in the saga's order: book before paying
    name: seam_spec(CONTAINED_EXPECTATIONS[name]) for name in SAGA_SEAMS}
# Chapter 11: the estimate's send returns once the task exists, so a frame estimate
# that waits on Northgate's assessor desk is a task in flight, not a send that blocks.
CONTAINED_SEAMS["repair_estimate"] = replace(CONTAINED_SEAMS["repair_estimate"],
                                             return_immediately=True)
DESK = "claims/desk_assessor"  # the Claims role that estimates in Northgate's place
OWNERS = {**OWNERS_A2A, "repair_booking": "Northgate"}


class ContainedState(SagaState, total=False):
    screening_basis: str  # what Risk's screening was made from (Risk 2027.02)
    # seam -> how the claim went on without the seam's partner, and who answered
    degraded: Annotated[dict[str, str], operator.or_]


def hold_horizon(plan: SagaPlan, ledger: Mapping[str, LedgerEntry],
                 margin: float) -> float | None:
    """When the first committed step's undo window closes, less a margin; or None."""
    closes = [seen(ledger[s.seam]) + s.window for s in plan.steps
              if s.compensated_by and s.seam in ledger
              and ledger[s.seam]["state"] == "completed"]
    return min(closes) - margin if closes else None  # None: nothing to undo is at stake


def seen(entry: LedgerEntry) -> float:
    """When Claims saw the entry's state: no earlier than the partner's commit."""
    return datetime.fromisoformat(entry["seen_at"]).timestamp()


def desk_tag(assessor: str) -> Provenance:
    """The tag of an estimate a Claims desk assessor gave in Northgate's place."""
    return Provenance(seam="repair_estimate", task_id="", release="", kind="value",
                      derived_from=[], substitute=assessor)


@dataclass
class Containment:
    """The breakers, the hold list and the stops of one Claims process."""

    breakers: dict[str, SeamBreaker]
    stops: PartnerStops
    plan: SagaPlan = MERIDIAN_PLAN
    margin: float = HOUR  # covers the gap between a commit and Claims seeing it
    held: dict[str, tuple[str, float | None]] = field(default_factory=dict)

    def due(self, now: float) -> list[str]:
        """Held threads to ring: their breaker would admit a send, or time is up."""
        return [thread for thread, (seam, horizon) in self.held.items()
                if not self.breakers[seam].refuses(now)
                or (horizon is not None and now >= horizon)]

    def unwinds(self, state: Mapping[str, Any]) -> str:
        """Why a stop that unwinds aborts this claim, or ""."""
        if "decision" in state:
            return ""
        ledger = state.get("ledger", {})
        for step in self.plan.steps:
            stop = self.stops.unwinding(step.owner)
            entry = ledger.get(step.seam)
            if stop and step.compensated_by and entry and entry["state"] == "completed":
                return stop.name
        return ""


@dataclass
class Contained:
    """A seam's node, with the seam's degraded mode for when its breaker refuses."""

    node: RemoteAgentNode
    seam: str
    room: Containment

    async def __call__(self, state: ContainedState) -> Mapping[str, Any]:
        room, breaker = self.room, self.room.breakers[self.seam]
        if why := room.unwinds(state):
            return self.abort(why)
        entry = state.get("ledger", {}).get(self.seam)
        if entry and entry.get("stopped_by"):  # ended under a stop by this thread
            if breaker.refuses(seamkit.ledger.CLOCK()):
                return self.degrade(state, entry["stopped_by"])
            again = {k: v for k, v in follow_on(entry).items() if k != "stopped_by"}
            return {"ledger": {self.seam: again}}  # released: a new task, a new key
        if breaker.pinned and entry and entry["task_id"] and (
                entry["state"] not in TERMINAL):
            return await self.end_for_stop(entry, breaker.pinned)
        try:
            update = await self.node(state)
        except BreakerOpen as exc:
            return self.degrade(state, exc.detail)
        room.held.pop(thread_of(), None)
        return self.screened(update)

    async def end_for_stop(self, entry: LedgerEntry, stop: str) -> dict[str, Any]:
        """Cancel this thread's own task at a stopped owner, and record the stop."""
        await self.node.cancel(entry["task_id"])  # attempted; the read says what held
        task = await self.node.read(entry["task_id"])
        ended = observe(entry, task, self.node.partner.release)
        if ended["state"] == "completed":  # it committed first: map it on the next pass
            return {"ledger": {self.seam: ended}}
        return {"ledger": {self.seam: {**ended, "stopped_by": stop}}}

    def degrade(self, state: ContainedState, why: str) -> dict[str, Any]:
        mode = self.room.breakers[self.seam].policy.mode
        if mode == "substitute":
            return self.substitute(state, why)
        horizon = hold_horizon(self.room.plan, state.get("ledger", {}),
                               self.room.margin)
        if mode == "abort" or (horizon is not None
                               and seamkit.ledger.CLOCK() >= horizon):
            return self.abort(why if mode == "abort" else "held to the hold horizon")
        self.room.held[thread_of()] = (self.seam, horizon)
        raise BreakerOpen(self.seam, f"held: {why}")  # the run ends; a ring resumes it

    def substitute(self, state: ContainedState, why: str) -> dict[str, Any]:
        """The estimate from a Claims desk assessor, asked as Chapter 10 asks a person."""
        answer = interrupt({"to": DESK, "needs": "repair_estimate", "why": why,
                            "claim_id": state["claim_id"],
                            "damaged_parts": list(state["damaged_parts"])})
        assessor = f"{DESK}:{answer['assessor']}"
        return {"repair_estimate": float(answer["estimate"]),
                "completed": [self.seam],
                "provenance": {"repair_estimate": desk_tag(assessor)},
                "degraded": {self.seam: f"substituted by {assessor}"}}

    def abort(self, why: str) -> dict[str, Any]:
        self.room.held.pop(thread_of(), None)
        return {"decision": "refer", "aborted": f"{self.seam}: {why}",
                "degraded": {self.seam: f"aborted: {why}"}}

    def screened(self, update: Mapping[str, Any]) -> Mapping[str, Any]:
        """A screening made without estimate history may refer a claim, never clear it."""
        degraded = update.get("screening_basis") == BASIS_DEGRADED
        if not degraded or float(update["fraud_score"]) > FRAUD_THRESHOLD:
            return update
        return {**update, "decision": "refer",
                "degraded": {self.seam: f"referred: screened {BASIS_DEGRADED}"}}


def thread_of() -> str:
    return str(get_config()["configurable"]["thread_id"])


def contained_breakers(
        expects: Mapping[str, Expectations] = CONTAINED_EXPECTATIONS,
) -> dict[str, SeamBreaker]:
    """One breaker per seam whose expectations file has a [breaker] table."""
    return breakers_for({seam: (e.owner, e.skill, e.breaker)
                         for seam, e in expects.items() if e.breaker})


async def build_contained(
        partners: A2APartners,
        http: Mapping[str, httpx.AsyncClient] | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        issuer: MeridianIssuer = CLAIMS_ISSUER,
        log: Callable[[CrossingRecord], None] = log_crossing,
        findings: Callable[[Finding], None] = log_finding,
        reader: Reader = keyword_reader,
        undo_door: bool = True,
) -> tuple[SagaGraph, dict[str, RemoteAgentNode], Compensator, Containment]:
    """The saga's graph with every seam contained. `undo_door=False` deploys the stop
    that pins the compensating skills' seams too: the mistake the drill keeps."""
    connections, undo = await connect_saga(saga_apps(partners), http, issuer,
                                           SAGA_TRUST, expects=CONTAINED_EXPECTATIONS)
    breakers = contained_breakers()
    nodes = a2a_workers(connections, log, CONTAINED_SEAMS, issuer)
    for name, node in nodes.items():
        node.breaker = breakers[name]
    undoer = compensator(undo, log, issuer, findings, forward=nodes)
    stopped = dict(breakers)
    if not undo_door:  # the wrong design: the undo seams carry breakers the stop pins
        for seam, node in undoer.undo.items():
            node.breaker = stopped[f"{seam}.undo"] = SeamBreaker(
                f"{seam}.undo", node.seam.owner, node.seam.skill,
                breakers[seam].policy)
    room = Containment(breakers, PartnerStops(stopped))
    workers: dict[str, Node] = {
        name: cast(Node, Contained(node, name, room)) for name, node in nodes.items()}
    graph = build_saga_graph(with_review(workers, reviewer(reader)), undoer,
                             checkpointer, ContainedState)
    return graph, dict(nodes), undoer, room


def with_breaker(inventory: SeamInventory,
                 expects: Mapping[str, Expectations] = CONTAINED_EXPECTATIONS,
                 ) -> SeamInventory:
    """Fill each seam's failure answer from its [breaker] table, never by hand."""
    for entry in inventory.entries:
        policy = (e := expects.get(entry.callee)) and e.breaker
        if not policy:
            continue
        entry.answers["failure"] = (
            f"breaker: {policy.failures} in {policy.window_seconds:g} s opens it, "
            f"late after {policy.late_seconds:g} s, trial "
            f"{policy.trial or 'none'}, {policy.max_in_flight} in flight; {policy.mode}")
    return inventory


__all__ = ["CONTAINED_EXPECTATIONS", "CONTAINED_SEAMS", "DESK", "OWNERS",
           "Contained", "ContainedState", "Containment", "build_contained",
           "contained_breakers", "desk_tag", "hold_horizon", "with_breaker"]
