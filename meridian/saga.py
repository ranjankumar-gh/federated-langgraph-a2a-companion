"""Meridian's saga: the four forward steps of a claim, and what undoes each one.

The claim graph gains a booking seam (Northgate's book_repair, after Claims' review and
before the payment), one node, the compensator, and two routing rules: a follow-on
entry is sent before anything else, and a reversed decision goes to the compensator
before the run ends. A forward step whose task ends without committing (failed,
rejected, cancelled at its horizon, or revoked) aborts the saga: the claim goes to an
adjuster, and the steps already taken are undone first.

Two triggers reverse a claim here. `reopen` re-screens a claim whose facts changed, as
a follow-on task on the screening seam (Chapter 9), and a referral follows if the new
score says so. `adopt` gives the claim's thread an orphan the reconciliation job found
(Chapter 12), which the compensator undoes whatever the claim's decision is. The job
itself never writes a ledger.
"""

import operator
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from starlette.types import ASGIApp

import seamkit.ledger
from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import (
    EXPECTATIONS,
    GRANTS,
    SEAMS,
    A2APartners,
    a2a_workers,
    connect_partners,
    in_process_http,
    with_review,
)
from meridian.finance_server import FINANCE_URL
from meridian.identity import CLAIMS_ISSUER, KEYS, MeridianIssuer
from meridian.inprocess import ClaimState, Node, claims_intake
from meridian.review import Reader, keyword_reader, reviewer
from meridian.trust import SAGA_TRUST
from northgate.server import NORTHGATE_URL
from seamkit.contract import Expectations, load_expectations, seam_spec
from seamkit.delegation import Grant, chain_id
from seamkit.errors import (
    DelegationRevoked,
    PartnerFailed,
    PartnerRejected,
    SeamError,
    WaitExpired,
)
from seamkit.ledger import LedgerEntry, follow_on
from seamkit.reconcile import Finding, log_finding
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    connect,
    log_crossing,
)
from seamkit.saga import HOUR, Compensator, SagaPlan, SagaStep, UndoAuth, held_to_plan
from seamkit.trust import PartnerTrust

EXPECTED = Path(__file__).parent / "expectations"
COMPENSATE = "compensate"

MERIDIAN_PLAN = SagaPlan((
    SagaStep("fraud_screening", "screen_fraud", "Risk", "", 0,
             "nothing to undo"),  # a screening changes nothing at Risk
    SagaStep("repair_estimate", "estimate_repair", "Northgate", "", 0,
             "cost accepted"),  # an estimate is billed and kept
    SagaStep("repair_booking", "book_repair", "Northgate", "cancel_booking",
             72 * HOUR,  # a refused approval can come up to 72 h after the booking
             "Claims' adjuster with Northgate's network desk"),
    SagaStep("payments", "pay_claim", "Finance", "void_payment",
             HOUR,  # the reconciliation job finds an orphan within the hour
             "Finance's recovery team, from the payee"),
))

# The forward seams in the saga's order, and the undo seams the compensator sends on.
SAGA_EXPECTATIONS: dict[str, Expectations] = {
    **EXPECTATIONS,
    "repair_booking": load_expectations(EXPECTED / "repair_booking.toml")}
SAGA_SEAMS: dict[str, SeamSpec] = {
    name: SEAMS.get(name) or seam_spec(SAGA_EXPECTATIONS[name])
    for name in ("fraud_screening", "repair_estimate", "repair_booking", "payments")}
UNDO_EXPECTATIONS: dict[str, Expectations] = {
    step.seam: load_expectations(EXPECTED / f"{step.seam}.undo.toml")
    for step in MERIDIAN_PLAN.steps if step.compensated_by}
UNDO_GRANTS: dict[str, Grant] = {
    seam: Grant(frozenset({e.skill}), frozenset(e.sends), hops=0)
    for seam, e in UNDO_EXPECTATIONS.items()}

# The state fields only the saga's graph writes.
Undone = Annotated[dict[str, str], operator.or_]


class SagaState(ClaimState, total=False):
    booking_ref: str
    aborted: str  # the forward step that ended without committing, and why
    undone: Undone  # ledger key -> how its step was settled
    orphans: Annotated[dict[str, str], operator.or_]  # ledger key -> its plan seam


SagaGraph = Any  # a compiled StateGraph over SagaState

# A forward step that ends like this will never commit: the saga aborts.
ABORTS: tuple[type[SeamError], ...] = (PartnerFailed, PartnerRejected, WaitExpired,
                                       DelegationRevoked)


@dataclass
class Aborts:
    """A forward step whose task ended without committing sends the claim to people,
    after the compensator has undone the steps before it."""

    node: Node

    async def __call__(self, state: SagaState) -> Mapping[str, Any]:
        try:
            result = self.node(state)
            return await result if isinstance(result, Awaitable) else result
        except ABORTS as exc:
            return {"decision": "refer",
                    "aborted": f"{exc.seam}: {type(exc).__name__}: {exc.detail}"}


def saga_apps(partners: A2APartners) -> dict[str, tuple[str, ASGIApp]]:
    """Every seam the saga sends on, forward and undo, with its partner's deployment."""
    return {**partners.apps(),
            "repair_booking": (NORTHGATE_URL, partners.northgate.app),
            "repair_booking.undo": (NORTHGATE_URL, partners.northgate.app),
            "payments.undo": (FINANCE_URL, partners.finance.app)}


async def connect_saga(
        apps: Mapping[str, tuple[str, ASGIApp]],
        http: Mapping[str, httpx.AsyncClient] | None = None,
        issuer: MeridianIssuer = CLAIMS_ISSUER,
        trust: Mapping[str, PartnerTrust] = SAGA_TRUST,
        plan: SagaPlan = MERIDIAN_PLAN,
        expects: Mapping[str, Expectations] = SAGA_EXPECTATIONS,
) -> tuple[dict[str, PartnerConnection], dict[str, PartnerConnection]]:
    """The forward connections, held to the plan, and one undo connection per row.

    A forward seam whose partner does not promise the plan's undo is breached: it
    refuses to send. An undo connection sends every call under a compensation link:
    the issuer's root for Claims and the claim, narrowed to one skill and one task.
    """
    forward = {seam: apps[seam] for seam in SAGA_SEAMS}
    connections = held_to_plan(plan, await connect_partners(
        forward, http, issuer, trust, expects, GRANTS))
    undo: dict[str, PartnerConnection] = {}
    for seam, wants in UNDO_EXPECTATIONS.items():
        url, app = apps[f"{seam}.undo"]
        client = (http or {}).get(f"{seam}.undo") or in_process_http(url, app, "claims")
        undo[seam] = conn = await connect(url, client, wants, MERIDIAN_POLICY,
                                          trust=trust[wants.owner])
        assert conn.trust is not None
        client.auth = UndoAuth("claims", KEYS["claims"], wants.owner.lower(),
                               UNDO_GRANTS[seam], issuer.compensation_chain, conn.card,
                               conn.trust.covers)
    return connections, undo


def compensator(undo: Mapping[str, PartnerConnection],
                log: Callable[[CrossingRecord], None] = log_crossing,
                issuer: MeridianIssuer = CLAIMS_ISSUER,
                findings: Callable[[Finding], None] = log_finding,
                plan: SagaPlan = MERIDIAN_PLAN,
                forward: Mapping[str, RemoteAgentNode] | None = None) -> Compensator:
    """The compensator, with one RemoteAgentNode per undo seam."""
    nodes = {seam: RemoteAgentNode(seam_spec(UNDO_EXPECTATIONS[seam]), conn, log)
             for seam, conn in undo.items()}
    return Compensator(plan, nodes, lambda: chain_id(issuer.compensation_chain()),
                       findings, dict(forward or {}))


def build_saga_graph(workers: Mapping[str, Node], undo: Compensator | None,
                     checkpointer: BaseCheckpointSaver[Any] | None = None,
                     schema: type[SagaState] = SagaState) -> SagaGraph:
    """Chapter 1's hub and spokes, with the compensator. `undo=None` is the incident's
    graph: it refers a reversed claim and ends, and undoes nothing. `schema` is a
    later chapter's state, which adds fields to SagaState."""
    steps = {name: Aborts(node) for name, node in workers.items()}

    def next_step(state: SagaState) -> str:
        ledger, done = state.get("ledger", {}), state.get("completed", [])
        for name in steps:  # a follow-on entry (Chapter 9) waits to be sent
            entry = ledger.get(name)
            if name in done and entry and entry["state"] == "intended":
                return name
        if "decision" in state:
            return COMPENSATE if undo and undo.pending(state) else END
        return next((w for w in steps if w not in done), END)

    graph = StateGraph(schema)
    graph.add_node("claims_intake", claims_intake)
    for name, node in steps.items():
        graph.add_node(name, node)
        graph.add_edge(name, "claims_intake")  # every worker reports to the hub
    targets = [*steps, END]
    if undo is not None:
        graph.add_node(COMPENSATE, cast(Node, undo))
        graph.add_conditional_edges(
            COMPENSATE, lambda s: COMPENSATE if undo.pending(s) else END,
            [COMPENSATE, END])
        targets.append(COMPENSATE)
    graph.add_edge(START, "claims_intake")
    graph.add_conditional_edges("claims_intake", next_step, targets)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


async def build_saga(partners: A2APartners,
                     http: Mapping[str, httpx.AsyncClient] | None = None,
                     checkpointer: BaseCheckpointSaver[Any] | None = None,
                     issuer: MeridianIssuer = CLAIMS_ISSUER,
                     log: Callable[[CrossingRecord], None] = log_crossing,
                     findings: Callable[[Finding], None] = log_finding,
                     reader: Reader = keyword_reader,
                     undoes: bool = True,
                     ) -> tuple[SagaGraph, dict[str, RemoteAgentNode], Compensator]:
    """Connect every seam, and build the claim graph with its compensator."""
    connections, undo = await connect_saga(saga_apps(partners), http, issuer)
    nodes = a2a_workers(connections, log, SAGA_SEAMS, issuer)
    undoer = compensator(undo, log, issuer, findings, forward=nodes)
    workers = with_review(cast(Mapping[str, Node], nodes), reviewer(reader))
    return (build_saga_graph(workers, undoer if undoes else None, checkpointer),
            dict(nodes), undoer)


async def reopen(graph: SagaGraph, config: RunnableConfig,
                 **changed: Any) -> dict[str, Any]:
    """A claim's facts changed after screening: re-screen it, as a follow-on task.

    A new run on the claim's own thread, with the corrected facts and the screening
    seam's follow-on entry. Whatever the new score decides, the run decides.
    """
    snapshot = await graph.aget_state(config)
    ledger: Mapping[str, LedgerEntry] = snapshot.values["ledger"]
    again = follow_on(ledger["fraud_screening"])
    out: dict[str, Any] = await graph.ainvoke(
        {**changed, "ledger": {"fraud_screening": again}}, config)
    return out


async def adopt(graph: SagaGraph, config: RunnableConfig,
                finding: Finding) -> dict[str, Any]:
    """The claim's thread takes an orphan the reconciliation job found, and undoes
    it. The job never writes a ledger; the thread's own run does."""
    key = f"{finding.seam}#{finding.task_id[:8]}"
    entry = LedgerEntry(seam=finding.seam, skill=MERIDIAN_PLAN.step(finding.seam).skill,
                        idempotency_key=finding.idempotency_key, context_id="",
                        task_id=finding.task_id, card_digest="",
                        authority=finding.authority,
                        partner_release=finding.partner_release, state="unknown",
                        seen_at=seamkit.ledger.now(), wrote=[], earlier=[], version=1)
    out: dict[str, Any] = await graph.ainvoke(
        {"ledger": {key: entry}, "orphans": {key: finding.seam}}, config)
    return out


__all__ = ["ABORTS", "COMPENSATE", "MERIDIAN_PLAN", "SAGA_EXPECTATIONS", "SAGA_SEAMS",
           "UNDO_EXPECTATIONS", "Aborts", "SagaState", "adopt", "build_saga",
           "build_saga_graph", "compensator", "connect_saga", "reopen", "saga_apps"]

