"""Meridian in one deployment: a supervisor and three workers sharing one ClaimState.

This is the hub-and-spoke shape of the LangGraph book's supervisor chapter. One owner
runs every node, so one checkpointer sees every write and one credential covers every
call.
"""

import operator
from collections.abc import Awaitable, Mapping
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meridian.agents import estimate_repair, pay, screen_fraud
from seamkit.ledger import LedgerEntry, merge_ledger


class ClaimState(TypedDict, total=False):
    claim_id: str
    amount: float
    days_to_report: int
    prior_claims: int
    damaged_parts: list[str]
    policy: dict[str, Any]  # the policy record the cover check read (Chapter 2)
    covered_parts: list[str]
    fraud_score: float  # 0-100: the scale the Claims threshold was written against
    fraud_signals: list[str]
    repair_estimate: float
    payment_ref: str
    decision: str  # "paid" or "refer"
    completed: Annotated[list[str], operator.add]
    # seam -> Claims' record of the task it sent there; merged with a version check
    ledger: Annotated[dict[str, LedgerEntry], merge_ledger]


CLAIM_FIELDS = frozenset(ClaimState.__annotations__)
FRAUD_THRESHOLD = 70.0


class Node(Protocol):
    # A plain function in one deployment; from Chapter 3 also an awaitable remote call.
    def __call__(
        self, state: ClaimState) -> ClaimState | Awaitable[Mapping[str, Any]]: ...


ClaimGraph = CompiledStateGraph[ClaimState, None, ClaimState, ClaimState]


def claims_intake(state: ClaimState) -> ClaimState:
    screened = "fraud_screening" in state.get("completed", [])
    # No default: after screening, a missing score raises instead of reading as 0.0.
    if screened and state["fraud_score"] > FRAUD_THRESHOLD:
        return {"decision": "refer"}
    if "payment_ref" in state:
        return {"decision": "paid"}
    return {}


def build_graph(workers: Mapping[str, Node],
                checkpointer: BaseCheckpointSaver[Any] | None = None) -> ClaimGraph:
    def next_step(state: ClaimState) -> str:
        if "decision" in state:
            return END
        done = state.get("completed", [])
        return next((w for w in workers if w not in done), END)

    graph = StateGraph(ClaimState)
    graph.add_node("claims_intake", claims_intake)
    for name, node in workers.items():
        graph.add_node(name, node)
        graph.add_edge(name, "claims_intake")  # every worker reports to the hub
    graph.add_edge(START, "claims_intake")
    graph.add_conditional_edges("claims_intake", next_step, [*workers, END])
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def fraud_screening(state: ClaimState) -> ClaimState:
    score, signals = screen_fraud(state)
    return {"fraud_score": score, "fraud_signals": signals,
            "completed": ["fraud_screening"]}


def repair_estimate(state: ClaimState) -> ClaimState:
    return {"repair_estimate": estimate_repair(state), "completed": ["repair_estimate"]}


def payments(state: ClaimState) -> ClaimState:
    return {"payment_ref": pay(state), "completed": ["payments"]}


IN_PROCESS_WORKERS: dict[str, Node] = {
    "fraud_screening": fraud_screening,
    "repair_estimate": repair_estimate,
    "payments": payments,
}


def build_inprocess_graph() -> ClaimGraph:
    return build_graph(IN_PROCESS_WORKERS)
