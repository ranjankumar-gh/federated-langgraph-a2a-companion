"""Meridian in one deployment: a supervisor and three workers sharing one ClaimState.

This is the hub-and-spoke shape of the LangGraph book's supervisor chapter. One owner
runs every node, so one checkpointer sees every write and one credential covers every
call.
"""

import operator
from collections.abc import Mapping
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meridian.agents import estimate_repair, pay, screen_fraud


class ClaimState(TypedDict, total=False):
    claim_id: str
    amount: float
    days_to_report: int
    prior_claims: int
    damaged_parts: list[str]
    fraud_score: float  # 0-100: the scale the Claims threshold was written against
    fraud_signals: list[str]
    repair_estimate: float
    payment_ref: str
    decision: str  # "paid" or "refer"
    completed: Annotated[list[str], operator.add]


CLAIM_FIELDS = frozenset(ClaimState.__annotations__)
FRAUD_THRESHOLD = 70.0


class Node(Protocol):
    def __call__(self, state: ClaimState) -> ClaimState: ...


ClaimGraph = CompiledStateGraph[ClaimState, None, ClaimState, ClaimState]


def claims_intake(state: ClaimState) -> ClaimState:
    # A missing score reads as 0.0. In one deployment it is never missing.
    if state.get("fraud_score", 0.0) > FRAUD_THRESHOLD:
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
