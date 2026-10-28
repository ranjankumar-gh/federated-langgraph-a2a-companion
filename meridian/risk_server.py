"""The Risk team's fraud screening, served over A2A.

Risk's graph is LangGraph with its own checkpointer, as in Chapter 1. From Chapter 3 it
asks for a police report reference on claims over 10,000 by calling interrupt(), and the
executor turns that pause into an input-required task. The caller never reads Risk's
checkpoint; it sees a task, a question, and in the end one artifact.
"""

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from starlette.applications import Starlette

from meridian.agents import screen_fraud
from seamkit.server import GraphExecutor, a2a_app, agent_card

RISK_URL = "http://risk.meridian.test/"
POLICE_REPORT_OVER = 10_000.0


class RiskScreenState(TypedDict, total=False):
    claim: dict[str, Any]
    police_report: str
    fraud_score: float
    fraud_signals: list[str]


def require_police_report(state: RiskScreenState) -> RiskScreenState:
    if float(state["claim"]["amount"]) <= POLICE_REPORT_OVER:
        return {}
    answer = interrupt({"needs": "police_report",
                        "why": "claims over 10,000 need a police report"})
    return {"police_report": str(answer["police_report"])}


def screen(state: RiskScreenState) -> RiskScreenState:
    score, signals = screen_fraud(state["claim"])
    return {"fraud_score": score, "fraud_signals": signals}


def build_risk_graph(checkpointer: InMemorySaver) -> Any:
    graph = StateGraph(RiskScreenState)
    graph.add_node("require_police_report", require_police_report)
    graph.add_node("screen", screen)
    graph.add_edge(START, "require_police_report")
    graph.add_edge("require_police_report", "screen")
    graph.add_edge("screen", END)
    return graph.compile(checkpointer=checkpointer)


@dataclass
class RiskPartner:
    """Risk's deployment: its graph, its checkpointer, its executor, its app."""

    release: str = "2026.09"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)

    def __post_init__(self) -> None:
        self.graph = build_risk_graph(self.checkpointer)
        self.executor = GraphExecutor(
            self.graph, "fraud_screening",
            to_input=lambda claim: {"claim": claim},
            to_artifact=lambda out: {"fraud_score": out["fraud_score"]})
        self.card = agent_card("Risk fraud screening", RISK_URL, self.release,
                               {"screen_fraud": "Fraud score 0-100 for one claim"})
        self.app: Starlette = a2a_app(self.card, self.executor)
