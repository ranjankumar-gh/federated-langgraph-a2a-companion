"""Finance's payments over A2A: a one-node LangGraph graph behind GraphExecutor."""

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from starlette.applications import Starlette

from meridian.agents import pay
from seamkit.server import GraphExecutor, a2a_app, agent_card

FINANCE_URL = "http://finance.meridian.test/"


class PaymentState(TypedDict, total=False):
    claim: dict[str, Any]
    payment_ref: str


def make_payment(state: PaymentState) -> PaymentState:
    return {"payment_ref": pay(state["claim"])}


@dataclass
class FinancePartner:
    release: str = "2026.10"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)

    def __post_init__(self) -> None:
        graph = StateGraph(PaymentState)
        graph.add_node("make_payment", make_payment)
        graph.add_edge(START, "make_payment")
        graph.add_edge("make_payment", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)
        self.executor = GraphExecutor(
            self.graph, "payments",
            to_input=lambda claim: {"claim": claim},
            to_artifact=lambda out: {"payment_ref": out["payment_ref"]})
        self.card = agent_card("Finance payments", FINANCE_URL, self.release,
                               {"pay_claim": "Pay a claim against its estimate"})
        self.app: Starlette = a2a_app(self.card, self.executor)
