"""Finance's payments over A2A: a one-node LangGraph graph behind GraphExecutor.

Finance publishes its seam terms on its card (Chapter 4).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from a2a.types import AgentCard
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from starlette.applications import Starlette

from meridian.agents import pay
from seamkit.cards import card_digest
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.server import GraphExecutor, a2a_app, agent_card

FINANCE_URL = "http://finance.meridian.test/"

FINANCE_TERMS = SeamTerms(
    "2026.10", date(2026, 10, 1),
    {"pay_claim": SkillTerms(
        inputs=(Field("claim_id", "string"), Field("amount", "number"),
                Field("repair_estimate", "number", unit="claim currency")),
        outputs=(Field("payment_ref", "string"),), p95_seconds=3.0)},
    auth=("mtls", "delegation"), skew_seconds=30,
    accepts=frozenset({"claim", "estimate"}))


class PaymentState(TypedDict, total=False):
    claim: dict[str, Any]
    payment_ref: str


def make_payment(state: PaymentState) -> PaymentState:
    return {"payment_ref": pay(state["claim"])}


def finance_card(release: str) -> AgentCard:
    """Finance's card for one release, as Finance's pipeline builds and publishes it."""
    return agent_card("Finance payments", FINANCE_URL, release,
                      {"pay_claim": "Pay a claim against its estimate"},
                      extensions=[publish(FINANCE_TERMS)])


# Finance's pipeline publishes each release's card digest with the release.
PUBLISHED_DIGESTS = frozenset({card_digest(finance_card("2026.10"))})


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
        self.card = finance_card(self.release)
        self.app: Starlette = a2a_app(self.card, self.executor)
