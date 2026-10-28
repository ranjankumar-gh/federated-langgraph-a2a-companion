"""The Risk team's fraud screening, served over A2A.

Risk's graph is LangGraph with its own checkpointer, as in Chapter 1. From Chapter 3 it
asks for a police report reference on claims over 10,000 by calling interrupt(), and the
executor turns that pause into an input-required task. The caller never reads Risk's
checkpoint; it sees a task, a question, and in the end one artifact.

Risk publishes its seam terms on its card (Chapter 4): the score is on 0-100.
`rescaled=True` deploys release 2026.10's change, a score on 0-1, whatever the card says.
"""

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, TypedDict

from a2a.types import AgentCard
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from starlette.applications import Starlette

from meridian.agents import screen_fraud
from seamkit.cards import card_digest
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.server import GraphExecutor, a2a_app, agent_card

RISK_URL = "http://risk.meridian.test/"
POLICE_REPORT_OVER = 10_000.0

SCREEN_FRAUD = SkillTerms(
    inputs=(Field("claim_id", "string"), Field("amount", "number"),
            Field("days_to_report", "integer"), Field("prior_claims", "integer")),
    outputs=(Field("fraud_score", "number", bounds=(0.0, 100.0)),),
    asks=("police_report",), p95_seconds=5.0)
RISK_TERMS = SeamTerms("2026.09", date(2026, 9, 14), {"screen_fraud": SCREEN_FRAUD},
                       auth=("mtls", "delegation"), skew_seconds=30,
                       accepts=frozenset({"claim", "vehicle", "claim_history"}))
# The 0-1 change, published honestly: what release 2026.10's card should have said.
RISK_TERMS_0_1 = replace(RISK_TERMS, revision="2026.10", effective=date(2026, 10, 6),
                         skills={"screen_fraud": replace(SCREEN_FRAUD, outputs=(
                             Field("fraud_score", "number", bounds=(0.0, 1.0)),))})


def risk_card(release: str, terms: SeamTerms = RISK_TERMS) -> AgentCard:
    """Risk's card for one release, as Risk's pipeline builds and publishes it."""
    return agent_card("Risk fraud screening", RISK_URL, release,
                      {"screen_fraud": "Fraud score 0-100 for one claim"},
                      extensions=[publish(terms)])


# Every card Risk has released, with the 0-1 card Chapter 4's test publishes honestly.
# Risk's pipeline publishes each card's digest with the release; Claims pins them.
RELEASED: tuple[tuple[str, SeamTerms], ...] = (
    ("2026.09", RISK_TERMS), ("2026.10", RISK_TERMS), ("2026.10", RISK_TERMS_0_1),
    ("2026.11", RISK_TERMS), ("2026.12", RISK_TERMS))
PUBLISHED_DIGESTS = frozenset(card_digest(risk_card(*release)) for release in RELEASED)


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
    terms: SeamTerms = RISK_TERMS
    rescaled: bool = False  # release 2026.10: the score leaves Risk on 0-1

    def __post_init__(self) -> None:
        scale = 100.0 if self.rescaled else 1.0
        self.graph = build_risk_graph(self.checkpointer)
        self.executor = GraphExecutor(
            self.graph, "fraud_screening",
            to_input=lambda claim: {"claim": claim},
            to_artifact=lambda out: {"fraud_score": out["fraud_score"] / scale})
        self.card = risk_card(self.release, self.terms)
        self.app: Starlette = a2a_app(self.card, self.executor)
