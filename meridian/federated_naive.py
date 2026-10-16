"""Meridian, naively federated: the same graph, three workers behind owner lines.

Each partner is a separately constructed runtime with its own state store, its own
release and its own log, and it accepts JSON only. Nothing crosses the line but a
string. The caller keeps every habit it had in one deployment, which is the point:
Chapter 1 shows what those habits cost, and later chapters replace them one seam
property at a time.
"""

import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from meridian.agents import estimate_repair, pay, screen_fraud
from meridian.inprocess import CLAIM_FIELDS, ClaimGraph, ClaimState, Node, build_graph

CLAIMS_CREDENTIAL = "claims-team-service-token"

OWNERS: dict[str, str] = {
    "claims_intake": "Claims",
    "fraud_screening": "Risk",
    "repair_estimate": "Northgate",
    "payments": "Finance",
}


class PartnerRuntime:
    """One owner's deployment seen from outside: JSON request in, JSON reply out."""

    def __init__(self, owner: str, release: str) -> None:
        self.owner = owner
        self.release = release
        # The partner's own log. The caller has no way to read it.
        self.received: list[dict[str, Any]] = []

    def handle(self, body: str, credential: str | None) -> str:
        request = json.loads(body)
        self.received.append(
            {"request": request, "credential": credential, "release": self.release})
        return json.dumps(self.answer(request))

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RiskState(TypedDict, total=False):
    claim: dict[str, Any]
    fraud_score: float
    fraud_signals: list[str]


class RiskRuntime(PartnerRuntime):
    """The Risk team's fraud screening: LangGraph, with its own checkpointer.

    Release 2026.09 returns the score on 0-100. Release 2026.10 returns it on 0-1, under
    the same field name and the same JSON type.
    """

    def __init__(self, release: str = "2026.09", available: bool = True) -> None:
        super().__init__("Risk", release)
        self.available = available
        self.checkpointer = InMemorySaver()
        graph = StateGraph(RiskState)
        graph.add_node("screen", self._screen)
        graph.add_edge(START, "screen")
        graph.add_edge("screen", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _screen(state: RiskState) -> RiskState:
        score, signals = screen_fraud(state["claim"])
        return {"fraud_score": score, "fraud_signals": signals}

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return {"error": "upstream timeout"}
        config: RunnableConfig = {"configurable": {"thread_id": request["claim_id"]}}
        risk_input: RiskState = {"claim": request}
        out = self.graph.invoke(risk_input, config)
        score = float(out["fraud_score"])
        if self.release >= "2026.10":
            score = score / 100
        return {"fraud_score": score}  # the signals stay in Risk's checkpoint


class NorthgateService(PartnerRuntime):
    """Northgate Body Shop Network's repair estimate. Plain Python, not LangGraph."""

    def __init__(self, release: str = "v7") -> None:
        super().__init__("Northgate", release)
        self.estimates: dict[str, float] = {}

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        estimate = estimate_repair(request)
        self.estimates[request["claim_id"]] = estimate
        return {"repair_estimate": estimate}


class FinanceRuntime(PartnerRuntime):
    """Finance's payments runtime."""

    def __init__(self, release: str = "2026.10") -> None:
        super().__init__("Finance", release)
        self.ledger: dict[str, str] = {}

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        ref = pay(request)
        self.ledger[request["claim_id"]] = ref
        return {"payment_ref": ref}


@dataclass
class Partners:
    risk: RiskRuntime = field(default_factory=RiskRuntime)
    northgate: NorthgateService = field(default_factory=NorthgateService)
    finance: FinanceRuntime = field(default_factory=FinanceRuntime)


def remote_node(name: str, partner: PartnerRuntime) -> Node:
    """The naive seam: send the whole state, with our credential, and keep what fits."""

    def call(state: ClaimState) -> ClaimState:
        reply = json.loads(partner.handle(json.dumps(dict(state)), CLAIMS_CREDENTIAL))
        update = {k: v for k, v in reply.items() if k in CLAIM_FIELDS}
        return {**update, "completed": [name]}  # type: ignore[typeddict-item]

    return call


def build_federated_graph(partners: Partners | None = None) -> ClaimGraph:
    p = partners or Partners()
    return build_graph({
        "fraud_screening": remote_node("fraud_screening", p.risk),
        "repair_estimate": remote_node("repair_estimate", p.northgate),
        "payments": remote_node("payments", p.finance),
    })
