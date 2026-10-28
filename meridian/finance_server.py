"""Finance's payments over A2A: a one-node LangGraph graph behind GraphExecutor.

Finance publishes its seam terms on its card (Chapter 4). Since Chapter 8, release
2026.11's terms accept an optional remittance memo, which Finance prints on the payment.
Claims cannot see who or what reads it next at Finance.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, TypedDict

from a2a.types import AgentCard
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from starlette.types import ASGIApp

from meridian.agents import pay
from meridian.identity import MERIDIAN_TRUSTS, revocation_sources
from seamkit.cards import card_digest
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.delegation import Guard
from seamkit.revocation import RevocationSource, checker
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
# Chapter 8: one optional input, the memo Finance prints on the remittance.
FINANCE_TERMS_2026_11 = replace(
    FINANCE_TERMS, revision="2026.11", effective=date(2026, 10, 19),
    skills={"pay_claim": replace(FINANCE_TERMS.skills["pay_claim"], inputs=(
        *FINANCE_TERMS.skills["pay_claim"].inputs,
        Field("payment_memo", "string", required=False)))})
TERMS = {"2026.10": FINANCE_TERMS, "2026.11": FINANCE_TERMS_2026_11}


class PaymentState(TypedDict, total=False):
    claim: dict[str, Any]
    payment_ref: str


def make_payment(state: PaymentState) -> PaymentState:
    return {"payment_ref": pay(state["claim"])}


def finance_card(release: str) -> AgentCard:
    """Finance's card for one release, as Finance's pipeline builds and publishes it."""
    return agent_card("Finance payments", FINANCE_URL, release,
                      {"pay_claim": "Pay a claim against its estimate"},
                      extensions=[publish(TERMS[release])])


# Finance's pipeline publishes each release's card digest with the release.
PUBLISHED_DIGESTS = frozenset(card_digest(finance_card(r)) for r in TERMS)


@dataclass
class FinancePartner:
    release: str = "2026.11"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)
    # Chapter 7: Meridian's revocation list, checked by the guard on every call.
    revocations: Mapping[str, RevocationSource] | None = None
    now: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        sources = (revocation_sources() if self.revocations is None
                   else self.revocations)
        graph = StateGraph(PaymentState)
        graph.add_node("make_payment", make_payment)
        graph.add_edge(START, "make_payment")
        graph.add_edge("make_payment", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)
        self.memos: dict[str, str] = {}  # claim -> the memo printed on its payment
        self.executor = GraphExecutor(
            self.graph, "payments",
            to_input=self._received,
            to_artifact=lambda out: {"payment_ref": out["payment_ref"]})
        self.card = finance_card(self.release)
        inner = a2a_app(self.card, self.executor)
        self.guard = Guard(inner, "finance", MERIDIAN_TRUSTS, now=self.now,
                           revoked=checker(sources))
        self.app: ASGIApp = self.guard

    def _received(self, claim: dict[str, Any]) -> PaymentState:
        if "payment_memo" in claim:
            self.memos[str(claim["claim_id"])] = str(claim["payment_memo"])
        return {"claim": claim}
