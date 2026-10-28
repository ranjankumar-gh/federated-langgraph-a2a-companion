"""The Risk team's fraud screening, served over A2A.

Risk's graph is LangGraph with its own checkpointer, as in Chapter 1. Release 2026.09
asks for a police report reference on claims over 10,000 by calling interrupt(), and the
executor turns that pause into an input-required task. Release 2026.12 stops asking and
looks the reference up in the police records service instead, which is slow. The caller
never reads Risk's checkpoint; it sees a task, perhaps a question, and one artifact.

Risk's fraud assistant, when deployed, may call Northgate's estimate_history as a remote
skill tool: the peer seam. `assistant` stands in for the model's tool choices. Since
Chapter 6, Risk's tool client can only mint a link under the chain Risk verified on the
screening request; `forwards=True` deploys the incident's client instead, which sends
on the Authorization header the screening request arrived with.

Risk publishes its seam terms on its card (Chapter 4): the score is on 0-100.
`rescaled=True` deploys release 2026.10's change, a score on 0-1, whatever the card says.

Since Chapter 8, the history tool's answer passes the seam ingress filter under Risk's
own rules: every field is closed, rows keep only the keys Risk names, and a shop's
free-text remark never reaches the assistant. `filtered=False` deploys Chapter 7's tool.
"""

import asyncio
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import httpx
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from starlette.types import ASGIApp

from meridian.agents import screen_fraud
from meridian.identity import KEYS, MERIDIAN_TRUSTS, revocation_sources
from meridian.incident import ForwardingAuth
from meridian.rates import RATES
from seamkit.breaker import BreakerPolicy, SeamBreaker
from seamkit.cards import card_digest
from seamkit.contract import Field, SeamTerms, SkillTerms, publish, revised
from seamkit.costs import Ceiling, Meter, RateCard, tokens
from seamkit.delegation import (
    DelegationAuth,
    Grant,
    Guard,
    as_peer,
    chain_id,
    inbound_chain,
)
from seamkit.errors import SeamError
from seamkit.evidence import Opening, Witness, genuine, shut, signature, witness_head
from seamkit.ingress import IngressRule
from seamkit.receipt import Receipts, digest, nonce_for
from seamkit.remote import CrossingRecord, connect
from seamkit.revocation import RevocationSource, checked, checker
from seamkit.server import (
    Countersigner,
    GraphExecutor,
    a2a_app,
    agent_card,
    delegation_config,
)
from seamkit.tools import remote_skill_tool

RISK_URL = "http://risk.meridian.test/"
POLICE_REPORT_OVER = 10_000.0
# What the police records service holds, by claim.
POLICE_RECORDS = {"CLM-1002": "PR-5521"}

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
# Chapter 14: release 2027.02 states how each screening was made. While Risk's own
# breaker on Northgate's estimate_history is open, Risk screens without the history,
# and the answer says so in `basis`, never as a silent change in the score.
BASIS_FULL, BASIS_DEGRADED = "full", "without estimate history"
RISK_TERMS_2027_02 = replace(
    RISK_TERMS, revision="2027.02", effective=date(2027, 2, 1),
    skills={"screen_fraud": replace(SCREEN_FRAUD, outputs=(
        *SCREEN_FRAUD.outputs, Field("basis", "string")))})
# Chapter 18: release 2027.04 reports what every screening spent, signed, and honours
# the ceiling a caller sends. Release 2027.03 shipped the self-check under 2027.02's
# terms, which state no price and promise no report.
RISK_TERMS_2027_04 = replace(
    RISK_TERMS_2027_02, revision="2027.04", effective=date(2027, 4, 5),
    skills={"screen_fraud": replace(RISK_TERMS_2027_02.skills["screen_fraud"],
                                    usage="reported", honours_budget=True)})
# Chapter 19: release 2027.06 keeps a signed decision record for every screening and
# signs the head of Claims' receipt log daily. Risk's file names the evidence terms.
RISK_TERMS_2027_06 = revised(RISK_TERMS_2027_04, Path(__file__).parent / "contracts"
                             / "risk.2027.06.toml")
# The closer's hub shape: release 2027.08 reads the estimate history Claims sends as a
# fifth input, and accepts the estimate class to read it. Risk then calls nobody.
_HUB = revised(RISK_TERMS_2027_06, Path(__file__).parent / "contracts"
               / "risk.2027.08.toml")
RISK_TERMS_2027_08 = replace(
    _HUB, accepts=_HUB.accepts | {"estimate"},
    skills={"screen_fraud": replace(_HUB.skills["screen_fraud"], inputs=(
        *_HUB.skills["screen_fraud"].inputs, Field("estimate_history", "object[]")))})


def risk_card(release: str, terms: SeamTerms = RISK_TERMS,
              protocols: tuple[str, ...] = ("1.0",)) -> AgentCard:
    """Risk's card for one release, as Risk's pipeline builds and publishes it."""
    return agent_card("Risk fraud screening", RISK_URL, release,
                      {"screen_fraud": "Fraud score 0-100 for one claim"},
                      extensions=[publish(terms)], protocols=protocols)


# Every card Risk has released, with the 0-1 card Chapter 4's test publishes honestly.
# Risk's pipeline publishes each card's digest with the release; Claims pins them.
RELEASED: tuple[tuple[str, SeamTerms], ...] = (
    ("2026.09", RISK_TERMS), ("2026.10", RISK_TERMS), ("2026.10", RISK_TERMS_0_1),
    ("2026.11", RISK_TERMS), ("2026.12", RISK_TERMS), ("2027.02", RISK_TERMS_2027_02),
    ("2027.03", RISK_TERMS_2027_02), ("2027.04", RISK_TERMS_2027_04),
    ("2027.06", RISK_TERMS_2027_06), ("2027.08", RISK_TERMS_2027_08))
PUBLISHED_DIGESTS = frozenset(card_digest(risk_card(*release)) for release in RELEASED)


@dataclass(frozen=True)
class DecisionRecord:
    """Why Risk scored one claim as it did: digests and names, never its values."""

    chain_id: str  # the delegation chain the screening ran under
    task_id: str
    skill: str
    release: str  # Risk's release, and so its code
    behaviour: str  # how the score was made
    basis: str  # what the screening was made from, as the reply to Claims says
    score: float
    inputs: Mapping[str, str]  # field name -> digest under Risk's nonce
    signals: str  # one digest of the fraud signals, never the signals
    onward: tuple[str, ...]  # digests of Risk's own estimate_history receipts
    signer: str
    sig: str = ""

    def verifies(self, key: Ed25519PublicKey) -> bool:
        return genuine(self, key)


@dataclass
class EvidenceStore:
    """What Risk keeps for the evidence period, apart from any checkpoint: the signed
    record, and the opening of each field, which only Risk can produce."""

    records: dict[str, DecisionRecord] = field(default_factory=dict)  # by chain id
    openings: dict[str, dict[str, Opening]] = field(default_factory=dict)

    def keep(self, record: DecisionRecord, openings: dict[str, Opening]) -> None:
        self.records[record.chain_id] = record
        self.openings[record.chain_id] = openings

    def erase(self, chain: str) -> None:
        """A claimant's right to erasure: the values go, the signed digests stay."""
        self.openings[chain] = {}  # a tombstone: the record kept, nothing to open


class RiskScreenState(TypedDict, total=False):
    claim: dict[str, Any]
    police_report: str
    fraud_score: float
    fraud_signals: list[str]
    history_seen: list[dict[str, Any]]  # what the assistant's tool calls came back with


# A model's tool choices, given the claim and what earlier calls returned: the next
# call's arguments, or None to stop. Tests script it; no model key is needed.
Assistant = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any] | None]


# Risk's link to Northgate: estimate_history for the claim it is screening, no further.
NORTHGATE_GRANT = Grant(frozenset({"estimate_history"}), frozenset({"claim_id"}), 0)
MAX_TOOL_CALLS = 3
# Chapter 14: Risk's own breaker on the peer seam. It trips at three late or missing
# history answers in a minute, sooner than Claims' breaker on Risk (five), so the
# breaker nearest the cause opens first. While it is open, Risk screens without the
# history and says so.
HISTORY_BREAKER = BreakerPolicy(late_seconds=4.0, failures=3, window_seconds=60,
                                open_seconds=300, trial="estimate_history",
                                max_in_flight=4, mode="substitute")

# Risk's rules for what estimate_history may put in front of its assistant: all closed.
HISTORY_RULES: dict[str, IngressRule] = {
    "shop": IngressRule("shop", "value", pattern=r"SHOP-[0-9]+"),
    "estimates": IngressRule("estimates", "value", items={
        "claim_id": IngressRule("claim_id", "value", pattern=r"CLM-[0-9]{4}"),
        "part": IngressRule("part", "value", pattern=r"[a-z]+"),
        "amount": IngressRule("amount", "value")}),
}


# Chapter 18: Risk's screening model, scripted. The score is screen_fraud's on every
# pass; what a model would read and write is counted with the stand-in counter. The
# instructions' sizes are design values: the self-check's are twice the screening's.
SCREEN_PROMPT_TOKENS = 300
CHECK_PROMPT_TOKENS = 600


@dataclass
class ScreeningModel:
    """`self_check` deploys release 2027.03's second pass, which re-reads the claim,
    the first pass's reasons and the estimate history. `honours` deploys 2027.04,
    which prices the second pass before it runs and skips it over the ceiling."""

    self_check: bool = False
    honours: bool = False
    rates: RateCard = RATES
    meter: Meter | None = None  # the signed usage report, when Risk reports one
    calls: dict[str, int] = field(default_factory=dict)  # task -> model calls
    spent: dict[str, int] = field(default_factory=dict)  # task -> tokens, Risk's own

    def call(self, task: str, prompt: int, asked: Any, said: Any) -> int:
        count = prompt + tokens(asked) + tokens(said)  # input and output
        self.calls[task] = self.calls.get(task, 0) + 1
        self.spent[task] = self.spent.get(task, 0) + count
        if self.meter is not None:
            self.meter.spend(task, count)
        return self.spent[task]

    def screen(self, claim: dict[str, Any], history: list[dict[str, Any]],
               config: RunnableConfig) -> tuple[float, list[str]]:
        task = str(config["configurable"].get("task_id", ""))
        score, signals = screen_fraud(claim)
        first = {"score": score, "reasons": signals}
        spent = self.call(task, SCREEN_PROMPT_TOKENS, {"claim": claim}, first)
        if not self.self_check:
            return score, signals
        asked = {"claim": claim, "first_pass": first, "history": history}
        ceiling: Ceiling | None = config["configurable"].get("ceiling")
        if self.honours and ceiling is not None:  # 2027.04: priced before it runs
            would = spent + CHECK_PROMPT_TOKENS + tokens(asked) + tokens(first)
            if self.rates.price(would) > ceiling.per_task:
                if self.meter is not None:
                    self.meter.notes[task] = "self-check skipped: ceiling"
                return score, signals
        self.call(task, CHECK_PROMPT_TOKENS, asked, first)  # the same score again
        return score, signals

    def statement(self) -> dict[str, float]:
        """Risk's own record of each task's spend, priced: what the chargeback reads."""
        return {task: self.rates.price(n) for task, n in self.spent.items()}


def require_police_report(state: RiskScreenState) -> RiskScreenState:
    if float(state["claim"]["amount"]) <= POLICE_REPORT_OVER:
        return {}
    answer = interrupt({"needs": "police_report",
                        "why": "claims over 10,000 need a police report"})
    return {"police_report": str(answer["police_report"])}


def screen(state: RiskScreenState) -> RiskScreenState:
    score, signals = screen_fraud(state["claim"])
    return {"fraud_score": score, "fraud_signals": signals}


def build_risk_graph(checkpointer: InMemorySaver,
                     lookups: asyncio.Event | None = None,
                     consult: Any = None,
                     model: ScreeningModel | None = None,
                     keep: Any = None) -> Any:
    async def look_up_police_report(state: RiskScreenState) -> RiskScreenState:
        if float(state["claim"]["amount"]) <= POLICE_REPORT_OVER:
            return {}
        assert lookups is not None
        await lookups.wait()  # the records service answers when it answers
        return {"police_report": POLICE_RECORDS.get(state["claim"]["claim_id"], "")}

    def screen_by_model(state: RiskScreenState,
                        config: RunnableConfig) -> RiskScreenState:
        assert model is not None
        score, signals = model.screen(state["claim"], state.get("history_seen", []),
                                      config)
        return {"fraud_score": score, "fraud_signals": signals}

    graph = StateGraph(RiskScreenState)
    first = require_police_report if lookups is None else look_up_police_report
    graph.add_node(first.__name__, first)
    graph.add_node("screen", screen if model is None else screen_by_model)
    if consult is not None:  # the assistant looks before the police-report question
        graph.add_node("consult_northgate", consult)
        graph.add_edge(START, "consult_northgate")
        graph.add_edge("consult_northgate", first.__name__)
    else:
        graph.add_edge(START, first.__name__)
    graph.add_edge(first.__name__, "screen")
    if keep is not None:  # release 2027.06: the record is written before Risk answers
        graph.add_node("keep_record", keep)
        graph.add_edge("screen", "keep_record")
        graph.add_edge("keep_record", END)
    else:
        graph.add_edge("screen", END)
    return graph.compile(checkpointer=checkpointer)


@dataclass
class RiskPartner:
    """Risk's deployment: its graph, its checkpointer, its executor, its app.

    Passing `lookups` deploys the 2026.12 behaviour: no question for the caller, and a
    records lookup that finishes only when the event is set. Passing `assistant` deploys
    the fraud assistant and its Northgate tool; `peers` holds Northgate's address.
    """

    release: str = "2026.09"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)
    lookups: asyncio.Event | None = None
    assistant: Assistant | None = None
    forwards: bool = False  # the incident's tool client
    peers: dict[str, tuple[str, ASGIApp]] = field(default_factory=dict)
    terms: SeamTerms = RISK_TERMS
    rescaled: bool = False  # release 2026.10: the score leaves Risk on 0-1
    # Chapter 7: Meridian's revocation list, checked by the guard on every call and
    # before Risk mints onward; None is the list at Meridian's URL.
    revocations: Mapping[str, RevocationSource] | None = None
    now: Callable[[], float] = time.time
    filtered: bool = True  # Chapter 8: the history tool's answer passes the filter
    # Chapter 14: Risk's own breaker on its peer seam, and the transport its history
    # tool reaches Northgate over (None: the in-process wire, with no latency)
    breaker: BreakerPolicy | None = None
    wire: Callable[[ASGIApp], httpx.AsyncBaseTransport] | None = None
    # Chapter 15: the protocol versions Risk's card offers, one interface each
    protocols: tuple[str, ...] = ("1.0",)
    # Chapter 17: Risk signs its half of every seam receipt, a CI rule inside Meridian
    countersigns: bool = False
    # Chapter 18: the screening model behind screen_fraud (None: Chapter 1's rules
    # alone), and whether Risk signs a usage report into every task it answers
    model: ScreeningModel | None = None
    meters: bool = False
    # Chapter 19: the receipts Risk cuts on its own calls to Northgate (None: none),
    # the secret its decision records' nonces derive from, and where it keeps them
    onward: Receipts | None = None
    secret: bytes = field(default_factory=lambda: os.urandom(32))
    evidence: EvidenceStore = field(default_factory=EvidenceStore)

    def __post_init__(self) -> None:
        sources = (revocation_sources() if self.revocations is None
                   else self.revocations)
        self.check = checker(sources)
        consult = self._consult_northgate if self.assistant else None
        self.meter = (Meter(KEYS["risk"], "risk", RATES, self.now) if self.meters
                      else None)
        if self.model is not None:
            self.model.meter = self.meter
        keeps = self._keep if self.terms.evidence is not None else None
        self.graph = build_risk_graph(self.checkpointer, self.lookups, consult,
                                      self.model, keeps)
        self.executor = GraphExecutor(
            self.graph, "fraud_screening",
            to_input=lambda claim: {"claim": claim},
            to_artifact=self._artifact, configure=self._configure)
        self.history_breaker: SeamBreaker | None = None
        # Chapter 16: Risk's own crossing records on its peer seam, which only Risk
        # holds; Claims' records end at Risk.
        self.records: list[CrossingRecord] = []
        if self.breaker is not None:
            self.history_breaker = SeamBreaker("estimate_history", "Northgate",
                                               "estimate_history", self.breaker)
        self.card = risk_card(self.release, self.terms, self.protocols)
        signs = Countersigner(KEYS["risk"], "risk", self.now) if self.countersigns else None
        inner = a2a_app(self.card, self.executor, countersign=signs, meter=self.meter)
        self.guard = Guard(inner, "risk", MERIDIAN_TRUSTS, now=self.now,
                           revoked=self.check)
        self.app: ASGIApp = inner if self.forwards else self.guard
        self._tool: StructuredTool | None = None

    def _artifact(self, out: dict[str, Any]) -> dict[str, Any]:
        answer: dict[str, Any] = {
            "fraud_score": out["fraud_score"] / (100.0 if self.rescaled else 1.0)}
        outputs = {f.name for f in self.terms.skills["screen_fraud"].outputs}
        if "basis" in outputs:  # 2027.02: a screening says what it was made from
            answer["basis"] = self._basis(out)
        return answer

    @staticmethod
    def _basis(out: Mapping[str, Any]) -> str:
        missed = any("error" in seen for seen in out.get("history_seen", []))
        return BASIS_DEGRADED if missed else BASIS_FULL

    def _keep(self, state: RiskScreenState, config: RunnableConfig) -> RiskScreenState:
        """Release 2027.06: sign why this screening came out as it did, and keep it."""
        run = config["configurable"]
        task, claim = str(run["task_id"]), state["claim"]
        nonce = nonce_for(self.secret, task)  # derived, so it is never stored apart
        chain = chain_id(run["delegation"]) if "delegation" in run else ""
        onward = tuple(r.digest for r in self.onward.log.receipts()
                       if r.thread == run["thread_id"]) if self.onward else ()
        unsigned = DecisionRecord(
            chain, task, "screen_fraud", self.release,
            "self-check" if self.model and self.model.self_check else "single pass",
            self._basis(state), state["fraud_score"], shut(claim, nonce),
            digest({"signals": state["fraud_signals"]}, nonce), onward, "risk")
        self.evidence.keep(replace(unsigned, sig=signature(unsigned, KEYS["risk"])),
                           {k: Opening(k, v, nonce) for k, v in claim.items()})
        return {}

    def witness(self, seam: str, head: str, count: int) -> Witness | None:
        """Sign the head of Claims' receipt log, if Risk's terms promise to."""
        promised = self.terms.evidence
        if promised is None or promised.witness != "daily":
            return None
        return witness_head(KEYS["risk"], "risk", seam, head, count, self.now())

    def prune(self, context_id: str) -> None:
        """Risk prunes a closed task's thread 30 days after it closed; the evidence
        store is not the thread."""
        self.checkpointer.delete_thread(context_id)

    def _configure(self, call: ServerCallContext) -> dict[str, Any]:
        if self.forwards:  # what the incident's client reads and sends on
            headers: dict[str, str] = call.state.get("headers", {})
            return {"authorization": headers.get("authorization", "")}
        return delegation_config(call)

    async def history_tool(self) -> StructuredTool:
        """Northgate's estimate_history as a tool, on a client that mints or forwards."""
        if self._tool is None:
            url, app = self.peers["northgate"]
            peer = as_peer(app, "risk")
            transport = (self.wire(peer) if self.wire
                         else httpx.ASGITransport(app=peer))
            http = httpx.AsyncClient(transport=transport, base_url=url)
            from meridian.trust import MERIDIAN_TRUST  # the policy reads Risk's digests

            northgate = await connect(url, http, trust=MERIDIAN_TRUST["Northgate"])
            assert northgate.trust is not None
            # The onward act point (Chapter 7): the run's chain is re-checked first.
            http.auth = ForwardingAuth() if self.forwards else DelegationAuth(
                "risk", KEYS["risk"], "northgate", NORTHGATE_GRANT,
                checked(inbound_chain, self.check, self.now),
                northgate.card, northgate.trust.covers)
            # The argument schema comes from Northgate's published input terms.
            self._tool = remote_skill_tool(northgate, "estimate_history", "Northgate",
                                           log=self.records.append,
                                           rules=HISTORY_RULES,
                                           filtered=self.filtered,
                                           breaker=self.history_breaker,
                                           receipts=self.onward,
                                           authority=lambda: chain_id(inbound_chain()))
        return self._tool

    async def _consult_northgate(self, state: RiskScreenState) -> RiskScreenState:
        assert self.assistant is not None
        tool = await self.history_tool()
        seen: list[dict[str, Any]] = []
        while len(seen) < MAX_TOOL_CALLS and (
                args := self.assistant(state["claim"], seen)) is not None:
            try:
                seen.append(await tool.ainvoke(args))
            except SeamError as exc:  # a tool error goes back to the model, as text
                seen.append({"error": type(exc).__name__, "detail": exc.detail})
        return {"history_seen": seen}

    async def finish_lookups(self) -> None:
        """Let every pending records lookup answer, and wait for Risk's tasks to end."""
        assert self.lookups is not None
        self.lookups.set()
        for _ in range(500):
            if len(self.executor.finished) == len(self.executor.received):
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("Risk's tasks did not finish")
