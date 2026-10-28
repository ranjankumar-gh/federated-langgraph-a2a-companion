"""Finance's payments over A2A: a one-node LangGraph graph behind GraphExecutor.

Finance publishes its seam terms on its card (Chapter 4). Since Chapter 8, release
2026.11's terms accept an optional remittance memo, which Finance prints on the payment.
Claims cannot see who or what reads it next at Finance.

Release 2026.12 (Chapter 10) adds two-party approval: a payment over 2,500 needs one
approver at Claims and one at Finance, two different people. The graph asks each with
`ask`, which names who must answer; the executor keeps Finance's own question inside,
and Finance's inbox answers it. `addressed=False` deploys 2026.12 as it first shipped,
with two bare interrupt() calls behind Chapter 3's executor: the opening incident.

Release 2026.14 (Chapter 12) states a key window on pay_claim: for 24 hours, a send that
repeats a key gets the task the first send opened. The key index is kept across releases
with the task store, and a restore from backup takes both back; the `paid` log, which
stands for money that left Finance's bank, never goes back.

Release 2026.15 (Chapter 13) declares `void_payment` as the compensating skill of
pay_claim. A completed payment waits for Finance's payment run, which sends only
payments that have waited out the 12-hour void window; until the run sends it,
void_payment voids it. After that the money has left, and the answer is a refusal: what
remains is a recovery from the payee, which is for people. `void_payment` has no graph
behind it, so the executor answers it directly.
"""

import copy

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, TypedDict, cast

import httpx

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from starlette.types import ASGIApp

from meridian.agents import pay
from meridian.identity import (
    ISSUER,
    KEYS,
    MERIDIAN,
    MERIDIAN_TRUSTS,
    revocation_sources,
)
from seamkit.approval import decode_approval, refuse_approval
from seamkit.cards import card_digest
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.delegation import Guard, Verified, chain_id
from seamkit.errors import SeamError
from seamkit.ledger import question_digest
from seamkit.revocation import RevocationSource, checker, still_holds
from seamkit.server import (
    Countersigner,
    GraphExecutor,
    Keys,
    Waiting,
    a2a_app,
    agent_card,
    ask,
    push_sender,
    skill_of,
)

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
# Chapter 10: two-party approval over 2,500. A task may wait 48 hours on Finance's
# approver; after that Finance reassigns it to an alternate approver.
FINANCE_TERMS_2026_12 = replace(
    FINANCE_TERMS_2026_11, revision="2026.12", effective=date(2026, 12, 7),
    skills={"pay_claim": replace(FINANCE_TERMS_2026_11.skills["pay_claim"],
                                 asks=("approval",), wait_seconds=48 * 3600)})
# Two later releases, shipped while a task waits: the first changes nothing about
# pay_claim; the second shortens the wait horizon, a change to the skill's terms.
FINANCE_TERMS_2026_12_1 = replace(FINANCE_TERMS_2026_12, revision="2026.12.1")
FINANCE_TERMS_2026_13 = replace(
    FINANCE_TERMS_2026_12, revision="2026.13", effective=date(2026, 12, 14),
    skills={"pay_claim": replace(FINANCE_TERMS_2026_12.skills["pay_claim"],
                                 wait_seconds=24 * 3600)})
# Chapter 12: a repeat of a key inside 24 hours gets the task the first send opened.
FINANCE_TERMS_2026_14 = replace(
    FINANCE_TERMS_2026_13, revision="2026.14", effective=date(2027, 1, 11),
    skills={"pay_claim": replace(FINANCE_TERMS_2026_13.skills["pay_claim"],
                                 key_seconds=24 * 3600)})
# Chapter 13: a payment can be voided until the payment run sends it, 12 hours on.
VOID_PAYMENT = SkillTerms(
    inputs=(Field("claim_id", "string"), Field("task", "string")),
    outputs=(Field("outcome", "string"), Field("owed", "number", unit="claim currency"),
             Field("residue", "string")),
    p95_seconds=2.0, key_seconds=24 * 3600)
FINANCE_TERMS_2026_15 = replace(
    FINANCE_TERMS_2026_14, revision="2026.15", effective=date(2027, 2, 1),
    skills={"pay_claim": replace(FINANCE_TERMS_2026_14.skills["pay_claim"],
                                 compensated_by="void_payment",
                                 compensate_seconds=12 * 3600),
            "void_payment": VOID_PAYMENT})
TERMS = {"2026.10": FINANCE_TERMS, "2026.11": FINANCE_TERMS_2026_11,
         "2026.12": FINANCE_TERMS_2026_12, "2026.12.1": FINANCE_TERMS_2026_12_1,
         "2026.13": FINANCE_TERMS_2026_13, "2026.14": FINANCE_TERMS_2026_14,
         "2026.15": FINANCE_TERMS_2026_15}
TWO_PARTY = frozenset({"2026.12", "2026.12.1", "2026.13", "2026.14", "2026.15"})
TWO_PARTY_OVER = 2_500.0


@dataclass(frozen=True)
class Backup:
    tasks: dict[str, Any]
    keys: dict[Any, Any]
    checkpoints: Any
    waiting: dict[str, Waiting]


class ApprovalRefused(SeamError):
    """An answer to Finance's approval question is not an approval Finance can verify."""


class PaymentState(TypedDict, total=False):
    claim: dict[str, Any]
    payment_ref: str
    approvers: list[str]  # Chapter 10: who approved, as each owner's identity says
    task_id: str  # Chapter 13: the task that made the payment, which a void names


def make_payment(state: PaymentState, config: RunnableConfig) -> PaymentState:
    return {"payment_ref": pay(state["claim"]),
            "task_id": str(config["configurable"].get("task_id", ""))}


class PaymentsExecutor(GraphExecutor):
    """pay_claim runs the payment graph; void_payment, a plain skill, is answered here
    with no graph behind it."""

    def __init__(self, *args: Any,
                 voids: Callable[[Mapping[str, Any]], dict[str, Any] | str],
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.voids = voids

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        message = context.message
        if not (message and skill_of(message) == "void_payment"):
            await super().execute(context, queue)
            return
        assert context.task_id and context.context_id
        await queue.enqueue_event(new_task_from_user_message(message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        answer = self.voids(get_data_parts(message.parts)[0])
        if isinstance(answer, str):  # refused: the reason is the answer
            await updater.reject(updater.new_agent_message(
                [new_data_part({"why": answer})]))
            return
        await updater.add_artifact([new_data_part(answer)], name="void_payment")
        await updater.complete()


def finance_card(release: str) -> AgentCard:
    """Finance's card for one release, as Finance's pipeline builds and publishes it."""
    skills = {"pay_claim": "Pay a claim against its estimate"}
    if "void_payment" in TERMS[release].skills:  # Chapter 13
        skills["void_payment"] = "Void a payment before the payment run sends it"
    return agent_card("Finance payments", FINANCE_URL, release, skills,
                      extensions=[publish(TERMS[release])], push=True)


# Finance's pipeline publishes each release's card digest with the release.
PUBLISHED_DIGESTS = frozenset(card_digest(finance_card(r)) for r in TERMS)


@dataclass
class FinancePartner:
    release: str = "2026.11"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)
    # Chapter 7: Meridian's revocation list, checked by the guard on every call.
    revocations: Mapping[str, RevocationSource] | None = None
    now: Callable[[], float] = time.time
    # Chapter 10: False deploys 2026.12 as it first shipped, with bare interrupts
    addressed: bool = True
    # Chapter 11: the client the push sender posts with; None rings nobody
    push: httpx.AsyncClient | None = None
    # Chapter 17: Finance signs its half of every seam receipt, a CI rule inside Meridian
    countersigns: bool = False

    def __post_init__(self) -> None:
        sources = (revocation_sources() if self.revocations is None
                   else self.revocations)
        self.sources = sources
        self.countersigner = Countersigner(KEYS["finance"], "finance", self.now) if (
            self.countersigns) else None  # one across releases, like the task store
        graph = StateGraph(PaymentState)
        graph.add_node("make_payment", make_payment)
        if self.release in TWO_PARTY:  # Chapter 10: two approvals before the payment
            graph.add_node("claims_approval", self._claims_approval)
            graph.add_node("finance_approval", self._finance_approval)
            graph.add_edge(START, "claims_approval")
            graph.add_edge("claims_approval", "finance_approval")
            graph.add_edge("finance_approval", "make_payment")
        else:
            graph.add_edge(START, "make_payment")
        graph.add_edge("make_payment", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)
        self.memos: dict[str, str] = {}  # claim -> the memo printed on its payment
        self.paid: list[dict[str, Any]] = []  # every payment, with who approved it
        self.executor = PaymentsExecutor(
            self.graph, "payments",
            to_input=self._received,
            to_artifact=self._paid,
            own="finance" if self.addressed else "", now=self.now, voids=self._void)
        # One push config store across releases, like the task store; the inbox rings
        # through the same sender, because a write from outside a request pushes nothing
        self.pushing = push_sender(self.push) if self.push else None
        self.executor.push = self.pushing[1] if self.pushing else None
        # The guard's task bindings outlive any one release's app.
        self.bound: dict[str, tuple[str, str]] = {}
        self.holders: dict[str, str] = {}
        self.keys: Keys = {}  # Chapter 12: the key index, kept like the task store
        self.ship(self.release)

    def ship(self, release: str) -> None:
        """Deploy a release's card. Open tasks keep running on the same graph and store,
        and the guard keeps the tasks it has bound to a person and a claim."""
        self.release, self.card = release, finance_card(release)
        inner = a2a_app(self.card, self.executor, store=self.executor.store,
                        push=self.pushing, keys=self.keys, now=self.now,
                        countersign=self.countersigner)
        self.guard = Guard(inner, "finance", MERIDIAN_TRUSTS, now=self.now,
                           revoked=checker(self.sources), tasks=self.bound,
                           holders=self.holders)
        self.app: ASGIApp = self.guard

    def backup(self) -> "Backup":
        """Last night's backup: the task store, the key index, and the graph's state."""
        saver = self.checkpointer
        return Backup(copy.deepcopy(self._held_tasks()), dict(self.keys),
                      copy.deepcopy((saver.storage, saver.writes, saver.blobs)),
                      dict(self.executor.waiting))

    def restore(self, backup: "Backup") -> None:
        """Restore from backup. The records go back; payments already made do not."""
        held = self._held_tasks()
        held.clear()
        held.update(copy.deepcopy(backup.tasks))
        self.keys.clear()
        self.keys.update(backup.keys)
        saver = self.checkpointer
        stores: tuple[dict[Any, Any], ...] = (saver.storage, saver.writes, saver.blobs)
        for live, kept in zip(stores, copy.deepcopy(backup.checkpoints), strict=True):
            live.clear()
            live.update(kept)
        self.executor.waiting = dict(backup.waiting)

    def _held_tasks(self) -> dict[str, Any]:
        """The in-memory store's own dict: what a database restore would put back."""
        store = cast(InMemoryTaskStore, self.executor.store)
        tasks = cast(dict[str, Any], store._impl.tasks)  # the SDK keeps no handle
        return tasks

    def _paid(self, out: dict[str, Any]) -> dict[str, Any]:
        self.paid.append({"claim_id": out["claim"]["claim_id"],
                          "payment_ref": out["payment_ref"],
                          "approvers": out.get("approvers", []),
                          # Chapter 13: which task paid it, when, and where it is now
                          "task_id": out.get("task_id", ""), "at": self.now(),
                          "sent": False, "voided": False})
        return {"payment_ref": out["payment_ref"]}

    def _void(self, request: Mapping[str, Any]) -> dict[str, Any] | str:
        """void_payment: honoured until the payment run sends the payment."""
        if "void_payment" not in TERMS[self.release].skills:
            return f"void_payment is not offered on {self.release}"
        payment = next((p for p in self.paid if p["task_id"] == request["task"]
                        and p["claim_id"] == request["claim_id"]), None)
        if payment is None:
            return f"no payment was made by task {request['task']}"
        if payment["sent"]:  # the money has left: a recovery is for people
            return f"{payment['payment_ref']} left in the payment run"
        payment["voided"] = True
        return {"outcome": "voided", "owed": 0.0, "residue": ""}

    def payment_run(self) -> list[str]:
        """Finance's payment run: every payment past its void window leaves the bank."""
        held = TERMS[self.release].skills["pay_claim"].compensate_seconds or 0
        due = [p for p in self.paid if not (p["sent"] or p["voided"])
               and self.now() - p["at"] >= held]
        for payment in due:
            payment["sent"] = True
        return [p["payment_ref"] for p in due]

    def _question(self, state: PaymentState, stage: str) -> dict[str, Any]:
        claim = state["claim"]
        return {"needs": "approval", "stage": stage, "claim_id": claim["claim_id"],
                "amount": claim["amount"]}

    def _claims_approval(self, state: PaymentState,
                         config: RunnableConfig) -> PaymentState:
        if state["claim"]["amount"] <= TWO_PARTY_OVER:
            return {}
        question = self._question(state, "first approver")
        if not self.addressed:  # as first shipped: whoever resumes it, approves
            interrupt(question)
            return {"approvers": ["claims (unverified)"]}
        answer = ask(question, "claims", "claims_manager")
        task = str(config["configurable"]["task_id"])
        why = refuse_approval(answer, {ISSUER: MERIDIAN.keys[ISSUER]},
                              "claims/claims_manager", task, question["claim_id"],
                              question["amount"], question_digest(
                                  {**question, "to": "claims/claims_manager"}))
        if why:  # the answer's body never says who answered; the signature does
            raise ApprovalRefused("payments", why, task)
        approval = decode_approval(str(cast(dict[str, Any], answer)["approval"]))
        assert approval is not None  # refuse_approval decoded it already
        return {"approvers": [approval.approver]}

    def _finance_approval(self, state: PaymentState) -> PaymentState:
        if state["claim"]["amount"] <= TWO_PARTY_OVER:
            return {}
        question = self._question(state, "second approver")
        if not self.addressed:
            interrupt(question)
            return {"approvers": [*state.get("approvers", []), "finance (unverified)"]}
        answer = ask(question, "finance", "payments_approver")
        approver, approvers = str(answer["approver"]), state.get("approvers", [])
        if approver in approvers:  # two-party means two people
            raise ApprovalRefused("payments", f"{approver} has already approved")
        return {"approvers": [*approvers, approver]}

    async def approve(self, task_id: str, approver: str) -> str:
        """Finance's own inbox: one of Finance's approvers answers. Never over A2A."""
        return await self.executor.answer(task_id, {"approver": approver},
                                          holds=self._holds)

    def _holds(self, config: RunnableConfig) -> str:
        """The commit act point: does the chain the task last held still hold?"""
        chain = config["configurable"].get("delegation")
        if not chain:
            return "ok"
        verified = Verified(chain, chain_id(chain), chain[-1].subject, "meridian")
        return still_holds(verified, self.sources, self.now())

    def _received(self, claim: dict[str, Any]) -> PaymentState:
        if "payment_memo" in claim:
            self.memos[str(claim["claim_id"])] = str(claim["payment_memo"])
        return {"claim": claim}
