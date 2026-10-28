"""RemoteAgentNode: one LangGraph node that sends one partner skill one A2A task.

It is the node call shape for a seam. It sends only the fields the seam names, keeps a
task ledger entry for the task in the caller's own checkpoint (written before the send
and after every answer), turns every task state into a state update or a typed
SeamError, writes the partner's artifact into state only through the seam's field map,
and emits one crossing record per call.

Since Chapter 4, `connect` checks the partner's published terms against the seam's
expectations, a breached seam refuses to send, and a completed artifact is checked
against the shapes the seam depends on before the field map writes it.

Since Chapter 5, `connect` runs the card trust policy first: a card the policy does not
trust has no terms worth checking. The send step refuses an untrusted card, an
unapproved skill, or a suspended one, and every completed answer is observed in the
skill's track record under the card's digest (the trust gradient).

Since Chapter 8, every completed answer passes the seam ingress filter after the shape
check and the field map (seamkit.ingress): a closed value is admitted, free text is
cleaned, capped and admitted as text, and the tool shape hands a model values only. A
partner's question and a refusal reason are cleaned and capped before a person or a
log sees them.

Since Chapter 12, a send whose answer was lost is sent again under the same key, but
only while the partner's terms honour that key and the seam's retry policy allows it.
Otherwise the node looks the key up in the partner's task list, and adopts the task it
finds; it never sends under a key nobody honours. A seam whose partner honours no key
can look its key up before every first send (`lookup`), which closes the crash window
between a send and the checkpoint that records it.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, cast

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig
from a2a.client import A2AClientError, ClientFactory
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import AgentCard, AgentInterface, CancelTaskRequest, GetTaskRequest
from a2a.types import ListTasksRequest
from a2a.types import Message, Role
from a2a.types import SendMessageConfiguration
from a2a.types import SendMessageRequest, Task, TaskPushNotificationConfig, TaskState
from a2a.utils.constants import VERSION_HEADER
from a2a.utils.errors import A2AError
from langgraph.types import interrupt

import seamkit.ledger
from seamkit.cards import card_digest
from seamkit.contract import (
    NO_CROSSING,
    Breach,
    CrossingPolicy,
    Expectations,
    Field,
    SeamTerms,
    held_to,
    terms_of,
    violations,
)
from seamkit.breaker import SeamBreaker
from seamkit.costs import BUDGET_URI, BudgetSpent, CostEntry, Costs, purpose_of
from seamkit.errors import (
    BehaviourDrift,
    BreakerOpen,
    CardUntrusted,
    ContractBreach,
    DelegationRefused,
    MalformedAnswer,
    PartnerFailed,
    PartnerNeedsAuth,
    PartnerNeedsInput,
    PartnerRejected,
    PartnerUnavailable,
    SeamError,
    TaskUnfinished,
    UntrustedEgress,
)
from seamkit.ingress import (
    DETAIL_MAX_CHARS,
    QUESTION_MAX_CHARS,
    IngressRule,
    Provenance,
    admit,
    egress,
    log_findings,
    quote,
    tag,
    withhold,
)
from seamkit.ledger import (
    LedgerEntry,
    intended,
    moved,
    observe,
    question_digest,
    state_name,
    unanswered,
)
from seamkit.receipt import (
    NONCE,
    ReceiptHalf,
    ReceiptMismatch,
    Receipts,
    SeamReceipt,
    answered,
    close,
    digest,
    thread_of,
)
from seamkit.retry import NO_RETRY, RetryPolicy, refuse_retry, retrying
from seamkit.tracing import current_trace_id, inject, seam_span, trace_id, traceparent
from seamkit.trust import (
    Gradient,
    PartnerTrust,
    TrackRecord,
    TrustDecision,
    decide,
    log_decision,
)

# Every state the A2A specification defines, and the one graph outcome each becomes.
# None means "read the artifact"; only a completed task carries an answer.
OUTCOMES: dict[int, type[SeamError] | None] = {
    TaskState.TASK_STATE_COMPLETED: None,
    TaskState.TASK_STATE_SUBMITTED: TaskUnfinished,
    TaskState.TASK_STATE_WORKING: TaskUnfinished,
    TaskState.TASK_STATE_INPUT_REQUIRED: PartnerNeedsInput,
    TaskState.TASK_STATE_AUTH_REQUIRED: PartnerNeedsAuth,
    TaskState.TASK_STATE_FAILED: PartnerFailed,
    TaskState.TASK_STATE_CANCELED: PartnerFailed,
    TaskState.TASK_STATE_REJECTED: PartnerRejected,
    TaskState.TASK_STATE_UNSPECIFIED: MalformedAnswer,
}

COMPLETED = TaskState.TASK_STATE_COMPLETED
FINAL = frozenset({COMPLETED, TaskState.TASK_STATE_FAILED,
                   TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED})
INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED


@dataclass(frozen=True)
class SeamSpec:
    """What one seam sends, what it keeps, and who answers a partner's question."""

    name: str  # the node name, which is also the seam's name in the inventory
    owner: str  # the partner that owns the other side
    skill: str  # the skill id on the partner's card
    sends: tuple[str, ...]  # state fields the partner receives; nothing else crosses
    receives: Mapping[str, str]  # artifact key -> state field
    asks_caller: bool = False  # input-required goes to the caller's own interrupt()
    timeout: float = 10.0  # seconds before a send counts as PartnerUnavailable
    return_immediately: bool = False  # the send returns once the task exists
    # artifact key -> the shape the seam depends on, from its expectations (Chapter 4)
    shapes: Mapping[str, Field] = field(default_factory=dict)
    # state field -> its ingress rule (Chapter 8); None is a seam with no filter yet
    rules: Mapping[str, IngressRule] | None = None
    # Chapter 10: the Claims roles a partner's question may be addressed to. Empty
    # keeps Chapter 3's rule: whoever runs the thread answers what the seam answers.
    roles: tuple[str, ...] = ()
    wait_seconds: float | None = None  # how long Claims waits on a partner's people
    # Chapter 12: how calls on this seam are retried, and whether a first send looks
    # its key up first (a partner that honours no key)
    retry: RetryPolicy = NO_RETRY
    lookup: bool = False


@dataclass(frozen=True)
class PartnerConnection:
    """A partner's card as fetched, an A2A client built from it, and the contract."""

    card: AgentCard
    client: Client
    terms: tuple[SeamTerms, ...] = ()  # the revisions held to, the one in force first
    breaches: tuple[Breach, ...] = ()  # in force now: the seam refuses to send
    upcoming: tuple[Breach, ...] = ()  # in announced revisions, each with its date
    trust: TrustDecision | None = None  # the card trust policy's verdict (Chapter 5)
    pinned: PartnerTrust | None = None  # what the verdict was decided against
    # (skill, card digest, behaviour version) -> what Claims observed; shared across
    # refreshes. Chapter 15 added the behaviour version to Chapter 5's key.
    records: dict[tuple[str, str, str], TrackRecord] = field(default_factory=dict)
    # Chapter 15: the protocol version of the interface Claims chose, the behaviour
    # version each skill's answers are recorded under, and the skills a drift holds
    protocol: str = ""
    behaviour: dict[str, str] = field(default_factory=dict)
    held: dict[str, str] = field(default_factory=dict)

    @property
    def release(self) -> str:
        return self.card.version

    def record(self, skill: str) -> TrackRecord:
        """What Claims observed of `skill` under this card and this behaviour."""
        digest = self.trust.digest if self.trust else ""
        key = (skill, digest, self.behaviour.get(skill, ""))
        return self.records.setdefault(key, TrackRecord())

    def rekey(self, skill: str, behaviour: str, keep: bool = False) -> None:
        """Record `skill` under a new behaviour version, from nothing unless `keep`."""
        earned = self.record(skill)
        self.behaviour[skill] = behaviour
        if keep:
            digest = self.trust.digest if self.trust else ""
            self.records[(skill, digest, behaviour)] = earned

    def gradient(self, skill: str) -> Gradient:
        gradients = self.pinned.gradients if self.pinned else {}
        return gradients.get(skill, Gradient())

    def refuses(self, skill: str) -> str:
        """Why the card trust policy will not let `skill` be called now, or ""."""
        if self.trust is None:  # no policy given: the card as fetched (Chapter 3)
            return ""
        if not self.trust.trusted:
            return self.trust.refused
        if skill not in self.trust.skills:  # capability inflation, or never offered
            return f"{skill} is not approved for {self.trust.owner}"
        if self.gradient(skill).tier(self.record(skill)) == "suspended":
            return f"{skill} is suspended under this card after a malformed answer"
        return ""

    def over_ceiling(self, skill: str, answer: Mapping[str, Any]) -> str:
        """Why this answer decides more than the skill has earned, or ""."""
        gradient = self.gradient(skill)
        limit = gradient.ceiling(self.record(skill))
        value = answer.get(gradient.output)
        if limit is None or not isinstance(value, (int, float)) or value <= limit:
            return ""
        return (f"{gradient.output} {value:,.0f} is over the {limit:,.0f} an unproven "
                f"{skill} may decide")


contract_logger = logging.getLogger("seamkit.contract")


def terms_digest(connection: "PartnerConnection", skill: str) -> str:
    """The skill's terms in force, hashed: what a resume compares if the card changes."""
    held = connection.terms[0].skills.get(skill) if connection.terms else None
    raw = json.dumps(asdict(held) if held else {}, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def choose(card: AgentCard, speaks: Sequence[str]) -> AgentInterface | None:
    """The card's interface on the newest protocol version this seam speaks, or None."""
    for version in speaks:  # newest first: the seam's file decides, not the SDK
        for interface in card.supported_interfaces:
            if interface.protocol_version == version:
                return interface
    return None


def speaking(card: AgentCard, interface: AgentInterface,
             http: httpx.AsyncClient) -> AgentCard:
    """The card as Claims' client sees it: the one interface Claims chose. Every
    request on `http` then carries that interface's protocol version in its header."""
    chosen = AgentCard()
    chosen.CopyFrom(card)
    del chosen.supported_interfaces[:]
    chosen.supported_interfaces.append(interface)
    http.headers[VERSION_HEADER] = interface.protocol_version
    return chosen


async def connect(url: str, http: httpx.AsyncClient,
                  expects: Expectations | None = None,
                  policy: CrossingPolicy = NO_CROSSING,
                  terms: Sequence[SeamTerms] = (),
                  trust: PartnerTrust | None = None,
                  records: dict[tuple[str, str, str], TrackRecord] | None = None,
                  seen: Mapping[str, date] | None = None,
                  ) -> PartnerConnection:
    """Fetch the partner's card, decide whether to trust it, and check the contract.

    With no expectations there is no seam to check (a lookup, a tool's partner).
    `terms` stands in for a partner that publishes none: the terms agreed on paper.
    `trust` is the owner's entry in the caller's card trust policy; without one the
    card is used as fetched, which is Chapter 3's behaviour and nobody's production.
    Since Chapter 15 the client speaks the protocol version the seam's expectations
    choose, and `seen` (when Claims first saw each revision) feeds the notice rule.
    """
    card = await A2ACardResolver(http, url).get_agent_card()
    speaks = expects.protocols if expects else ()
    interface = choose(card, speaks) if speaks else None
    client_card, protocol = card, ""
    if interface is not None:
        client_card = speaking(card, interface, http)
        protocol = interface.protocol_version
    factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
    kept = {} if records is None else records
    decision = None
    if trust is not None:  # decided on the card as fetched, every interface on it
        decision = decide(card, url, trust, (expects.skill,) if expects else ())
        log_decision(decision)
        if not decision.trusted:  # a card that fails the policy has no terms to check
            return PartnerConnection(card, factory.create(client_card), trust=decision,
                                     pinned=trust, records=kept, protocol=protocol)
    revisions = tuple(terms) or terms_of(card)
    if expects is None:
        return PartnerConnection(card, factory.create(client_card), revisions,
                                 trust=decision, pinned=trust, records=kept,
                                 protocol=protocol)
    now, announced = held_to(revisions, expects, policy, seen)
    if speaks and interface is None:  # a breach in force: nothing is sent on the seam
        now.append(unspoken(card, expects, revisions))
    for breach in announced:  # found on the day it is announced, not the day it lands
        contract_logger.warning("contract breach announced: %s", breach)
    return PartnerConnection(card, factory.create(client_card), revisions, tuple(now),
                             tuple(announced), decision, trust, kept, protocol)


def unspoken(card: AgentCard, expects: Expectations,
             revisions: Sequence[SeamTerms]) -> Breach:
    """The breach of a card with no interface on a protocol version the seam speaks."""
    offered = sorted(i.protocol_version or "unstated"
                     for i in card.supported_interfaces)
    current = revisions[0] if revisions else None
    return Breach(expects.seam, current.revision if current else "none",
                  current.effective if current else date.min,
                  f"{expects.owner}'s card offers protocol {', '.join(offered)}, and "
                  f"this seam speaks {', '.join(expects.protocols)}")


async def refresh(connection: PartnerConnection, url: str, http: httpx.AsyncClient,
                  expects: Expectations | None = None,
                  policy: CrossingPolicy = NO_CROSSING) -> PartnerConnection:
    """Refetch the card. The same digest keeps the connection; a new one is a new card.

    A new card is decided and checked again, and its skills start unproven. If it is
    not trusted, the seam fails closed: the old card is not kept, because the partner
    no longer serves it.
    """
    fresh = await connect(url, http, expects, policy, trust=connection.pinned,
                          records=connection.records)
    same = connection.trust and fresh.trust and (
        fresh.trust.digest == connection.trust.digest)
    if same:
        return connection
    fresh.behaviour.update(connection.behaviour)  # a new card keeps what canaries found
    fresh.held.update(connection.held)
    return fresh


@dataclass(frozen=True)
class CrossingRecord:
    """One structured record per seam crossing: the observability pattern from Ch 1."""

    seam: str
    partner: str
    partner_release: str
    call: str  # "send", "get", "list", "cancel", or "push" (Chapter 17)
    sent: tuple[str, ...]  # field names, never values
    context_id: str = ""
    task_id: str = ""
    state: str = ""
    error: str = ""
    idempotency_key: str = ""  # joins a crossing to its ledger entry
    behaviour: str = ""  # Chapter 15: the behaviour version the skill is recorded under
    # Chapter 17: when Claims recorded it, on the clock every owner reads; how long a
    # send waited for its answer; and the trace the crossing's span belongs to
    at: float = 0.0
    seconds: float | None = None
    trace_id: str = ""


crossing_logger = logging.getLogger("seamkit.crossing")


def log_crossing(record: CrossingRecord) -> None:
    crossing_logger.info("seam crossing", extra={"crossing": asdict(record)})


def refused(exc: A2AClientError) -> str:
    """The partner guard's reason if this is its 403 refusal; "" for an outage."""
    cause = exc.__cause__
    if not (isinstance(cause, httpx.HTTPStatusError)
            and cause.response.status_code == 403):
        return ""
    try:  # partner text on its way to a log and an error a model may read: capped
        return str(quote(str(cause.response.json()["error"]["message"]),
                         DETAIL_MAX_CHARS))
    except (ValueError, KeyError, TypeError):
        return "refused"


def question_of(task: Task) -> dict[str, Any]:
    """The partner's question, as the data part of its input-required status message.

    It goes to a person through interrupt(), so since Chapter 8 it arrives cleaned,
    capped, and marked as partner text: whoever shows it quotes it and follows no link.
    """
    parts = get_data_parts(task.status.message.parts)
    asked = quote(parts[0], QUESTION_MAX_CHARS) if parts else {}
    # Chapter 10: the digest an approval of this question is bound to.
    digest = question_digest(parts[0]) if parts else ""
    return {"task_id": task.id, **asked, "partner_text": sorted(asked),
            "asked": digest}


def addressee(task: Task) -> str:
    """Who the partner says a paused task waits on, "owner/role"; "" if it names none."""
    message = task.status.message
    parts = get_data_parts(message.parts) if message.parts else []
    said = parts[0] if parts else {}
    return str(said.get("to") or said.get("waiting_on") or "")


def no_push(seam: str, key: str | None) -> TaskPushNotificationConfig | None:
    """A seam nobody rings for: the caller reads the task when it next runs."""
    return None


@dataclass
class RemoteAgentNode:
    """The node call shape: one partner skill, one task, an update or a SeamError."""

    seam: SeamSpec
    partner: PartnerConnection
    log: Callable[[CrossingRecord], None] = field(default=log_crossing)
    # Names the authority a task is sent under, never its value: the run's chain id.
    authority: Callable[[], str] = field(default=lambda: "")
    # Chapter 11: the push config a new task is sent with (seamkit.push.PushTo).
    push: Callable[[str, str | None], TaskPushNotificationConfig | None] = field(
        default=no_push)
    # Chapter 12: how a retry waits; a test replaces it with one that moves a clock
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep)
    # Chapter 14: the seam's breaker, asked before a send opens work and told after
    breaker: SeamBreaker | None = None
    # Chapter 17: what the node needs to cut a seam receipt; None keeps none
    receipts: Receipts | None = None
    # key -> the traceparent of the send that opened its task, until the ledger has it
    traces: dict[str, str] = field(default_factory=dict)
    # Chapter 18: the seam's budget, its ledger and its rate card; None prices nothing.
    # key -> what the task is for, decided before the send
    costs: Costs | None = None
    purposes: dict[str, str] = field(default_factory=dict)

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        name = self.seam.name
        entry = state.get("ledger", {}).get(name)
        if entry is None:  # the intent and its key are checkpointed before any send
            if self.breaker and (why := self.breaker.refuses(seamkit.ledger.CLOCK())):
                self.log(self._record("send", self.seam.sends)(error="BreakerOpen"))
                raise BreakerOpen(name, why)  # nothing recorded, nothing sent
            return {"ledger": {name: self.intend()}}
        sending = {k: state[k] for k in self.seam.sends}
        said: Mapping[str, Any] | None = sending  # what this entry sends, if anything
        if entry["state"] == "unknown":  # sent, answer lost: the same key, or a lookup
            task = await self.recover(sending, entry)
        elif not entry["task_id"]:
            try:  # one superstep later: send under the key the checkpoint holds
                looks = self.seam.lookup and not self.held_back()
                found = await self.lookup(entry) if looks else None
                task = found or await self._send(sending, entry,
                                                 key=entry["idempotency_key"])
            except PartnerUnavailable:  # record the attempt; re-entry recovers it
                return {"ledger": {name: unanswered(entry)}}
        else:  # a task exists: read it, never send a second one
            said = None
            task = await self._get(entry["task_id"], entry.get("traceparent", ""))
            if self._waits_for_us(task) and entry["state"] == "input-required":
                reply = said = interrupt(question_of(task))  # asked once recorded
                task = await self._send(reply, entry)  # same task, same context
            elif (self.breaker and entry["state"] != "failed"
                  and state_name(task.status.state) == "failed"):
                self.breaker.failed_later(seamkit.ledger.CLOCK())  # never sent: read
        seen = self._accounted(entry, observe(entry, task, self.partner.release), task,
                               said)
        if task.status.state == COMPLETED:
            answer = self._outcome(task)
            return {**answer, "ledger": {name: {**seen, "wrote": sorted(answer)}},
                    "completed": [name]}
        if moved(entry, seen):  # record what came back, map it next entry
            return {"ledger": {name: seen}}
        return self._outcome(task)  # unchanged and unfinished: raises, task recorded

    def _accounted(self, entry: LedgerEntry, seen: LedgerEntry, task: Task,
                   said: Mapping[str, Any] | None) -> LedgerEntry:
        """Chapter 17: the entry keeps the trace that sent the task and what Claims
        last sent on it, and every answer that moves it is receipted."""
        key = entry["idempotency_key"]
        if parent := self.traces.pop(key, ""):
            seen = {**seen, "traceparent": parent}
        if self.receipts is not None:
            if said is not None:
                seen = {**seen, "sent": self.receipts.sent(said, key)}
            if moved(entry, seen) or task.status.state == COMPLETED:
                seen = {**seen, "receipt": self.receipt(seen, task, said).digest}
        return self._costed(entry, seen, task)  # Chapter 18: the charge, if it ended

    def _costed(self, entry: LedgerEntry, seen: LedgerEntry,
                task: Task) -> LedgerEntry:
        """Chapter 18: a task that ended is a charge, priced and put on the ledger."""
        if self.costs is None or "cost" in entry or task.status.state not in FINAL:
            return seen
        key, held = seen["idempotency_key"], (self.partner.terms[0]
                                             if self.partner.terms else None)
        skill = held.skills.get(self.seam.skill) if held else None
        fixed = (skill.cost_per_task if skill and task.status.state == COMPLETED
                 else None)
        price, usage = self.costs.priced(self.seam.name, task, fixed)
        purpose = self.purposes.pop(key, "claim")
        if purpose == "compensation":  # what the partner says is owed (Chapter 13)
            answer = {k: v for a in task.artifacts for p in get_data_parts(a.parts)
                      for k, v in p.items()}
            price = float(answer.get("owed", 0.0))
        now = seamkit.ledger.CLOCK()
        cost = self.costs.ledger.append(CostEntry(
            self.seam.name, self.seam.owner, thread_of(), task.id, key,
            self.seam.skill, purpose, held.revision if held else "", price, usage,
            seen.get("receipt", ""), now))
        self.costs.after(self.seam.name, now)
        return {**seen, "cost": cost.digest}

    def receipt(self, seen: LedgerEntry, task: Task,
                said: Mapping[str, Any] | None) -> SeamReceipt:
        """Cut, check and log the receipt for one answer; a mismatch is raised."""
        assert self.receipts is not None
        receipts, key = self.receipts, seen["idempotency_key"]
        nonce = receipts.nonce(key)
        sent: dict[str, Any] = seen.get("sent") or {"digest": "", "fields": []}
        ours = ReceiptHalf(self.seam.skill, task.id, task.context_id, seen["state"],
                           sent["digest"], digest(answered(task), nonce),
                           tuple(sent["fields"]), receipts.kid, seamkit.ledger.now())
        try:
            theirs = close(ours, task, receipts.pinned, receipts.countersigned,
                           self.seam.name)
        except ReceiptMismatch as exc:  # the answer is not taken; both owners hear
            self.log(self._record("get", ())(task_id=task.id, error=type(exc).__name__))
            receipts.mismatched(exc)
            raise
        held = self.partner.terms[0] if self.partner.terms else None
        undoes = held is not None and any(
            s.compensated_by == self.seam.skill for s in held.skills.values())
        answer = {k: v for p in answered(task)["artifacts"] for k, v in p.items()}
        return receipts.log.append(SeamReceipt(
            self.seam.name, thread_of(), ours.signed(receipts.key), theirs,
            seen["card_digest"], seen["authority"], seen.get("behaviour", ""),
            self.partner.release, held.revision if held else "", key,
            trace_id(seen.get("traceparent", "")),
            approval=str((said or {}).get("approval", "")),
            outcome=answer if undoes else {}))

    async def recover(self, sending: Mapping[str, Any], entry: LedgerEntry) -> Task:
        """An entry left unknown: resend under its key, or find the task it opened."""
        lost = PartnerUnavailable(self.seam.name, "sent, outcome unknown")
        honoured, skew = self.key_terms()
        first, clock = sent_at(entry, self.seam.timeout), seamkit.ledger.CLOCK
        why = refuse_retry(self.seam.retry, "send", lost, 1, honoured, skew, first,
                           clock(), self.breaker_holds())
        if not why:  # the partner answers a repeat with the task the first send opened
            return await retrying(
                lambda: self._send(sending, entry, key=entry["idempotency_key"]),
                "send", self.seam.retry, honoured, skew, first, clock, made=1,
                sleep=self.sleep, held=self.breaker_holds)
        try:  # a key nobody honours, or a window spent: look, and never send again
            if task := await self.lookup(entry):
                return task
        except PartnerUnavailable:
            pass
        raise PartnerUnavailable(self.seam.name, f"sent, outcome unknown: {why}",
                                 entry["task_id"])

    def purpose(self, data: Mapping[str, Any], key: str | None,
                link: Mapping[str, Any] | None) -> str:
        """What a send is for: a canary, a recording, an undo, a trial, or a claim's."""
        if key and key in self.purposes:
            return self.purposes[key]
        held = self.partner.terms[0] if self.partner and self.partner.terms else None
        undoes = held is not None and any(
            s.compensated_by == self.seam.skill for s in held.skills.values())
        trial = self.breaker is not None and self.breaker.state != "closed"
        resend = (link or {}).get("state") == "unknown"  # after a restart: no record
        return purpose_of(data, resend, trial, undoes)

    def breaker_holds(self) -> str:
        """Why the seam's breaker refuses new work now (Chapter 14), or ""."""
        return self.breaker.refuses(seamkit.ledger.CLOCK()) if self.breaker else ""

    def held_back(self) -> bool:
        """The send would be refused before it left: no lookup goes first, either."""
        return bool(self.partner.refuses(self.seam.skill) or self.partner.breaches
                    or self.seam.skill in self.partner.held)

    def key_terms(self) -> tuple[int | None, int]:
        """The key window the partner's terms in force state for the skill, and skew."""
        if not self.partner.terms:
            return None, 0
        held = self.partner.terms[0]
        skill = held.skills.get(self.seam.skill)
        return (skill.key_seconds if skill else None), held.skew_seconds

    async def lookup(self, entry: LedgerEntry) -> Task | None:
        """The task this entry's key opened, found in the partner's list, or None.

        It matches the key against each task's first message, which works only on a
        partner that keeps the first message in a task's history; the specification
        leaves that to the agent. A list can skip a task, so None proves nothing.
        """
        for task in await self.tasks():
            if task.history and task.history[0].message_id == entry["idempotency_key"]:
                return await self._get(task.id)  # the list carries no artifacts
        return None

    async def tasks(self) -> list[Task]:
        """Every task the partner lists for this caller, page by page: one record."""
        found: list[Task] = []
        record, token = self._record("list", ()), ""
        try:
            while True:
                page = await self.partner.client.list_tasks(
                    ListTasksRequest(page_size=50, page_token=token),
                    context=ClientCallContext(timeout=self.seam.timeout))
                found += page.tasks
                if not (token := page.next_page_token):
                    break
        except DelegationRefused as exc:
            self.log(record(error=type(exc).__name__))
            raise
        except (A2AClientError, A2AError) as exc:
            self.log(record(error=type(exc).__name__))
            raise PartnerUnavailable(self.seam.name, str(exc)) from exc
        self.log(record(state=f"{len(found)} tasks"))
        return found

    def intend(self) -> LedgerEntry:
        """The first version of this seam's entry: nothing sent yet, a key minted."""
        return {**intended(self.seam.name, self.seam.skill,
                           card_digest(self.partner.card), self.authority(),
                           self.partner.release),
                "terms": terms_digest(self.partner, self.seam.skill),
                "behaviour": self.partner.behaviour.get(self.seam.skill, "")}

    async def call_once(self, data: Mapping[str, Any],
                        key: str | None = None) -> dict[str, Any]:
        """One task and no ledger entry: the tool call shape (seamkit.tools).

        An argument the model left unset does not cross: only what it chose is sent.
        `key` is the message id, when the caller derives one (Chapter 12).
        """
        chosen = {k: v for k, v in data.items() if v is not None}
        task = await self._send(chosen, key=key)
        if self.receipts is not None and key:  # Chapter 19: a tool call is receipted
            seen = cast(LedgerEntry, {
                **self.intend(), "idempotency_key": key,
                "state": state_name(task.status.state),
                "sent": self.receipts.sent(chosen, key)})
            self.receipt(seen, task, chosen)
        answer = self._outcome(task)
        if self.seam.rules is None:
            return answer
        return withhold(answer, self.seam.rules)  # a model that can act reads no text

    def _waits_for_us(self, task: Task) -> bool:
        """A question is ours only if it names a role we answer for (Chapter 10)."""
        if task.status.state != INPUT_REQUIRED:
            return False
        if not self.seam.roles:  # Chapter 3's rule, for seams that list no roles
            return self.seam.asks_caller
        return addressee(task) in self.seam.roles  # unaddressed: nobody's, refused

    def _outcome(self, task: Task) -> dict[str, Any]:
        error = OUTCOMES[task.status.state]
        if error is not None:
            raise error(self.seam.name, TaskState.Name(task.status.state), task.id)
        answer: dict[str, Any] = {}
        for artifact in task.artifacts:
            for part in get_data_parts(artifact.parts):
                answer.update(part)
        missing = [key for key in self.seam.receives if key not in answer]
        wrong = ([f"artifact lacks {missing}"] if missing
                 else violations(answer, self.seam.shapes))  # the contract's shapes
        self._observe(task.id, clean=not wrong)  # the trust gradient's evidence
        if wrong:
            raise MalformedAnswer(self.seam.name, "; ".join(wrong), task.id)
        if over := self.partner and self.partner.over_ceiling(self.seam.skill, answer):
            raise CardUntrusted(self.seam.name, over, task.id)
        # The field map: only the keys this seam names reach state, under our names.
        mapped = {ours: answer[theirs] for theirs, ours in self.seam.receives.items()}
        if self.seam.rules is None:  # a seam with no ingress filter (before Chapter 8)
            return mapped
        try:
            admitted = admit(mapped, self.seam.rules, self.seam.name, task.id,
                             self.partner.release if self.partner else "")
        except MalformedAnswer:  # a refused field is a malformed answer: it counts
            self._observe(task.id, clean=False)
            raise
        log_findings(self.seam.name, task.id, admitted.findings)
        return admitted.update  # the tags go beside it, never inside it

    def _observe(self, task_id: str, clean: bool) -> None:
        if self.partner is None:  # a node built only to map an answer
            return
        record = self.partner.record(self.seam.skill)
        # One malformed answer suspends the skill under this card.
        (record.clean if clean else record.malformed).add(task_id)

    async def _send(self, data: Mapping[str, Any],
                    link: Mapping[str, Any] | None = None,
                    key: str | None = None) -> Task:
        if why := self.partner.refuses(self.seam.skill):  # the card trust policy
            self.log(self._record("send", tuple(data), key or "")(
                error=CardUntrusted.__name__))
            raise CardUntrusted(self.seam.name, why)
        if self.partner.breaches:  # held per seam: this one stops, the others run
            self.log(self._record("send", tuple(data), key or "")(
                error=ContractBreach.__name__))
            raise ContractBreach(
                self.seam.name, "; ".join(b.detail for b in self.partner.breaches))
        if drift := self.partner.held.get(self.seam.skill):  # Chapter 15: a hold
            self.log(self._record("send", tuple(data), key or "")(
                error=BehaviourDrift.__name__))
            raise BehaviourDrift(self.seam.name, drift)
        # Chapter 14: only a send that opens work asks the breaker; a reply to an open
        # task, a read and a cancel never do
        breaker = None if (link or {}).get("task_id") else self.breaker
        if self.costs is not None and not (link or {}).get("task_id"):
            purpose = self.purpose(data, key, link)
            if why := self.costs.refuses(self.seam.name, purpose,
                                         seamkit.ledger.CLOCK()):
                self.log(self._record("send", tuple(data), key or "")(
                    error=BudgetSpent.__name__))
                raise BudgetSpent(self.seam.name, why)  # Claims stops paying
        if breaker and (why := breaker.allow(seamkit.ledger.CLOCK())):
            self.log(self._record("send", tuple(data), key or "")(
                error=BreakerOpen.__name__))
            raise BreakerOpen(self.seam.name, why)
        started = seamkit.ledger.CLOCK()
        error: type[SeamError] | None = None
        try:
            task = await self._post(data, link, key)
            # a task that came back failed or canceled; any other state is an answer
            failed = OUTCOMES[task.status.state] is PartnerFailed
            error = PartnerFailed if failed else None
            return task
        except SeamError as exc:
            error = type(exc)
            raise
        finally:
            if breaker:  # the outcome, timed on the clock every owner reads
                now = seamkit.ledger.CLOCK()
                breaker.record(error, now - started, now)

    async def _post(self, data: Mapping[str, Any],
                    link: Mapping[str, Any] | None, key: str | None) -> Task:
        message = new_data_message(
            dict(data), role=Role.ROLE_USER,
            context_id=(link or {}).get("context_id") or None,
            task_id=(link or {}).get("task_id") or None)
        if key:  # the sender chooses a message's id; ours is the ledger's key
            message.message_id = key
        # A2A messages carry no skill field; the skill id travels in metadata.
        message.metadata.update({"skill": self.seam.skill})
        task_key = key or str((link or {}).get("idempotency_key", ""))
        if self.receipts is not None and task_key:  # Chapter 17: the task's nonce
            message.metadata.update({NONCE: self.receipts.nonce(task_key)})
        if self.costs is not None:  # Chapter 18: the most Claims will pay for it
            use = self.purposes.setdefault(task_key, self.purpose(data, key, link))
            message.metadata.update({BUDGET_URI: self.costs.ceiling(use).plain()})
        with seam_span(self.seam.name, "send", self.seam.skill, sorted(data)) as span:
            inject(message.metadata)  # the trace context crosses in the message
            if key and (parent := traceparent(span)):
                self.traces[key] = parent  # the ledger entry keeps it
            return await self._exchange(message, data, key)

    async def _exchange(self, message: Message, data: Mapping[str, Any],
                        key: str | None) -> Task:
        request = SendMessageRequest(message=message, configuration=(
            SendMessageConfiguration(
                return_immediately=self.seam.return_immediately,
                task_push_notification_config=self.push(self.seam.name, key))))
        record = self._record("send", tuple(data), key or "")
        task: Task | None = None
        started = seamkit.ledger.CLOCK()
        try:
            async for event in self.partner.client.send_message(
                    request, context=ClientCallContext(timeout=self.seam.timeout)):
                if event.HasField("task"):
                    task = event.task
        except DelegationRefused as exc:  # refused on our side: nothing was sent
            self.log(record(error=type(exc).__name__))
            raise
        except A2AClientError as exc:
            self.log(record(error=type(exc).__name__))
            if why := refused(exc):
                raise PartnerRejected(self.seam.name, why) from exc
            raise PartnerUnavailable(self.seam.name, str(exc)) from exc
        except A2AError as exc:
            self.log(record(error=type(exc).__name__))
            raise PartnerRejected(self.seam.name, str(exc)) from exc
        if task is None:
            self.log(record(error="no task"))
            raise MalformedAnswer(self.seam.name, "the partner replied without a task")
        self.log(record(context_id=task.context_id, task_id=task.id,
                        state=TaskState.Name(task.status.state),
                        seconds=seamkit.ledger.CLOCK() - started))
        return task

    async def probe(self, data: Mapping[str, Any],
                    key: str) -> tuple[dict[str, Any], dict[str, Provenance]]:
        """Chapter 15: one canary task on this seam, judged and never kept.

        It crosses as a real task does: only the fields the seam sends, to a card the
        policy trusts, under a contract in force, through the shape check and the
        ingress filter. It is never observed in a track record, never held to the
        unproven ceiling, and never asks or tells the breaker.
        """
        if why := self.partner.refuses(self.seam.skill):
            raise CardUntrusted(self.seam.name, why)
        if self.partner.breaches:
            raise ContractBreach(
                self.seam.name, "; ".join(b.detail for b in self.partner.breaches))
        task = await self._post({k: data[k] for k in self.seam.sends}, None, key)
        if self.costs is not None:  # Chapter 18: a canary is a real task, billed too
            probed = cast(LedgerEntry, {"idempotency_key": key})
            self._costed(cast(LedgerEntry, {}), probed, task)
        if (error := OUTCOMES[task.status.state]) is not None:
            raise error(self.seam.name, TaskState.Name(task.status.state), task.id)
        answer: dict[str, Any] = {}
        for artifact in task.artifacts:
            for part in get_data_parts(artifact.parts):
                answer.update(part)
        missing = [k for k in self.seam.receives if k not in answer]
        if wrong := ([f"artifact lacks {missing}"] if missing
                     else violations(answer, self.seam.shapes)):
            raise MalformedAnswer(self.seam.name, "; ".join(wrong), task.id)
        mapped = {ours: answer[theirs] for theirs, ours in self.seam.receives.items()}
        if self.seam.rules is None:
            return mapped, {}
        admitted = admit(mapped, self.seam.rules, self.seam.name, task.id,
                         self.partner.release)
        return admitted.update, admitted.tags

    async def read(self, task_id: str) -> Task:
        """Read a recorded task, as a sweep or a resume gate does: one logged get."""
        return await self._get(task_id)

    async def cancel(self, task_id: str) -> str:
        """Ask the partner to cancel a task. Success is not guaranteed: read it after."""
        try:
            await self.partner.client.cancel_task(
                CancelTaskRequest(id=task_id),
                context=ClientCallContext(timeout=self.seam.timeout))
            outcome = "canceled"
        except (A2AClientError, A2AError) as exc:  # already final, or unreachable
            outcome = type(exc).__name__
        self.log(self._record("cancel", ())(task_id=task_id, state=outcome))
        return outcome

    async def _get(self, task_id: str, after: str = "") -> Task:
        """One read; `after` links its span to the send that opened the task."""
        with seam_span(self.seam.name, "read", self.seam.skill, after=after):
            return await self._read(task_id)

    async def _read(self, task_id: str) -> Task:
        record = self._record("get", ())
        try:
            task = await self.partner.client.get_task(
                GetTaskRequest(id=task_id),
                context=ClientCallContext(timeout=self.seam.timeout))
        except DelegationRefused as exc:
            self.log(record(task_id=task_id, error=type(exc).__name__))
            raise
        except A2AClientError as exc:
            self.log(record(task_id=task_id, error=type(exc).__name__))
            if why := refused(exc):
                raise PartnerRejected(self.seam.name, why, task_id) from exc
            raise PartnerUnavailable(self.seam.name, str(exc), task_id) from exc
        except A2AError as exc:
            self.log(record(task_id=task_id, error=type(exc).__name__))
            raise PartnerFailed(self.seam.name, str(exc), task_id) from exc
        self.log(record(context_id=task.context_id, task_id=task.id,
                        state=TaskState.Name(task.status.state)))
        return task

    def _record(self, call: str, sent: tuple[str, ...],
                key: str = "") -> Callable[..., CrossingRecord]:
        def finish(**result: Any) -> CrossingRecord:
            return CrossingRecord(self.seam.name, self.seam.owner, self.partner.release,
                                  call, sent, idempotency_key=key,
                                  behaviour=self.partner.behaviour.get(
                                      self.seam.skill, ""),
                                  at=seamkit.ledger.CLOCK(),
                                  trace_id=current_trace_id(), **result)

        return finish


def sent_at(entry: LedgerEntry, timeout: float) -> float:
    """The earliest the first send can have left: an unknown entry is stamped after the
    send's timeout ran out, so the deadline is counted from before it."""
    return datetime.fromisoformat(entry["seen_at"]).timestamp() - timeout


@dataclass
class FilteredNode(RemoteAgentNode):
    """The node with the two halves of the ingress filter that need the run's state.

    Before a send: egress, against the provenance of every field the seam sends. After
    a completed answer: the provenance tag of every field the answer wrote, under
    `provenance`, beside the update. RemoteAgentNode.__call__ is unchanged.
    """

    policy: CrossingPolicy = NO_CROSSING  # whose classes may cross to which owner

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        entry = state.get("ledger", {}).get(self.seam.name)
        if self.seam.rules is not None and not (entry and entry["task_id"]):
            sending = {k: state[k] for k in self.seam.sends if k in state}
            try:
                egress(sending, state.get("provenance", {}), self.seam.owner,
                       self.policy, self.seam.name)
            except UntrustedEgress as exc:  # nothing was built, nothing crossed
                self.log(self._record("send", tuple(sending))(error=type(exc).__name__))
                raise
        update = await super().__call__(state)
        name = self.seam.name
        if self.seam.rules is None or name not in update.get("completed", []):
            return update
        done = update["ledger"][name]
        return {**update, "provenance": {
            wrote: tag(self.seam.rules[wrote], name, done["task_id"],
                       done["partner_release"]) for wrote in done["wrote"]}}
