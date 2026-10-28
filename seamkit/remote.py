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
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig
from a2a.client import A2AClientError, ClientFactory
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import AgentCard, GetTaskRequest, Role, SendMessageConfiguration
from a2a.types import SendMessageRequest, Task, TaskPushNotificationConfig, TaskState
from a2a.utils.errors import A2AError
from langgraph.types import interrupt

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
from seamkit.errors import (
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
    unanswered,
)
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
    # (skill, card digest) -> what Claims observed; shared across refreshes
    records: dict[tuple[str, str], TrackRecord] = field(default_factory=dict)

    @property
    def release(self) -> str:
        return self.card.version

    def record(self, skill: str) -> TrackRecord:
        digest = self.trust.digest if self.trust else ""
        return self.records.setdefault((skill, digest), TrackRecord())

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


async def connect(url: str, http: httpx.AsyncClient,
                  expects: Expectations | None = None,
                  policy: CrossingPolicy = NO_CROSSING,
                  terms: Sequence[SeamTerms] = (),
                  trust: PartnerTrust | None = None,
                  records: dict[tuple[str, str], TrackRecord] | None = None,
                  ) -> PartnerConnection:
    """Fetch the partner's card, decide whether to trust it, and check the contract.

    With no expectations there is no seam to check (a lookup, a tool's partner).
    `terms` stands in for a partner that publishes none: the terms agreed on paper.
    `trust` is the owner's entry in the caller's card trust policy; without one the
    card is used as fetched, which is Chapter 3's behaviour and nobody's production.
    """
    card = await A2ACardResolver(http, url).get_agent_card()
    factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
    kept = {} if records is None else records
    decision = None
    if trust is not None:
        decision = decide(card, url, trust, (expects.skill,) if expects else ())
        log_decision(decision)
        if not decision.trusted:  # a card that fails the policy has no terms to check
            return PartnerConnection(card, factory.create(card), trust=decision,
                                     pinned=trust, records=kept)
    revisions = tuple(terms) or terms_of(card)
    if expects is None:
        return PartnerConnection(card, factory.create(card), revisions,
                                 trust=decision, pinned=trust, records=kept)
    now, announced = held_to(revisions, expects, policy)
    for breach in announced:  # found on the day it is announced, not the day it lands
        contract_logger.warning("contract breach announced: %s", breach)
    return PartnerConnection(card, factory.create(card), revisions, tuple(now),
                             tuple(announced), decision, trust, kept)


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
    return connection if same else fresh


@dataclass(frozen=True)
class CrossingRecord:
    """One structured record per seam crossing: the observability pattern from Ch 1."""

    seam: str
    partner: str
    partner_release: str
    call: str  # "send" or "get"
    sent: tuple[str, ...]  # field names, never values
    context_id: str = ""
    task_id: str = ""
    state: str = ""
    error: str = ""
    idempotency_key: str = ""  # joins a crossing to its ledger entry


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

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        name = self.seam.name
        entry = state.get("ledger", {}).get(name)
        if entry is None:  # the intent and its key are checkpointed before any send
            return {"ledger": {name: self.intend()}}
        if entry["state"] == "unknown":  # sent once, answer lost: never send it again
            raise PartnerUnavailable(name, "sent, outcome unknown", entry["task_id"])
        if not entry["task_id"]:
            try:  # one superstep later: send under the key the checkpoint holds
                task = await self._send({k: state[k] for k in self.seam.sends}, entry,
                                        key=entry["idempotency_key"])
            except PartnerUnavailable:  # record the attempt, then stop on re-entry
                return {"ledger": {name: unanswered(entry)}}
        else:  # a task exists: read it, never send a second one
            task = await self._get(entry["task_id"])
            if self._waits_for_us(task) and entry["state"] == "input-required":
                reply = interrupt(question_of(task))  # asked only once it is recorded
                task = await self._send(reply, entry)  # same task, same context
        seen = observe(entry, task, self.partner.release)
        if task.status.state == COMPLETED:
            answer = self._outcome(task)
            return {**answer, "ledger": {name: {**seen, "wrote": sorted(answer)}},
                    "completed": [name]}
        if moved(entry, seen):  # record what came back, map it next entry
            return {"ledger": {name: seen}}
        return self._outcome(task)  # unchanged and unfinished: raises, task recorded

    def intend(self) -> LedgerEntry:
        """The first version of this seam's entry: nothing sent yet, a key minted."""
        return {**intended(self.seam.name, self.seam.skill,
                           card_digest(self.partner.card), self.authority(),
                           self.partner.release),
                "terms": terms_digest(self.partner, self.seam.skill)}

    async def call_once(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """One task and no ledger entry: the tool call shape (seamkit.tools).

        An argument the model left unset does not cross: only what it chose is sent.
        """
        chosen = {k: v for k, v in data.items() if v is not None}
        answer = self._outcome(await self._send(chosen))
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
        message = new_data_message(
            dict(data), role=Role.ROLE_USER,
            context_id=(link or {}).get("context_id") or None,
            task_id=(link or {}).get("task_id") or None)
        if key:  # the sender chooses a message's id; ours is the ledger's key
            message.message_id = key
        # A2A messages carry no skill field; the skill id travels in metadata.
        message.metadata.update({"skill": self.seam.skill})
        request = SendMessageRequest(message=message, configuration=(
            SendMessageConfiguration(
                return_immediately=self.seam.return_immediately,
                task_push_notification_config=self.push(self.seam.name, key))))
        record = self._record("send", tuple(data), key or "")
        task: Task | None = None
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
                        state=TaskState.Name(task.status.state)))
        return task

    async def read(self, task_id: str) -> Task:
        """Read a recorded task, as a sweep or a resume gate does: one logged get."""
        return await self._get(task_id)

    async def _get(self, task_id: str) -> Task:
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
        def finish(**result: str) -> CrossingRecord:
            return CrossingRecord(self.seam.name, self.seam.owner, self.partner.release,
                                  call, sent, idempotency_key=key, **result)

        return finish


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
