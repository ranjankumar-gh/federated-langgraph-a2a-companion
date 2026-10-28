"""RemoteAgentNode: one LangGraph node that sends one partner skill one A2A task.

It is the node call shape for a seam. It sends only the fields the seam names,
records the partner-issued contextId and taskId against the caller's thread (the
thread link), turns every task state into a state update or a typed SeamError, writes
the partner's artifact into state only through the seam's field map, and emits one
crossing record per call.

Since Chapter 4, `connect` checks the partner's published terms against the seam's
expectations, a breached seam refuses to send, and a completed artifact is checked
against the shapes the seam depends on before the field map writes it.

Since Chapter 5, `connect` runs the card trust policy first: a card the policy does not
trust has no terms worth checking. The send step refuses an untrusted card, an
unapproved skill, or a suspended one, and every completed answer is observed in the
skill's track record under the card's digest (the trust gradient).
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig
from a2a.client import A2AClientError, ClientFactory
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import AgentCard, GetTaskRequest, Role, SendMessageRequest, Task
from a2a.types import TaskState
from a2a.utils.errors import A2AError
from langgraph.types import interrupt

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
    # artifact key -> the shape the seam depends on, from its expectations (Chapter 4)
    shapes: Mapping[str, Field] = field(default_factory=dict)


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


crossing_logger = logging.getLogger("seamkit.crossing")


def log_crossing(record: CrossingRecord) -> None:
    crossing_logger.info("seam crossing", extra={"crossing": asdict(record)})


def refused(exc: A2AClientError) -> str:
    """The partner guard's reason if this is its 403 refusal; "" for an outage."""
    cause = exc.__cause__
    if not (isinstance(cause, httpx.HTTPStatusError)
            and cause.response.status_code == 403):
        return ""
    try:
        return str(cause.response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return "refused"


def question_of(task: Task) -> dict[str, Any]:
    """The partner's question, as the data part of its input-required status message."""
    parts = get_data_parts(task.status.message.parts)
    return {"task_id": task.id, **(parts[0] if parts else {})}


@dataclass
class RemoteAgentNode:
    """The node call shape: one partner skill, one task, an update or a SeamError."""

    seam: SeamSpec
    partner: PartnerConnection
    log: Callable[[CrossingRecord], None] = field(default=log_crossing)
    # Names the authority a task is sent under, never its value: the run's chain id.
    authority: Callable[[], str] = field(default=lambda: "")

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        links = dict(state.get("thread_links", {}))
        link = links.get(self.seam.name)
        if link is None:  # first entry: one new task, in a context the partner mints
            task = await self._send({k: state[k] for k in self.seam.sends})
        else:  # re-entry: read the task already sent, never send a second one
            task = await self._get(link["task_id"])
            if self._waits_for_us(task):
                reply = interrupt(question_of(task))  # Claims' own human answers
                task = await self._send(reply, link)  # same task, same context
        links[self.seam.name] = {"context_id": task.context_id, "task_id": task.id}
        if authority := self.authority():  # the run's chain id, beside its task (Ch 6)
            links[self.seam.name]["authority"] = authority
        if self._waits_for_us(task):
            return {"thread_links": links}  # checkpoint the link before interrupt()
        return {**self._outcome(task), "thread_links": links,
                "completed": [self.seam.name]}

    async def call_once(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """One task and no thread link: the tool call shape (seamkit.tools).

        An argument the model left unset does not cross: only what it chose is sent.
        """
        chosen = {k: v for k, v in data.items() if v is not None}
        return self._outcome(await self._send(chosen))

    def _waits_for_us(self, task: Task) -> bool:
        return self.seam.asks_caller and task.status.state == INPUT_REQUIRED

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
        return {ours: answer[theirs] for theirs, ours in self.seam.receives.items()}

    def _observe(self, task_id: str, clean: bool) -> None:
        if self.partner is None:  # a node built only to map an answer
            return
        record = self.partner.record(self.seam.skill)
        # One malformed answer suspends the skill under this card.
        (record.clean if clean else record.malformed).add(task_id)

    async def _send(self, data: Mapping[str, Any],
                    link: Mapping[str, str] | None = None) -> Task:
        if why := self.partner.refuses(self.seam.skill):  # the card trust policy
            self.log(self._record("send", tuple(data))(error=CardUntrusted.__name__))
            raise CardUntrusted(self.seam.name, why)
        if self.partner.breaches:  # held per seam: this one stops, the others run
            self.log(self._record("send", tuple(data))(error=ContractBreach.__name__))
            raise ContractBreach(
                self.seam.name, "; ".join(b.detail for b in self.partner.breaches))
        message = new_data_message(
            dict(data), role=Role.ROLE_USER,
            context_id=link["context_id"] if link else None,
            task_id=link["task_id"] if link else None)
        # A2A messages carry no skill field; the skill id travels in metadata.
        message.metadata.update({"skill": self.seam.skill})
        record = self._record("send", tuple(data))
        task: Task | None = None
        try:
            async for event in self.partner.client.send_message(
                    SendMessageRequest(message=message),
                    context=ClientCallContext(timeout=self.seam.timeout)):
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

    def _record(self, call: str,
                sent: tuple[str, ...]) -> Callable[..., CrossingRecord]:
        def finish(**result: str) -> CrossingRecord:
            return CrossingRecord(self.seam.name, self.seam.owner, self.partner.release,
                                  call, sent, **result)

        return finish
