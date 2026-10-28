"""RemoteAgentNode: one LangGraph node that sends one partner skill one A2A task.

It is the node call shape for a seam. It sends only the fields the seam names,
records the partner-issued contextId and taskId against the caller's thread (the
thread link), turns every task state into a state update or a typed SeamError, writes
the partner's artifact into state only through the seam's field map, and emits one
crossing record per call.

Since Chapter 4, `connect` checks the partner's published terms against the seam's
expectations, a breached seam refuses to send, and a completed artifact is checked
against the shapes the seam depends on before the field map writes it.
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
    ContractBreach,
    MalformedAnswer,
    PartnerFailed,
    PartnerNeedsAuth,
    PartnerNeedsInput,
    PartnerRejected,
    PartnerUnavailable,
    SeamError,
    TaskUnfinished,
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

    @property
    def release(self) -> str:
        return self.card.version


contract_logger = logging.getLogger("seamkit.contract")


async def connect(url: str, http: httpx.AsyncClient,
                  expects: Expectations | None = None,
                  policy: CrossingPolicy = NO_CROSSING,
                  terms: Sequence[SeamTerms] = ()) -> PartnerConnection:
    """Fetch the partner's card, build a client, and check the seam's contract.

    With no expectations there is no seam to check (a lookup, a tool's partner).
    `terms` stands in for a partner that publishes none: the terms agreed on paper.
    """
    card = await A2ACardResolver(http, url).get_agent_card()
    factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
    revisions = tuple(terms) or terms_of(card)
    if expects is None:
        return PartnerConnection(card, factory.create(card), revisions)
    now, announced = held_to(revisions, expects, policy)
    for breach in announced:  # found on the day it is announced, not the day it lands
        contract_logger.warning("contract breach announced: %s", breach)
    return PartnerConnection(card, factory.create(card), revisions,
                             tuple(now), tuple(announced))


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
        if self._waits_for_us(task):
            return {"thread_links": links}  # checkpoint the link before interrupt()
        return {**self._outcome(task), "thread_links": links,
                "completed": [self.seam.name]}

    async def call_once(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """One task and no thread link: the tool call shape (seamkit.tools)."""
        return self._outcome(await self._send(data))

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
        if missing:
            raise MalformedAnswer(self.seam.name, f"artifact lacks {missing}", task.id)
        if wrong := violations(answer, self.seam.shapes):  # the contract's shapes
            raise MalformedAnswer(self.seam.name, "; ".join(wrong), task.id)
        # The field map: only the keys this seam names reach state, under our names.
        return {ours: answer[theirs] for theirs, ours in self.seam.receives.items()}

    async def _send(self, data: Mapping[str, Any],
                    link: Mapping[str, str] | None = None) -> Task:
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
        except A2AClientError as exc:
            self.log(record(error=type(exc).__name__))
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
        except A2AClientError as exc:
            self.log(record(task_id=task_id, error=type(exc).__name__))
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
