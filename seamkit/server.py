"""The server side of a seam: a compiled LangGraph graph behind an A2A executor.

The executor is where the mapping runs in the other direction. The incoming contextId
becomes the partner's own thread_id, a data part becomes graph input (or a resume value
when the thread is paused), interrupt() becomes input-required, and the final state
becomes one artifact. The caller never sees the partner's checkpoint.

Since Chapter 10 a question can name who must answer it (`ask`). The executor keeps a
question addressed to its own people inside: the task stays working, with a status
message naming who it waits on and since when, and the partner's own inbox answers it
later, from outside any request (`answer`). Only a question for the caller crosses.

A partner that verifies delegation chains puts a Guard (seamkit.delegation) in front of
this app. The guard leaves the verified chain in the request's auth, and the executor
hands it to the partner's own run as config, never as state: a chain is a credential,
and a checkpoint is no place for one.

Since Chapter 12 the request handler honours a send's key for every skill whose terms on
the card state a key window (`KeyedHandler`): a repeat of a key inside the window gets
the task the first send opened, and nothing runs again. The SDK's own handler does
nothing with a message id but pass it on.
"""

import asyncio
import hashlib
import json
import time

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from a2a.server.context import ServerCallContext

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    PushNotificationSender,
    TaskManager,
    TaskStore,
    TaskUpdater,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    GetTaskRequest,
    Message,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import apply_history_length
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from jwt.algorithms import OKPAlgorithm
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route

from seamkit.contract import terms_of
from seamkit.costs import Meter, ceiling_of
from seamkit.delegation import Verified
from seamkit.errors import SeamError
from seamkit.ledger import state_name
from seamkit.receipt import NONCE, RECEIPT_URI, ReceiptHalf, answered, digest
from seamkit.tracing import served

AnyGraph = CompiledStateGraph[Any, Any, Any, Any]


def verified_of(call: ServerCallContext) -> Verified | None:
    auth = call.state.get("auth")
    return auth if isinstance(auth, Verified) else None


def delegation_config(call: ServerCallContext) -> dict[str, Any]:
    """The run's authority: the chain the guard verified, for the partner's calls."""
    verified = verified_of(call)
    return {"delegation": verified.chain} if verified else {}


def agent_card(name: str, url: str, release: str, skills: Mapping[str, str],
               extensions: Sequence[AgentExtension] = (),
               push: bool = False,
               protocols: Sequence[str] = ("1.0",)) -> AgentCard:
    """A minimal card: one JSON-RPC interface, no streaming, one entry per skill.

    `extensions` carries what the protocol has no field for, such as seam terms.
    `push` says the partner can send push notifications (Chapter 11).
    `protocols` lists one interface per protocol version, as a partner in the middle
    of a migration serves both (Chapter 15).
    """
    return AgentCard(
        name=name,
        description=f"{name} (Meridian claims federation)",
        version=release,
        supported_interfaces=[AgentInterface(
            protocol_binding="JSONRPC", protocol_version=version, url=url)
            for version in protocols],
        capabilities=AgentCapabilities(streaming=False, push_notifications=push,
                                       extensions=list(extensions)),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[AgentSkill(id=sid, name=sid, description=text, tags=["meridian"])
                for sid, text in skills.items()],
    )


def push_sender(http: httpx.AsyncClient) -> tuple[
        InMemoryPushNotificationConfigStore, PushNotificationSender]:
    """The SDK's push config store and sender: one POST per event, no retry."""
    configs = InMemoryPushNotificationConfigStore()
    return configs, BasePushNotificationSender(http, configs)


@dataclass
class Kept:
    """A key the partner has seen: what was asked, when, and the task it opened."""

    digest: str  # the request's skill and data, hashed: a repeat must match it
    at: float
    task: "asyncio.Future[str]"  # resolved with the task id once the first send has one


Keys = MutableMapping[tuple[str, str], Kept]


def skill_of(message: Message) -> str:
    return str(message.metadata["skill"]) if "skill" in message.metadata else ""


def request_digest(message: Message) -> str:
    """What a repeat must ask for again: the skill and the data, never the ids."""
    asked = {"skill": skill_of(message), "data": get_data_parts(message.parts)}
    raw = json.dumps(asked, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def caller_of(call: ServerCallContext) -> str:
    """Who sent the call: the workload that holds the verified chain, else the user."""
    verified = verified_of(call)
    return verified.path[-1] if verified else call.user.user_name


def key_windows(card: AgentCard) -> dict[str, int]:
    """skill -> the seconds its terms in force promise to honour a key."""
    revisions = terms_of(card)
    held = revisions[0].skills if revisions else {}
    return {sid: s.key_seconds for sid, s in held.items() if s.key_seconds is not None}


@dataclass(frozen=True)
class Arrived:
    """What a caller's message carried on one task, as the partner received it."""

    skill: str
    nonce: str
    sent: str  # digest of the data under the nonce
    fields: tuple[str, ...]


@dataclass
class Countersigner:
    """The partner's half of a seam receipt, cut as an answer leaves (Chapter 17).

    It hashes what arrived before the executor sees it, and what goes back as it
    leaves, under the nonce the caller sent, and signs both with the key behind the
    partner's card. Each task keeps what last arrived on it, so a read 21 minutes
    later is countersigned against the request it answers.
    """

    key: Ed25519PrivateKey
    kid: str
    now: Callable[[], float] = time.time
    arrived: dict[str, Arrived] = field(default_factory=dict)

    @staticmethod
    def received(message: Message) -> Arrived | None:
        """What arrived, hashed under the caller's nonce; None if it sent none."""
        nonce = str(message.metadata[NONCE]) if NONCE in message.metadata else ""
        if not nonce:  # a caller that asks for no receipt gets none
            return None
        data: dict[str, Any] = {}
        for part in get_data_parts(message.parts):
            data.update(part)
        return Arrived(skill_of(message), nonce, digest(data, nonce),
                       tuple(sorted(data)))

    def stamp(self, task: Task) -> Task:
        """The task as it leaves, with the partner's signed half in its metadata."""
        came = self.arrived.get(task.id)
        if came is None:
            return task
        at = datetime.fromtimestamp(self.now(), UTC).isoformat(timespec="seconds")
        half = ReceiptHalf(came.skill, task.id, task.context_id,
                           state_name(task.status.state), came.sent,
                           digest(answered(task), came.nonce), came.fields, self.kid,
                           at)
        out = Task()
        out.CopyFrom(task)  # the stored task is never changed by a receipt
        signed = asdict(half.signed(self.key))
        out.metadata.update({RECEIPT_URI: {**signed, "fields": list(came.fields)}})
        return out

    def forget(self) -> None:
        self.arrived.clear()


class KeyedHandler(DefaultRequestHandler):
    """The partner's half of a key: a repeat inside the window gets the first task.

    Since Chapter 17 it also runs the partner's half of the trace and the receipt:
    every send starts a span of the partner's own, linked to the caller's context,
    and every task it returns, on a send or a read, carries the partner's
    countersignature when the partner countersigns.
    """

    countersigner: Countersigner | None = None
    meter: Meter | None = None  # Chapter 18: the partner's signed usage reports

    def __init__(self, *args: Any, windows: Mapping[str, int], keys: Keys,
                 now: Callable[[], float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.windows, self.keys, self.now = windows, keys, now

    async def on_message_send(self, params: SendMessageRequest,
                              context: ServerCallContext) -> Message | Task:
        message = params.message
        signer = self.countersigner
        arrived = signer.received(message) if signer else None  # before the executor
        if (ceiling := ceiling_of(message)) is not None:  # Chapter 18: for the run
            context.state["ceiling"] = ceiling
        with served(skill_of(message), message.metadata):
            result = await self._keyed(params, context)
        if not isinstance(result, Task):
            return result
        if self.meter is not None:  # the report first: the half signs no metadata
            self.meter.received(result.id, message)
            result = self.meter.stamp(result)
        if signer is None:
            return result
        if arrived is not None:
            signer.arrived[result.id] = arrived
        return signer.stamp(result)

    async def on_get_task(self, params: GetTaskRequest,
                          context: ServerCallContext) -> Task | None:
        task: Task | None = await super().on_get_task(params, context)
        if task is not None and self.meter is not None:
            task = self.meter.stamp(task)
        if task is None or self.countersigner is None:
            return task
        return self.countersigner.stamp(task)

    async def _keyed(self, params: SendMessageRequest,
                     context: ServerCallContext) -> Message | Task:
        message = params.message
        window = self.windows.get(skill_of(message))
        if message.task_id or window is None:  # a follow-up, or no key promised
            return await self._send(params, context)
        index = (caller_of(context), message.message_id)  # scoped to who sent it
        asked, kept = request_digest(message), self.keys.get(index)
        if kept and self.now() - kept.at < window:  # a repeat inside the window
            if kept.digest != asked:  # the same key for another request
                raise InvalidParamsError(message="key reused for a different request")
            task = await self.task_store.get(await kept.task, context)  # may wait
            if not isinstance(task, Task):
                raise InvalidParamsError(message="key names a task no longer held")
            return apply_history_length(task, params.configuration)
        kept = self.keys[index] = Kept(asked, self.now(),
                                       asyncio.get_running_loop().create_future())
        try:
            result = await self._send(params, context)
        except BaseException:
            del self.keys[index]  # no task was recorded: the next send is a first
            kept.task.cancel()
            raise
        kept.task.set_result(result.id if isinstance(result, Task) else "")
        return result

    async def _send(self, params: SendMessageRequest,
                    context: ServerCallContext) -> Message | Task:
        result: Message | Task = await super().on_message_send(params, context)
        return result


def a2a_app(card: AgentCard, executor: AgentExecutor,
            extra: Sequence[BaseRoute] = (),
            store: TaskStore | None = None,
            push: tuple[InMemoryPushNotificationConfigStore,
                        PushNotificationSender] | None = None,
            keys: Keys | None = None,
            now: Callable[[], float] = time.time,
            countersign: Countersigner | None = None,
            meter: Meter | None = None) -> Starlette:
    """A Starlette app serving the card and the JSON-RPC endpoint at the card's URL.

    `extra` adds routes beside them, such as the key set a card's signature names.
    `store` is the task store, when the partner also writes tasks outside a request.
    `push` is the push config store and sender (Chapter 11); without it a push config
    on a send is ignored, and nobody is ever rung.
    `keys` is the key index (Chapter 12), kept across releases like the task store.
    Its windows come from the card's own terms, so what the partner promises and what
    its handler does cannot disagree.
    `countersign` is the partner's half of every seam receipt (Chapter 17), signed with
    the key behind its card; None returns no half, and the caller's receipts are its own.
    `meter` signs a usage report into every task it returns (Chapter 18), and the
    ceiling a caller's message carries reaches the run's config as `ceiling`.
    """
    configs, sender = push or (None, None)
    handler = KeyedHandler(
        agent_executor=executor, task_store=store or InMemoryTaskStore(),
        agent_card=card, push_config_store=configs, push_sender=sender,
        windows=key_windows(card), keys={} if keys is None else keys, now=now)
    handler.countersigner = countersign
    handler.meter = meter
    routes = [*create_agent_card_routes(card), *extra,
              *create_jsonrpc_routes(handler, "/")]
    app = Starlette(routes=routes)
    app.state.handler = handler  # what a test reaches the key index through
    return app


def key_set_route(path: str, keys: Mapping[str, Ed25519PublicKey]) -> Route:
    """A JWK Set at `path`: the keys a partner says its card signatures use.

    Serving it is harmless. Trusting a key because a card's header points here is not:
    whoever serves the card can serve this too.
    """
    body = {"keys": [{**OKPAlgorithm.to_jwk(key, as_dict=True), "kid": kid,
                      "alg": "EdDSA", "use": "sig"} for kid, key in keys.items()]}

    async def key_set(request: Request) -> JSONResponse:
        return JSONResponse(body)

    return Route(path, key_set, methods=["GET"])


def ask(question: Mapping[str, Any], owner: str, role: str) -> Any:
    """interrupt(), with the one thing the protocol never says: who must answer."""
    return interrupt({**question, "to": f"{owner}/{role}"})


@dataclass(frozen=True)
class Waiting:
    """A task kept working on the partner's own people: what to resume it with."""

    context_id: str
    config: RunnableConfig  # the run's config, with the chain the task last held
    to: str
    since: str


class StoreWriter(EventQueue):
    """A queue with no request behind it: each event goes straight to the task store.

    The SDK's way to run an executor's updates outside the framework is a subclass.
    The SDK pushes from a request's event consumer, so a write from here pushes
    nothing unless `push` is given (Chapter 11).
    """

    def __init__(self, store: TaskStore, task_id: str, context_id: str,
                 push: PushNotificationSender | None = None) -> None:
        self.manager = TaskManager(store, ServerCallContext(), task_id, context_id,
                                   None)
        self.task_id, self.push = task_id, push

    async def enqueue_event(self, event: Event) -> None:
        await self.manager.process(event)
        if self.push and isinstance(event, (Task, TaskStatusUpdateEvent,
                                            TaskArtifactUpdateEvent)):
            await self.push.send_notification(self.task_id, event)


class GraphExecutor(AgentExecutor):
    """Serves one compiled graph over A2A. The graph keeps its own checkpointer.

    `own` is the partner's owner name: a question addressed to "own/..." never
    crosses (Chapter 10). Empty keeps Chapter 3's mapping, where every interrupt does.
    """

    def __init__(self, graph: AnyGraph, name: str,
                 to_input: Callable[[dict[str, Any]], Any],
                 to_artifact: Callable[[dict[str, Any]], dict[str, Any]],
                 configure: Callable[[ServerCallContext], dict[str, Any]] = (
                     delegation_config),
                 own: str = "", store: TaskStore | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self.graph, self.name = graph, name
        self.to_input, self.to_artifact = to_input, to_artifact
        self.configure = configure
        self.own, self.store = own, store or InMemoryTaskStore()
        self.now = now
        self.push: PushNotificationSender | None = None  # rings from the inbox too
        self.waiting: dict[str, Waiting] = {}  # task -> a question for our own people
        self.held: dict[str, Any] = {}  # task -> an answer waiting on a fresh chain
        # The partner's own log of what it was asked: one entry per execution.
        self.received: list[dict[str, Any]] = []
        # Task ids whose execution has ended, finished or paused for an answer.
        self.finished: list[str] = []

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        try:
            await self._execute(context, queue)
        finally:  # finished, failed, or paused for an answer
            self.finished.append(context.task_id or "")

    async def _execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        await updater.start_work()
        # The caller's contextId is our thread_id. We minted it; the caller records it.
        call = context.call_context
        # Chapter 12: the key the caller sent, from which every onward key is derived
        config: RunnableConfig = {"configurable": {"thread_id": context.context_id,
                                                   "task_id": context.task_id,
                                                   "inbound_key":
                                                       context.message.message_id,
                                                   **self.configure(call)}}
        if "ceiling" in call.state:  # Chapter 18: the most the caller will pay
            config["configurable"]["ceiling"] = call.state["ceiling"]
        data = get_data_parts(context.message.parts)[0]
        if context.task_id in self.held:  # a renewal: resume with the held answer
            data = self.held.pop(context.task_id)
        paused = (await self.graph.aget_state(config)).interrupts
        run = Command(resume=data) if paused else self.to_input(data)
        headers = call.state.get("headers", {})
        verified = verified_of(call)
        self.received.append({"context_id": context.context_id, "data": data,
                              "task_id": context.task_id, "resumed": bool(paused),
                              "message_id": context.message.message_id,
                              "credential": headers.get("authorization", ""),
                              "chain": verified.chain_id if verified else "",
                              "subject": verified.subject if verified else ""})
        await self._run(updater, run, config)

    async def _run(self, updater: TaskUpdater, run: Any,
                   config: RunnableConfig) -> None:
        try:
            out = await self.graph.ainvoke(run, config)
        except SeamError as exc:  # our own partner failed us: say so, typed
            await updater.failed(updater.new_agent_message(
                [new_data_part({"error": type(exc).__name__, "detail": str(exc)})]))
            return
        if out.get("__interrupt__"):
            question = out["__interrupt__"][0].value
            to = str(question.get("to", "")) if isinstance(question, dict) else ""
            if self.own and to.startswith(f"{self.own}/"):  # ours: it never crosses
                since = self._stamp()
                self.waiting[updater.task_id] = Waiting(
                    updater.context_id, config, to, since)
                await updater.start_work(updater.new_agent_message(
                    [new_data_part({"waiting_on": to, "since": since})]))
                return
            await updater.requires_input(updater.new_agent_message(  # the caller's
                [new_data_part(question)]))
            return
        await updater.add_artifact([new_data_part(self.to_artifact(out))],
                                   name=self.name)
        await updater.complete()

    def _stamp(self) -> str:
        at = self.now() if self.now else datetime.now(UTC).timestamp()
        return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")

    async def answer(self, task_id: str, answer: Any,
                     holds: Callable[[RunnableConfig], str] | None = None) -> str:
        """One of our own people answered, in our own inbox: no request carries it.

        `holds` is the commit act point: does the chain the task last held still hold?
        If not, the answer is kept and the task asks the caller for a fresh chain.
        """
        waiting = self.waiting.pop(task_id)
        updater = TaskUpdater(StoreWriter(self.store, task_id, waiting.context_id,
                                          self.push), task_id, waiting.context_id)
        why = holds(waiting.config) if holds else "ok"
        if why == "revoked":  # final: Chapter 7's rule at the commit act point
            await updater.cancel(updater.new_agent_message(
                [new_data_part({"why": "delegation revoked"})]))
        elif why != "ok":  # expired: renewable by the same person
            self.held[task_id] = answer
            await updater.requires_auth(updater.new_agent_message(
                [new_data_part({"needs": "delegation", "why": f"chain {why}"})]))
        else:
            await self._run(updater, Command(resume=answer), waiting.config)
        return why

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()
