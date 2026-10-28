"""The partner stub server: a plain A2A SDK server that answers as a recording says.

`StubPartner` is built from a recording (the card as fetched, and the exchanges), the
partner's terms (the card's own, or the terms agreed on paper for a partner that
publishes none), scripted exchanges, and a fault script. At build time every exchange
is held to the terms in force, and one the terms do not allow refuses the build
(`StubOutOfContract`): a stub cannot promise what the partner never did.

The executor answers a send from the fault script first, then from the first exchange
whose request matches, recorded before canary before scripted. A send nothing matches
is rejected with "no recording", never answered with an invented value.

The stub serves the partner's card, re-signed with a test key. That key belongs in a
CI trust policy and nowhere else: anywhere else, a stub's card is a look-alike card that
your own policy approved.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    PushNotificationSender,
    TaskUpdater,
)
from a2a.types import AgentCard
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from google.protobuf.json_format import ParseDict
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seamkit.cards import card_digest, sign_card
from seamkit.contract import SeamTerms, terms_of, violations
from seamkit.push import TOKEN_HEADER
from seamkit.server import (
    Countersigner,
    StoreWriter,
    a2a_app,
    key_set_route,
    push_sender,
    skill_of,
)
from stub.faults import EXECUTOR, WIRE, Fault, FaultScript
from stub.recording import Exchange, Recording

# The key the stub signs its copy of the partner's card with. Only a CI trust policy
# pins it; a test asserts production's policy never does.
STUB_KID = "claims-ci-stub"
STUB_KEY = Ed25519PrivateKey.generate()
STUB_KEYS_PATH = "/.well-known/stub-jwks.json"

ANSWERS = frozenset({"completed", "input-required", "auth-required", "rejected"})
ORDER = {"recorded": 0, "canary": 1, "scripted": 2}
ANY = "*"  # a scripted request may leave a value open, such as a task id


class StubOutOfContract(Exception):
    """An exchange the partner's terms do not allow: the stub refuses to build."""


def out_of_contract(ex: Exchange, terms: SeamTerms, offered: set[str]) -> str:
    """Why the terms in force do not allow this exchange, or "" if they do."""
    skill = terms.skills.get(ex.skill)
    if skill is None or ex.skill not in offered:
        return f"{ex.skill} is not a skill on this card under terms {terms.revision}"
    takes = {f.name for f in skill.inputs} | set(skill.asks)  # a reply carries an ask
    if extra := sorted(set(ex.request) - takes):
        return f"the request carries {', '.join(extra)}, which {ex.skill} never takes"
    if ex.state not in ANSWERS:  # a failure to answer belongs in the fault script
        return f"a {ex.state} task is a fault, not an answer: script it as one"
    needs = ex.message.get("needs")
    if ex.state == "input-required" and needs not in skill.asks:
        return f"{ex.skill} asks for {needs!r}, which it never declares"
    if ex.state != "completed":
        return ""
    shapes = {f.name: f for f in skill.outputs}
    if lacks := [n for n, f in shapes.items() if f.required and n not in ex.artifact]:
        return f"the answer lacks {', '.join(lacks)}"
    if extra := sorted(set(ex.artifact) - set(shapes)):
        return f"the answer carries {', '.join(extra)}, which the terms never declare"
    return "; ".join(violations(ex.artifact, shapes))


def _matches(asked: Mapping[str, Any], held: Mapping[str, Any]) -> bool:
    return asked.keys() == held.keys() and all(
        v == ANY or asked[k] == v for k, v in held.items())


@dataclass
class Hung:
    """A task the fault script keeps working until the test finishes it."""

    context_id: str
    skill: str
    exchange: Exchange | None
    fault: Fault


class StubExecutor(AgentExecutor):
    """Answers from the fault script, then the exchanges, then "no recording"."""

    def __init__(self, exchanges: Sequence[Exchange], script: FaultScript) -> None:
        self.exchanges = sorted(exchanges, key=lambda e: ORDER[e.source])
        self.script = script
        self.received: list[dict[str, Any]] = []  # the stub's log of what it was sent
        self.hung: dict[str, Hung] = {}

    def answer_for(self, skill: str, request: Mapping[str, Any]) -> Exchange | None:
        return next((e for e in self.exchanges
                     if e.skill == skill and _matches(request, e.request)), None)

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        skill = skill_of(context.message)
        request = get_data_parts(context.message.parts)[0]
        self.received.append({"skill": skill, "request": request,
                              "task_id": context.task_id})
        fault = self.script.match(skill, request, EXECUTOR)
        exchange = self.answer_for(skill, request)
        if fault is not None and fault.kind == "fail":
            await updater.failed(updater.new_agent_message(
                [new_data_part({"error": fault.detail or "stub fault"})]))
        elif fault is not None:  # hang, or push_lost: working until finish()
            await updater.start_work()
            self.hung[context.task_id] = Hung(context.context_id, skill, exchange,
                                              fault)
        else:
            await self.reply(updater, skill, request, exchange)

    async def reply(self, updater: TaskUpdater, skill: str,
                    request: Mapping[str, Any], exchange: Exchange | None) -> None:
        if exchange is None:  # never an invented answer
            await updater.reject(updater.new_agent_message([new_data_part(
                {"why": f"no recording for {skill} {json.dumps(request)}"[:200]})]))
            return
        said = ([new_data_part(dict(exchange.message))] if exchange.message else [])
        if exchange.state == "completed":
            await updater.add_artifact([new_data_part(dict(exchange.artifact))],
                                       name=skill)
            await updater.complete()
        elif exchange.state == "input-required":
            await updater.requires_input(updater.new_agent_message(said))
        elif exchange.state == "auth-required":
            await updater.requires_auth(updater.new_agent_message(said))
        else:
            await updater.reject(updater.new_agent_message(said) if said else None)

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        self.hung.pop(context.task_id, None)
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()


@dataclass
class Faulted:
    """The stub's front door: the wire faults, applied before the SDK sees a send."""

    app: ASGIApp
    script: FaultScript
    advance: Callable[[float], None] | None  # the drill's clock, for "slow"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        fault = self._fault(body)

        async def replay() -> Message:
            return {"type": "http.request", "body": body, "more_body": False}

        if fault is not None and fault.kind == "down":  # nothing ran
            await _status(send, 503)
            return
        if fault is not None and fault.kind == "slow" and self.advance:
            self.advance(fault.seconds)  # the partner's own time, on the clock
        if fault is not None and fault.kind == "lose_answer":

            async def drop(message: Message) -> None:  # the work is done, unseen
                return None

            await self.app(scope, replay, drop)
            await _status(send, 502)
            return
        await self.app(scope, replay, send)

    def _fault(self, body: bytes) -> Fault | None:
        try:
            sent = json.loads(body)
        except ValueError:
            return None
        if not isinstance(sent, dict) or sent.get("method") != "SendMessage":
            return None
        message = sent["params"]["message"]
        data: dict[str, Any] = {}
        for part in message.get("parts", []):
            data.update(part.get("data", {}))
        return self.script.match(message.get("metadata", {}).get("skill", ""), data,
                                 WIRE)


async def _status(send: Send, code: int) -> None:
    await send({"type": "http.response.start", "status": code, "headers": []})
    await send({"type": "http.response.body", "body": b""})


@dataclass
class StubPartner:
    """A partner Claims cannot deploy, stood in for by what Claims holds about it.

    `terms` stands in for a partner that publishes none: the terms agreed on paper,
    as Chapter 4's `connect(terms=...)` takes them. `push` is the client the stub
    rings Claims' receiver with; `advance` moves the drill's clock for a slow fault.
    """

    recording: Recording
    scripted: Sequence[Exchange] = ()
    script: FaultScript = field(default_factory=FaultScript)
    terms: Sequence[SeamTerms] = ()
    push: httpx.AsyncClient | None = None
    now: Callable[[], float] = time.time
    advance: Callable[[float], None] | None = None

    def __post_init__(self) -> None:
        self.card = ParseDict(dict(self.recording.card), AgentCard(),
                              ignore_unknown_fields=True)
        self.url = self.card.supported_interfaces[0].url
        revisions = tuple(self.terms) or terms_of(self.card)
        if not revisions:
            raise StubOutOfContract("no terms: a stub is held to the partner's terms, "
                                    "published or agreed on paper")
        held, offered = revisions[0], {s.id for s in self.card.skills}
        for ex in [*self.recording.exchanges, *self.scripted]:
            if why := out_of_contract(ex, held, offered):
                raise StubOutOfContract(f"{ex.source} {ex.skill}: {why}")
            if ex.source != "scripted" and ANY in ex.request.values():
                raise StubOutOfContract(f"{ex.source} {ex.skill}: only a scripted "
                                        "request may leave a value open")
        sign_card(self.card, STUB_KEY, STUB_KID, self.url.rstrip("/") + STUB_KEYS_PATH)
        self.executor = StubExecutor([*self.recording.exchanges, *self.scripted],
                                     self.script)
        self.store = InMemoryTaskStore()
        pushes = push_sender(self.push) if self.push else None
        self.configs: InMemoryPushNotificationConfigStore | None = None
        self.sender: PushNotificationSender | None = None
        if pushes:
            self.configs, self.sender = pushes
        # Chapter 17: terms that countersign get a stub that does, with its CI key
        signs = Countersigner(STUB_KEY, STUB_KID, self.now) if (
            held.receipts == "countersigned") else None
        inner = a2a_app(self.card, self.executor,
                        [key_set_route(STUB_KEYS_PATH,
                                       {STUB_KID: STUB_KEY.public_key()})],
                        store=self.store, push=pushes, now=self.now, countersign=signs)
        if self.terms:  # paper terms: the key windows are the ones agreed, not carded
            inner.state.handler.windows = {
                sid: s.key_seconds for sid, s in held.skills.items()
                if s.key_seconds is not None}
        self.handler = inner.state.handler
        self.app: ASGIApp = Faulted(inner, self.script, self.advance)

    @property
    def digest(self) -> str:
        return card_digest(self.card)

    def sent(self, skill: str) -> int:
        """How many sends of `skill` reached the stub's executor."""
        return sum(1 for r in self.executor.received if r["skill"] == skill)

    async def finish(self, task_id: str) -> None:
        """A hung task answers now, as its exchange says; a lost push rings nobody."""
        hung = self.executor.hung.pop(task_id)
        sender = None if hung.fault.kind == "push_lost" else self.sender
        writer = StoreWriter(self.store, task_id, hung.context_id, sender)
        updater = TaskUpdater(writer, task_id, hung.context_id)
        task = await self.store.get(task_id, ServerCallContext())
        assert task is not None
        request = get_data_parts(task.history[0].parts)[0] if task.history else {}
        await self.executor.reply(updater, hung.skill, request, hung.exchange)

    async def forge(self, task_id: str, state: str = "TASK_STATE_FAILED") -> int:
        """A push nobody at the partner sent: the task's real push config, as if its
        token had leaked, and a state the task never reached. The receiver's status."""
        assert self.configs is not None and self.push is not None
        [config] = await self.configs.get_info_for_dispatch(task_id)
        task = await self.store.get(task_id, ServerCallContext())
        assert task is not None
        body = {"statusUpdate": {"taskId": task_id, "contextId": task.context_id,
                                 "status": {"state": state}}}
        response = await self.push.post(config.url, json=body,
                                        headers={TOKEN_HEADER: config.token})
        return response.status_code

    async def restore(self) -> None:
        """The partner comes back from a backup taken before any of these tasks: every
        task and every key it held is gone, and nothing tells the caller."""
        for task_id in {r["task_id"] for r in self.executor.received}:
            await self.store.delete(task_id, ServerCallContext())
        self.executor.hung.clear()
        self.handler.keys.clear()
        if self.handler.countersigner:
            self.handler.countersigner.forget()


__all__ = ["ANY", "STUB_KEY", "STUB_KEYS_PATH", "STUB_KID", "StubExecutor",
           "StubOutOfContract", "StubPartner", "out_of_contract"]
