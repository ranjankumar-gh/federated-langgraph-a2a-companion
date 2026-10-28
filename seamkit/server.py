"""The server side of a seam: a compiled LangGraph graph behind an A2A executor.

The executor is where the mapping runs in the other direction. The incoming contextId
becomes the partner's own thread_id, a data part becomes graph input (or a resume value
when the thread is paused), interrupt() becomes input-required, and the final state
becomes one artifact. The caller never sees the partner's checkpoint.

A partner that verifies delegation chains puts a Guard (seamkit.delegation) in front of
this app. The guard leaves the verified chain in the request's auth, and the executor
hands it to the partner's own run as config, never as state: a chain is a credential,
and a checkpoint is no place for one.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from a2a.server.context import ServerCallContext

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jwt.algorithms import OKPAlgorithm
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route

from seamkit.delegation import Verified
from seamkit.errors import SeamError

AnyGraph = CompiledStateGraph[Any, Any, Any, Any]


def verified_of(call: ServerCallContext) -> Verified | None:
    auth = call.state.get("auth")
    return auth if isinstance(auth, Verified) else None


def delegation_config(call: ServerCallContext) -> dict[str, Any]:
    """The run's authority: the chain the guard verified, for the partner's calls."""
    verified = verified_of(call)
    return {"delegation": verified.chain} if verified else {}


def agent_card(name: str, url: str, release: str, skills: Mapping[str, str],
               extensions: Sequence[AgentExtension] = ()) -> AgentCard:
    """A minimal card: one JSON-RPC interface, no streaming, one entry per skill.

    `extensions` carries what the protocol has no field for, such as seam terms.
    """
    return AgentCard(
        name=name,
        description=f"{name} (Meridian claims federation)",
        version=release,
        supported_interfaces=[AgentInterface(
            protocol_binding="JSONRPC", protocol_version="1.0", url=url)],
        capabilities=AgentCapabilities(streaming=False, extensions=list(extensions)),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[AgentSkill(id=sid, name=sid, description=text, tags=["meridian"])
                for sid, text in skills.items()],
    )


def a2a_app(card: AgentCard, executor: AgentExecutor,
            extra: Sequence[BaseRoute] = ()) -> Starlette:
    """A Starlette app serving the card and the JSON-RPC endpoint at the card's URL.

    `extra` adds routes beside them, such as the key set a card's signature names.
    """
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    routes = [*create_agent_card_routes(card), *extra,
              *create_jsonrpc_routes(handler, "/")]
    return Starlette(routes=routes)


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


class GraphExecutor(AgentExecutor):
    """Serves one compiled graph over A2A. The graph keeps its own checkpointer."""

    def __init__(self, graph: AnyGraph, name: str,
                 to_input: Callable[[dict[str, Any]], Any],
                 to_artifact: Callable[[dict[str, Any]], dict[str, Any]],
                 configure: Callable[[ServerCallContext], dict[str, Any]] = (
                     delegation_config)) -> None:
        self.graph, self.name = graph, name
        self.to_input, self.to_artifact = to_input, to_artifact
        self.configure = configure
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
        config: RunnableConfig = {"configurable": {"thread_id": context.context_id,
                                                   **self.configure(call)}}
        data = get_data_parts(context.message.parts)[0]
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
        try:
            out = await self.graph.ainvoke(run, config)
        except SeamError as exc:  # our own partner failed us: say so, typed
            await updater.failed(updater.new_agent_message(
                [new_data_part({"error": type(exc).__name__, "detail": str(exc)})]))
            return
        if out.get("__interrupt__"):  # interrupt() becomes input-required
            question = out["__interrupt__"][0].value
            await updater.requires_input(updater.new_agent_message(
                [new_data_part(question)]))
            return
        await updater.add_artifact([new_data_part(self.to_artifact(out))],
                                   name=self.name)
        await updater.complete()

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()
