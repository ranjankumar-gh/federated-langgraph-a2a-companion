"""The server side of a seam: a compiled LangGraph graph behind an A2A executor.

The executor is where the mapping runs in the other direction. The incoming contextId
becomes the partner's own thread_id, a data part becomes graph input (or a resume value
when the thread is paused), interrupt() becomes input-required, and the final state
becomes one artifact. The caller never sees the partner's checkpoint.
"""

from collections.abc import Callable, Mapping
from typing import Any

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from starlette.applications import Starlette

from seamkit.errors import SeamError

AnyGraph = CompiledStateGraph[Any, Any, Any, Any]


def agent_card(name: str, url: str, release: str,
               skills: Mapping[str, str]) -> AgentCard:
    """A minimal card: one JSON-RPC interface, no streaming, one entry per skill."""
    return AgentCard(
        name=name,
        description=f"{name} (Meridian claims federation)",
        version=release,
        supported_interfaces=[AgentInterface(
            protocol_binding="JSONRPC", protocol_version="1.0", url=url)],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[AgentSkill(id=sid, name=sid, description=text, tags=["meridian"])
                for sid, text in skills.items()],
    )


def a2a_app(card: AgentCard, executor: AgentExecutor) -> Starlette:
    """A Starlette app serving the card and the JSON-RPC endpoint at the card's URL."""
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")]
    return Starlette(routes=routes)


class GraphExecutor(AgentExecutor):
    """Serves one compiled graph over A2A. The graph keeps its own checkpointer."""

    def __init__(self, graph: AnyGraph, name: str,
                 to_input: Callable[[dict[str, Any]], Any],
                 to_artifact: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.graph, self.name = graph, name
        self.to_input, self.to_artifact = to_input, to_artifact
        # The partner's own log of what it was asked: one entry per execution.
        self.received: list[dict[str, Any]] = []

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        # The caller's contextId is our thread_id. We minted it; the caller records it.
        config: RunnableConfig = {"configurable": {"thread_id": context.context_id}}
        data = get_data_parts(context.message.parts)[0]
        paused = (await self.graph.aget_state(config)).interrupts
        run = Command(resume=data) if paused else self.to_input(data)
        headers = context.call_context.state.get("headers", {})
        self.received.append({"context_id": context.context_id, "data": data,
                              "task_id": context.task_id, "resumed": bool(paused),
                              "credential": headers.get("authorization", "")})
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
