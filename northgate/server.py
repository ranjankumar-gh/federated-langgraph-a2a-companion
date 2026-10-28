"""Northgate's repair-estimate agent: a plain A2A SDK executor, no LangGraph anywhere.

The wire is the same one Risk's LangGraph executor speaks. Nothing in a caller's
RemoteAgentNode can tell which runtime is behind a card, which is the point.
"""

from dataclasses import dataclass, field
from typing import Any

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from starlette.applications import Starlette

from meridian.agents import PART_PRICES, estimate_repair
from seamkit.server import a2a_app, agent_card

NORTHGATE_URL = "http://northgate.example.test/"

# Past estimates per part across the network, the data behind estimate_history.
HISTORY: dict[str, list[float]] = {
    "bumper": [1_150.0, 1_200.0, 1_240.0],
    "door": [1_700.0, 1_800.0, 1_950.0],
    "frame": [6_100.0, 6_500.0, 7_200.0],
}


class NorthgateExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.estimates: dict[str, float] = {}

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.message and context.task_id and context.context_id
        if context.current_task is None:
            await queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(queue, context.task_id, context.context_id)
        request = get_data_parts(context.message.parts)[0]
        skill = str(context.message.metadata["skill"])
        self.received.append({"skill": skill, "request": request,
                              "context_id": context.context_id})
        if skill == "estimate_history":
            parts = [p for p in request["parts"] if p in HISTORY]
            history = {p: {"count": len(HISTORY[p]),
                           "mean": sum(HISTORY[p]) / len(HISTORY[p])} for p in parts}
            await updater.add_artifact([new_data_part({"history": history})],
                                       name="estimate_history")
            await updater.complete()
            return
        if skill != "estimate_repair" or not set(request["damaged_parts"]) <= set(
                PART_PRICES):
            await updater.reject()
            return
        if "frame" in request["damaged_parts"] and "photo_url" not in request:
            # Structural damage is never estimated unseen.
            await updater.requires_input(updater.new_agent_message(
                [new_data_part({"needs": "photo_url",
                                "why": "frame damage needs a photo"})]))
            return
        estimate = estimate_repair(request)
        self.estimates[request["claim_id"]] = estimate
        await updater.add_artifact([new_data_part({"estimate": estimate})],
                                   name="repair_estimate")
        await updater.complete()

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()


@dataclass
class NorthgatePartner:
    release: str = "v7"
    executor: NorthgateExecutor = field(default_factory=NorthgateExecutor)

    def __post_init__(self) -> None:
        self.card = agent_card(
            "Northgate repair estimates", NORTHGATE_URL, self.release,
            {"estimate_repair": "List-price estimate for a claim's damaged parts",
             "estimate_history": "Past estimates across the network for given parts"})
        self.app: Starlette = a2a_app(self.card, self.executor)
