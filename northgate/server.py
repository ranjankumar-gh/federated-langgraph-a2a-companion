"""Northgate's repair-estimate agent: a plain A2A SDK executor, no LangGraph anywhere.

The wire is the same one Risk's LangGraph executor speaks. Nothing in a caller's
RemoteAgentNode can tell which runtime is behind a card, which is the point.

Northgate publishes its seam terms on its card (Chapter 4), as revisions with a date.
Release v8 prices parts per model and so requires the vehicle's VIN; a v7 deployment
may announce v8's terms ahead of the release. The executor enforces the terms it is
deployed with: a request missing a required input is rejected.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from starlette.applications import Starlette

from meridian.agents import PART_PRICES, estimate_repair
from seamkit.contract import Field, SeamTerms, SkillTerms, publish
from seamkit.server import a2a_app, agent_card

NORTHGATE_URL = "http://northgate.example.test/"

# Northgate's terms. Northgate owns them; Meridian's code only reads them off the card.
ESTIMATE_REPAIR = SkillTerms(
    inputs=(Field("claim_id", "string"), Field("damaged_parts", "string[]")),
    outputs=(Field("estimate", "number", unit="claim currency"),),
    asks=("photo_url",),  # frame damage is never estimated unseen
    p95_seconds=4.0, cost_per_task=3.5)
ESTIMATE_HISTORY = SkillTerms(
    inputs=(Field("parts", "string[]"),),
    outputs=(Field("history", "object"),),
    p95_seconds=2.0, cost_per_task=0.5)
NORTHGATE_V7 = SeamTerms(
    "v7", date(2026, 6, 1),
    {"estimate_repair": ESTIMATE_REPAIR, "estimate_history": ESTIMATE_HISTORY},
    auth=("mtls", "delegation"), skew_seconds=30,
    accepts=frozenset({"claim", "vehicle"}))
# v8 prices parts per model, so it needs the VIN: one new required input.
NORTHGATE_V8 = replace(
    NORTHGATE_V7, revision="v8", effective=date(2026, 11, 2),
    skills={**NORTHGATE_V7.skills, "estimate_repair": replace(
        ESTIMATE_REPAIR, inputs=(*ESTIMATE_REPAIR.inputs, Field("vin", "string")))})
TERMS = {"v7": NORTHGATE_V7, "v8": NORTHGATE_V8}

# Past estimates per part across the network, the data behind estimate_history.
HISTORY: dict[str, list[float]] = {
    "bumper": [1_150.0, 1_200.0, 1_240.0],
    "door": [1_700.0, 1_800.0, 1_950.0],
    "frame": [6_100.0, 6_500.0, 7_200.0],
}


class NorthgateExecutor(AgentExecutor):
    def __init__(self, terms: SeamTerms = NORTHGATE_V7) -> None:
        self.terms = terms
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
        if self._lacks(skill, request):
            await updater.reject()
            return
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

    def _lacks(self, skill: str, request: Mapping[str, Any]) -> bool:
        """A required input of the terms this release runs under is missing."""
        terms = self.terms.skills.get(skill)
        return terms is None or any(
            f.required and f.name not in request for f in terms.inputs)

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()


@dataclass
class NorthgatePartner:
    """Northgate's deployment.

    `announces` puts a later revision of the terms on the card beside the one in force.
    `publishes=False` deploys the card as it was before Chapter 4: skills and a
    description, and no terms.
    """

    release: str = "v7"
    announces: SeamTerms | None = None
    publishes: bool = True
    executor: NorthgateExecutor = field(init=False)

    def __post_init__(self) -> None:
        terms = TERMS[self.release]
        self.executor = NorthgateExecutor(terms)
        revisions = [terms, *([self.announces] if self.announces else [])]
        needs = "; needs the vehicle's VIN" if self.release == "v8" else ""
        repair = f"List-price estimate for a claim's damaged parts{needs}"
        self.card = agent_card(
            "Northgate repair estimates", NORTHGATE_URL, self.release,
            {"estimate_repair": repair,
             "estimate_history": "Past estimates across the network for given parts"},
            extensions=[publish(*revisions)] if self.publishes else [])
        self.app: Starlette = a2a_app(self.card, self.executor)
