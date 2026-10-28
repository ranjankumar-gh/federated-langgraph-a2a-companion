"""The tool call shape: a partner skill a model may call, built from the partner's card.

The card supplies the tool's name and description. It cannot supply the argument
schema: an A2A skill carries an id, a name, a description, tags, examples, and media
types, but no input schema. The schema here is the caller's own, which is the gap
Chapter 4 closes.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    log_crossing,
)


def remote_skill_tool(partner: PartnerConnection, skill_id: str, owner: str,
                      args: type[BaseModel], receives: dict[str, str],
                      log: Callable[[CrossingRecord], None] = log_crossing,
                      ) -> StructuredTool:
    skill = next(s for s in partner.card.skills if s.id == skill_id)  # as fetched
    seam = SeamSpec(skill_id, owner, skill_id, tuple(args.model_fields), receives)
    node = RemoteAgentNode(seam, partner, log)

    async def call(**kwargs: Any) -> dict[str, Any]:
        return await node.call_once(kwargs)  # no thread to link: one task per call

    return StructuredTool.from_function(coroutine=call, name=skill.id,
                                        description=skill.description, args_schema=args)
