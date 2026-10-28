"""The tool call shape: a partner skill a model may call, built from the partner's card.

The card supplies the tool's name and description. An A2A skill carries no input
schema, so since Chapter 4 the argument schema is generated from the input terms the
partner publishes in its seam contract, and the outputs it promises become the field
map and the shapes the answer is checked against. A partner that publishes no terms
gets no tool.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from seamkit.contract import args_model
from seamkit.errors import ContractBreach
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    log_crossing,
)


def remote_skill_tool(partner: PartnerConnection, skill_id: str, owner: str,
                      log: Callable[[CrossingRecord], None] = log_crossing,
                      ) -> StructuredTool:
    skill = next(s for s in partner.card.skills if s.id == skill_id)  # as fetched
    terms = partner.terms[0].skills.get(skill_id) if partner.terms else None
    if terms is None:  # no published input terms: no schema a model may fill
        raise ContractBreach(skill_id, f"{owner} publishes no terms for {skill_id}")
    args = args_model(skill_id, terms)  # the partner's inputs, not ours
    outputs = {f.name: f for f in terms.outputs}
    seam = SeamSpec(skill_id, owner, skill_id, tuple(args.model_fields),
                    receives={name: name for name in outputs}, shapes=outputs)
    node = RemoteAgentNode(seam, partner, log)

    async def call(**kwargs: Any) -> dict[str, Any]:
        return await node.call_once(kwargs)  # no thread to link: one task per call

    return StructuredTool.from_function(coroutine=call, name=skill.id,
                                        description=skill.description, args_schema=args)
