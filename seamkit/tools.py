"""The tool call shape: a partner skill a model may call, built from the partner's card.

The card supplies the tool's name and description. An A2A skill carries no input
schema, so since Chapter 4 the argument schema is generated from the input terms the
partner publishes in its seam contract, and the outputs it promises become the field
map and the shapes the answer is checked against. A partner that publishes no terms
gets no tool.

Since Chapter 5, a tool is built only for a skill the card trust policy approved on a
trusted card. A tool built from every skill a card lists is how an unapproved skill
reaches a model.

Since Chapter 8, the tool's answer passes the seam ingress filter, and what the model
gets back is its closed values only. The tool's description is still the card's text;
the card trust policy's pin is what bounds it.
"""

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.tools import StructuredTool

from seamkit.contract import args_model
from seamkit.errors import CardUntrusted, ContractBreach
from seamkit.ingress import IngressRule, rule_for
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    log_crossing,
)


def remote_skill_tool(partner: PartnerConnection, skill_id: str, owner: str,
                      log: Callable[[CrossingRecord], None] = log_crossing,
                      rules: Mapping[str, IngressRule] | None = None,
                      filtered: bool = True) -> StructuredTool:
    """`rules` are the caller's (Chapter 8); without them every output the partner
    promises gets the rule its type gives it, so every string is text and withheld."""
    if why := partner.refuses(skill_id):  # the card trust policy decides, not the card
        raise CardUntrusted(skill_id, why)
    skill = next(s for s in partner.card.skills if s.id == skill_id)  # as fetched
    terms = partner.terms[0].skills.get(skill_id) if partner.terms else None
    if terms is None:  # no published input terms: no schema a model may fill
        raise ContractBreach(skill_id, f"{owner} publishes no terms for {skill_id}")
    args = args_model(skill_id, terms)  # the partner's inputs, not ours
    outputs = {f.name: f for f in terms.outputs}
    held = ({**{n: rule_for(n, f) for n, f in outputs.items()}, **(rules or {})}
            if filtered else None)
    seam = SeamSpec(skill_id, owner, skill_id, tuple(args.model_fields),
                    receives={name: name for name in outputs}, shapes=outputs,
                    rules=held)
    node = RemoteAgentNode(seam, partner, log)

    async def call(**kwargs: Any) -> dict[str, Any]:
        return await node.call_once(kwargs)  # no thread to link: one task per call

    return StructuredTool.from_function(coroutine=call, name=skill.id,
                                        description=skill.description, args_schema=args)
