"""The server side: Claims intake as an A2A agent, and the remote skill tool."""

import httpx
import pytest
from a2a.types import TaskState

from meridian.claims_server import CLAIMS_URL, claims_agent, claims_app
from meridian.federated import SEAMS, A2APartners, connect_partners, in_process_http
from meridian.samples import clean_claim, suspect_claim
from seamkit.errors import PartnerNeedsInput
from seamkit.remote import CrossingRecord, RemoteAgentNode, SeamSpec, connect
from seamkit.tools import remote_skill_tool

pytestmark = pytest.mark.anyio


async def claims_node(partners: A2APartners) -> RemoteAgentNode:
    """A caller of Claims: Claims is now the partner on the other side of a seam."""
    agent = claims_agent(await connect_partners(partners.apps()))
    claims = await connect(CLAIMS_URL, in_process_http(CLAIMS_URL, agent.app, "caller"))
    seam = SeamSpec("settle_claim", "Claims", "settle_claim",
                    sends=tuple(suspect_claim()), receives={"decision": "decision"},
                    asks_caller=False)
    return RemoteAgentNode(seam, claims, log=lambda r: None)


async def test_claims_intake_answers_over_a2a() -> None:
    node = await claims_node(A2APartners())
    assert await node.call_once(clean_claim()) == {"decision": "paid"}


async def test_a_claims_interrupt_reaches_its_caller_as_input_required() -> None:
    partners = A2APartners()
    node = await claims_node(partners)
    with pytest.raises(PartnerNeedsInput):  # Risk's question, passed up by Claims
        await node.call_once(suspect_claim())
    task = await node._send(suspect_claim())
    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    link = {"context_id": task.context_id, "task_id": task.id}
    done = await node._send({"police_report": "PR-5521"}, link)  # same task
    assert node._outcome(done) == {"decision": "refer"}


async def test_the_lifespan_connects_partners_once_and_serves_claims() -> None:
    app = claims_app(A2APartners().apps())
    async with app.router.lifespan_context(app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=CLAIMS_URL)
        claims = await connect(CLAIMS_URL, http)
        assert claims.card.skills[0].id == "settle_claim"


async def test_a_remote_skill_tool_is_built_from_the_card_as_fetched() -> None:
    partners = A2APartners()
    conns = await connect_partners(partners.apps())
    records: list[CrossingRecord] = []
    # Since Chapter 4 the argument schema comes from Northgate's published terms.
    tool = remote_skill_tool(conns["repair_estimate"], "estimate_history", "Northgate",
                             log=records.append)
    assert tool.name == "estimate_history"
    assert tool.description.startswith("Past estimates")
    out = await tool.ainvoke({"parts": ["frame"]})  # what a model's tool call does
    assert out["history"]["frame"]["count"] == 3
    assert records[0].sent == ("parts",)
    assert SEAMS["repair_estimate"].skill == "estimate_repair"  # the node seam differs
