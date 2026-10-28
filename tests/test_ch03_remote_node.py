"""RemoteAgentNode over the real A2A SDK, in process: the thread link and the wire."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from meridian.federated import OWNERS_A2A, A2APartners
from meridian.federated_naive import CLAIMS_CREDENTIAL
from meridian.incident import send_then_ask
from meridian.inprocess import ClaimState
from meridian.risk_server import RISK_URL
from meridian.samples import clean_claim, suspect_claim
from seamkit.errors import PartnerNeedsInput, PartnerUnavailable
from seamkit.inventory import inventory_from_graph
from tests.wire import Federation, Outage, thread

pytestmark = pytest.mark.anyio

POLICE_REPORT = {"police_report": "PR-5521"}


async def test_clean_claim_settles_over_a2a() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("t1"))
    assert out["decision"] == "paid"
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    assert sorted(out["thread_links"]) == ["fraud_screening", "payments",
                                           "repair_estimate"]
    assert [r.seam for r in fed.records] == ["fraud_screening", "repair_estimate",
                                             "payments"]  # one record per crossing


async def test_each_seam_sends_only_the_fields_it_names() -> None:
    fed = Federation()
    await (await fed.graph()).ainvoke(clean_claim(), thread("t2"))
    sent = fed.partners.northgate.executor.received[-1]["request"]
    assert sorted(sent) == ["claim_id", "damaged_parts"]  # no fraud_score this time
    # Since Chapter 6 no credential of Claims' crosses at all: a chain does.
    received = fed.partners.risk.executor.received[-1]
    assert received["credential"] == "" and received["chain"].startswith("chain:")
    assert CLAIMS_CREDENTIAL not in str(received)


async def test_the_partner_mints_the_context_and_we_record_it() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("t3"))
    link = out["thread_links"]["fraud_screening"]
    risk = fed.partners.risk
    assert risk.executor.received[-1]["context_id"] == link["context_id"]
    # Risk keyed its own thread by the context it minted, not by our thread id.
    theirs = risk.graph.get_state({"configurable": {"thread_id": link["context_id"]}})
    assert theirs.values["fraud_signals"] == []


async def test_the_incident_a_send_before_interrupt_sends_twice() -> None:
    fed = Federation()
    await fed.graph()
    graph = StateGraph(ClaimState)
    graph.add_node("fraud_screening",
                   send_then_ask(fed.connections["fraud_screening"]))
    graph.add_edge(START, "fraud_screening")
    graph.add_edge("fraud_screening", END)
    app = graph.compile(checkpointer=InMemorySaver())
    await app.ainvoke(suspect_claim(), thread("i1"))
    await app.ainvoke(Command(resume=POLICE_REPORT), thread("i1"))
    received = fed.partners.risk.executor.received
    new_tasks = [r for r in received if not r["resumed"]]
    assert len(new_tasks) == 2  # the resume re-ran the send
    assert new_tasks[0]["context_id"] != new_tasks[1]["context_id"]


async def test_the_thread_link_makes_the_resume_answer_the_same_task() -> None:
    fed = Federation()
    graph = await fed.graph()
    paused = await graph.ainvoke(suspect_claim(), thread("i2"))
    assert paused["__interrupt__"][0].value["needs"] == "police_report"
    out = await graph.ainvoke(Command(resume=POLICE_REPORT), thread("i2"))
    assert out["decision"] == "refer"
    received = fed.partners.risk.executor.received
    assert [r["resumed"] for r in received] == [False, True]  # one task, answered
    assert {r["task_id"] for r in received} == {
        out["thread_links"]["fraud_screening"]["task_id"]}


async def test_a_question_nobody_planned_to_answer_raises() -> None:
    fed = Federation()
    graph = await fed.graph()
    # Frame damage: Northgate asks for a photo the seam never sends.
    claim: ClaimState = {**clean_claim(), "damaged_parts": ["frame"]}
    with pytest.raises(PartnerNeedsInput) as caught:
        await graph.ainvoke(claim, thread("p1"))
    assert caught.value.seam == "repair_estimate"
    assert "payment_ref" not in (await graph.aget_state(thread("p1"))).values


async def test_an_unreachable_partner_raises_and_is_still_recorded() -> None:
    partners = A2APartners()
    risk = Outage(partners.risk.app)
    fed = Federation(partners)
    graph = await fed.graph({"fraud_screening": risk.http(RISK_URL)})
    risk.down = True
    with pytest.raises(PartnerUnavailable):
        await graph.ainvoke(clean_claim(), thread("u1"))
    assert fed.records[-1].error == "A2AClientError"
    assert fed.records[-1].seam == "fraud_screening"


async def test_the_inventory_reads_the_a2a_graph_unchanged() -> None:
    inventory = inventory_from_graph(await Federation().graph(), OWNERS_A2A)
    assert sorted(e.callee_owner for e in inventory.entries) == [
        "Finance", "Northgate", "Risk"]
