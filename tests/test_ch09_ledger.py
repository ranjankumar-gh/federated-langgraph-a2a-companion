"""The task ledger: the incident it fixes, a restart, a forgetful partner, two writers."""

import asyncio
from typing import Any

import pytest
from a2a.types import GetTaskRequest, Task, TaskState, TaskStatus
from a2a.utils.errors import TaskNotFoundError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from meridian.federated import A2APartners, connect_partners
from meridian.incident import LinkOnReturn
from meridian.inprocess import ClaimState
from meridian.samples import clean_claim, suspect_claim
from northgate.server import NorthgatePartner
from seamkit.cards import card_digest
from seamkit.errors import PartnerUnavailable, StaleLedgerWrite, TaskUnfinished
from seamkit.ledger import LedgerEntry, follow_on, intended, merge_ledger, observe
from seamkit.remote import RemoteAgentNode
from tests.wire import (
    RETURNS_EARLY,
    Federation,
    Outage,
    as_run,
    new_tasks,
    slow_risk,
    thread,
)

pytestmark = pytest.mark.anyio

SCREENING = "fraud_screening"


def task(state: int, task_id: str = "t-1", context_id: str = "c-1") -> Task:
    status = TaskStatus(state=state)  # type: ignore[arg-type]
    return Task(id=task_id, context_id=context_id, status=status)


def entry(**changes: Any) -> LedgerEntry:
    base = intended(SCREENING, "screen_fraud", "sha256:x", "chain:0123456789abcdef",
                    "2026.12")
    return {**base, **changes}  # type: ignore[typeddict-item]


# The reducer on its own -------------------------------------------------------------

def test_a_write_from_an_older_read_is_refused() -> None:
    read = entry()
    first = observe(read, task(TaskState.TASK_STATE_WORKING), "2026.12")
    ledger = merge_ledger({}, {SCREENING: read})
    ledger = merge_ledger(ledger, {SCREENING: first})
    late = observe(read, task(TaskState.TASK_STATE_SUBMITTED), "2026.12")  # also v2
    with pytest.raises(StaleLedgerWrite):
        merge_ledger(ledger, {SCREENING: late})


def test_a_final_state_never_slides_back() -> None:
    done = entry(state="completed", task_id="t-1", version=4)
    working = observe(done, task(TaskState.TASK_STATE_WORKING), "2026.12")  # v5
    with pytest.raises(StaleLedgerWrite, match="completed is final"):
        merge_ledger({SCREENING: done}, {SCREENING: working})


def test_a_second_task_while_one_is_open_is_refused() -> None:
    open_task = entry(state="working", task_id="t-1", version=3)
    second = entry(version=4)  # a fresh key over an open task: the incident
    with pytest.raises(StaleLedgerWrite, match="second task"):
        merge_ledger({SCREENING: open_task}, {SCREENING: second})


# The incident, on Chapter 3's node and on the ledger ----------------------------------

async def test_the_incident_an_unfinished_task_loses_its_link() -> None:
    risk = slow_risk()
    fed = Federation(A2APartners(risk=risk), seams=RETURNS_EARLY)
    await fed.graph()
    graph = StateGraph(ClaimState)
    graph.add_node(SCREENING, LinkOnReturn(  # type: ignore[type-var]
        RETURNS_EARLY[SCREENING],
        fed.connections[SCREENING]))
    graph.add_edge(START, SCREENING)
    graph.add_edge(SCREENING, END)
    app = graph.compile(checkpointer=InMemorySaver())
    for _ in range(3):  # the adjuster re-runs the claim; every run sends again
        with pytest.raises(TaskUnfinished):
            await app.ainvoke(suspect_claim(), thread("i9"))
    await risk.finish_lookups()
    assert len(new_tasks(risk)) == 3  # three tasks at Risk, all completed at 82
    assert "thread_links" not in (await app.aget_state(thread("i9"))).values


async def test_the_ledger_keeps_the_task_through_the_same_incident() -> None:
    risk = slow_risk()
    fed = Federation(A2APartners(risk=risk), seams=RETURNS_EARLY)
    graph = await fed.graph()
    with pytest.raises(TaskUnfinished):
        await graph.ainvoke(suspect_claim(), thread("l1"))
    seen = (await graph.aget_state(thread("l1"))).values["ledger"][SCREENING]
    assert seen["state"] == "working" and seen["task_id"]
    await risk.finish_lookups()
    out = await graph.ainvoke(suspect_claim(), thread("l1"))  # the same re-run
    assert out["decision"] == "refer"
    assert [r["task_id"] for r in new_tasks(risk)] == [seen["task_id"]]  # one task
    final = out["ledger"][SCREENING]
    assert final["state"] == "completed" and final["wrote"] == ["fraud_score"]
    assert final["version"] > seen["version"]


async def test_the_key_is_sent_as_the_message_id_and_logged() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("k1"))
    screening = out["ledger"][SCREENING]
    key = screening["idempotency_key"]
    assert fed.partners.risk.executor.received[0]["message_id"] == key
    sends = [r for r in fed.records if r.seam == SCREENING and r.call == "send"]
    assert [r.idempotency_key for r in sends] == [key]
    assert screening["card_digest"] == card_digest(fed.connections[SCREENING].card)
    # The chain the task was sent under, by id: a name, never the chain itself.
    assert screening["authority"] == fed.partners.risk.executor.received[0]["chain"]


# A restart, and a partner that forgets -----------------------------------------------

async def test_a_restarted_claims_reads_the_task_it_already_sent() -> None:
    saver, risk = InMemorySaver(), slow_risk()
    before = Federation(A2APartners(risk=risk), seams=RETURNS_EARLY)
    with pytest.raises(TaskUnfinished):
        await (await before.graph(checkpointer=saver)).ainvoke(suspect_claim(),
                                                               thread("r1"))
    await risk.finish_lookups()
    after = Federation(before.partners, seams=RETURNS_EARLY)  # new clients and graph
    graph = await after.graph(checkpointer=saver)  # over the same checkpoint store
    out = await graph.ainvoke(suspect_claim(), thread("r1"))
    assert out["decision"] == "refer"
    assert len(new_tasks(risk)) == 1


async def test_when_northgate_forgets_the_ledger_is_the_only_record() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("n1"))
    estimate = out["ledger"]["repair_estimate"]
    fed.partners.northgate = NorthgatePartner()  # restarted; its task store was memory
    northgate = (await connect_partners(fed.partners.apps()))["repair_estimate"]
    with as_run(thread("n2")), pytest.raises(TaskNotFoundError):
        await northgate.client.get_task(GetTaskRequest(id=estimate["task_id"]))
    assert estimate["state"] == "completed" and estimate["partner_release"] == "v7"
    assert estimate["wrote"] == ["estimate_notes", "repair_estimate"]  # notes: Ch 8
    assert estimate["seen_at"]


# Sent, and no answer -------------------------------------------------------------------

async def test_a_send_with_no_answer_is_recorded_and_never_resent() -> None:
    partners = A2APartners()
    risk = Outage(partners.risk.app)
    fed = Federation(partners)
    graph = await fed.graph({SCREENING: risk.http("http://risk.meridian.test/")})
    risk.down = True
    with pytest.raises(PartnerUnavailable, match="outcome unknown"):
        await graph.ainvoke(clean_claim(), thread("u9"))
    assert (await graph.aget_state(thread("u9"))).values["ledger"][SCREENING][
        "state"] == "unknown"
    risk.down = False
    with pytest.raises(PartnerUnavailable, match="outcome unknown"):
        await graph.ainvoke(clean_claim(), thread("u9"))  # still not resent
    assert partners.risk.executor.received == []


# Two writers, one entry ---------------------------------------------------------------

async def test_a_write_from_outside_a_running_graph_is_overwritten() -> None:
    gate = asyncio.Event()

    async def slow_write(state: ClaimState) -> ClaimState:
        read = state["ledger"][SCREENING]
        await gate.wait()
        return {"ledger": {SCREENING: observe(read, task(TaskState.TASK_STATE_WORKING),
                                              "2026.12")}}

    graph = StateGraph(ClaimState)
    graph.add_node("slow_write", slow_write)
    graph.add_edge(START, "slow_write")
    graph.add_edge("slow_write", END)
    app = graph.compile(checkpointer=InMemorySaver())
    first = entry()
    await app.aupdate_state(thread("w1"), {"ledger": {SCREENING: first}})
    run = asyncio.create_task(app.ainvoke(None, thread("w1")))
    await asyncio.sleep(0.05)  # the run has read v1 and is waiting
    update = observe(first, task(TaskState.TASK_STATE_COMPLETED), "2026.12")
    await app.aupdate_state(thread("w1"), {"ledger": {SCREENING: update}})  # accepted
    gate.set()
    await run
    ledger = (await app.aget_state(thread("w1"))).values["ledger"]
    assert ledger[SCREENING]["state"] == "working"  # the completed write is gone


async def test_a_status_update_read_before_the_resume_is_refused() -> None:
    risk = slow_risk()
    fed = Federation(A2APartners(risk=risk), seams=RETURNS_EARLY)
    graph = await fed.graph()
    with pytest.raises(TaskUnfinished):
        await graph.ainvoke(suspect_claim(), thread("s1"))
    read = (await graph.aget_state(thread("s1"))).values["ledger"][SCREENING]
    late = observe(read, task(TaskState.TASK_STATE_WORKING, read["task_id"],
                              read["context_id"]), risk.release)  # queued meanwhile
    await risk.finish_lookups()
    await graph.ainvoke(suspect_claim(), thread("s1"))  # the resume records completed
    with pytest.raises(StaleLedgerWrite):
        await graph.aupdate_state(thread("s1"), {"ledger": {SCREENING: late}})
    now = (await graph.aget_state(thread("s1"))).values["ledger"][SCREENING]
    assert now["state"] == "completed"


# The contextId across a claim's life --------------------------------------------------

async def test_an_amended_claim_is_a_new_task_in_the_same_context() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("c1"))
    first = out["ledger"][SCREENING]
    node = RemoteAgentNode(fed.seams[SCREENING], fed.connections[SCREENING])
    amended: ClaimState = {**clean_claim(), "amount": 3_400.0,
                           "ledger": {SCREENING: follow_on(first)}}
    with as_run(thread("c2")):  # a direct call, outside a graph, under a session
        update = await node(amended)
    second = update["ledger"][SCREENING]
    assert second["context_id"] == first["context_id"]
    assert second["task_id"] != first["task_id"]
    assert second["earlier"] == [first["task_id"]]
    assert merge_ledger(out["ledger"], {SCREENING: follow_on(first)})
    started = new_tasks(fed.partners.risk)
    assert len(started) == 2  # two tasks, one context
    assert {r["context_id"] for r in started} == {first["context_id"]}
