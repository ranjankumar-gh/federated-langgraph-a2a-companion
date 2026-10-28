"""What the incident's seam costs: policy_lookup as a subgraph against the same graph
behind its own A2A server. Measured on the in-process ASGI wire, so the latency is a
floor: no network, no TLS, no queue in front of the server."""

import statistics
import time

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from meridian.federated import in_process_http
from meridian.inprocess import ClaimGraph
from meridian.policy import build_policy_inprocess_graph
from meridian.policy_server import POLICY_URL, PolicyServer, build_incident_graph
from meridian.samples import clean_claim, suspect_claim
from seamkit.remote import CrossingRecord, connect


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


async def incident(bodies: list[int] | None = None,
                   ) -> tuple[ClaimGraph, PolicyServer, list[CrossingRecord]]:
    server = PolicyServer()
    http = in_process_http(POLICY_URL, server.app, "claims")

    async def body_size(request: httpx.Request) -> None:
        if bodies is not None and request.method == "POST":
            bodies.append(len(request.content))

    http.event_hooks["request"].append(body_size)
    records: list[CrossingRecord] = []
    graph = build_incident_graph(await connect(POLICY_URL, http), records.append)
    return graph, server, records


@pytest.mark.anyio
async def test_both_builds_decide_the_same() -> None:
    remote, _, _ = await incident()
    subgraph = build_policy_inprocess_graph()
    for claim in (clean_claim(), suspect_claim()):
        a = await remote.ainvoke(claim, thread(claim["claim_id"]))
        b = await subgraph.ainvoke(claim, thread(claim["claim_id"]))
        assert a["covered_parts"] == b["covered_parts"]
        assert a["decision"] == b["decision"]
    assert b["covered_parts"] == ["door"]  # CLM-1002: revision 4 excludes the frame


@pytest.mark.anyio
async def test_the_re_hydration_tax_crosses_on_every_task() -> None:
    bodies: list[int] = []
    remote, server, records = await incident(bodies)
    await remote.ainvoke(clean_claim(), thread("t1"))
    await remote.ainvoke(clean_claim(), thread("t2"))
    # Two claims, two tasks: the same two fields re-sent each time, never read in place.
    assert [r.sent for r in records if r.call == "send"] == [
        ("claim_id", "damaged_parts")] * 2
    assert bodies == [310, 310]  # the whole JSON-RPC request, for two fields
    assert len(server.executor.received) == 2


@pytest.mark.anyio
async def test_the_claims_checkpoint_loses_the_policy_it_read() -> None:
    remote, server, _ = await incident()
    subgraph = build_policy_inprocess_graph()
    config = thread("CLM-1002")
    a = await remote.ainvoke(suspect_claim(), config)
    b = await subgraph.ainvoke(suspect_claim(), config)
    assert set(b) - set(a) == {"policy"}  # which revision excluded the frame
    assert b["policy"]["revision"] == 4
    # The subgraph's two steps are in Claims' own checkpointer, under their namespace.
    assert isinstance(subgraph.checkpointer, BaseCheckpointSaver)
    assert isinstance(remote.checkpointer, BaseCheckpointSaver)
    in_ns = [c for c in subgraph.checkpointer.list(config)
             if c.config["configurable"].get("checkpoint_ns")]
    assert len(in_ns) == 4
    # Behind A2A they are in a checkpointer only the policy server holds.
    assert not [c for c in remote.checkpointer.list(config)
                if c.config["configurable"].get("checkpoint_ns")]
    assert len(list(server.checkpointer.list(None))) == 4


@pytest.mark.anyio
async def test_the_seam_adds_a_round_trip_per_claim() -> None:
    remote, _, _ = await incident()
    subgraph = build_policy_inprocess_graph()

    async def median_ms(graph: ClaimGraph, tag: str) -> float:
        times = []
        for i in range(25):
            start = time.perf_counter()
            await graph.ainvoke(clean_claim(), thread(f"{tag}{i}"))
            times.append(time.perf_counter() - start)
        return statistics.median(times) * 1000

    seam, local = await median_ms(remote, "a"), await median_ms(subgraph, "s")
    print(f"\npolicy_lookup per claim: A2A {seam:.1f} ms, subgraph {local:.1f} ms")
    assert seam > local
