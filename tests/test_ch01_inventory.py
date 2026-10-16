"""The seam inventory finds Meridian's three seams, and the one nobody listed."""

from langchain_core.runnables import RunnableConfig

from meridian.federated_naive import OWNERS, build_federated_graph
from meridian.inprocess import (
    IN_PROCESS_WORKERS,
    ClaimState,
    build_graph,
    build_inprocess_graph,
)
from meridian.samples import clean_claim
from seamkit.inventory import PROPERTIES, SeamEntry, SeamInventory, inventory_from_graph


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


def test_meridian_has_three_seams_each_with_five_open_properties() -> None:
    inventory = inventory_from_graph(build_federated_graph(), OWNERS)
    pairs = [(e.caller, e.callee_owner) for e in inventory.entries]
    assert sorted(pairs) == [("claims_intake", "Finance"),
                             ("claims_intake", "Northgate"),
                             ("claims_intake", "Risk")]
    assert all(e.open_properties() == list(PROPERTIES) for e in inventory.entries)


def test_the_inventory_reads_the_graph_not_the_deployment() -> None:
    # Same edges, same owners: the in-process graph has the same three seams waiting.
    inventory = inventory_from_graph(build_inprocess_graph(), OWNERS)
    assert len(inventory.entries) == 3


def test_a_node_added_later_gets_a_row() -> None:
    def payment_hold(state: ClaimState) -> ClaimState:
        return {"completed": ["payment_hold"]}

    workers = dict(IN_PROCESS_WORKERS)
    payments = workers.pop("payments")
    graph = build_graph({**workers, "payment_hold": payment_hold, "payments": payments})
    owners = {**OWNERS, "payment_hold": "Finance"}
    callees = {e.callee for e in inventory_from_graph(graph, owners).entries}
    assert "payment_hold" in callees
    assert "payment_hold" in graph.invoke(clean_claim(), thread("h1"))["completed"]


def test_same_owner_edges_are_not_seams() -> None:
    owners = {name: "Claims" for name in OWNERS}
    assert inventory_from_graph(build_inprocess_graph(), owners).entries == []


def test_render_marks_open_properties_with_the_chapter_that_answers_them() -> None:
    entry = SeamEntry("claims_intake", "repair_estimate", "Claims", "Northgate",
                      answers={"contract": "seam contract v1"}, paged="claims-oncall")
    table = SeamInventory([entry]).render()
    assert "seam contract v1" in table
    assert "open (Ch 5-8)" in table
    assert "claims-oncall" in table
