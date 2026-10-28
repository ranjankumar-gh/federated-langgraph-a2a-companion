"""The Claims team's policy lookup: a small LangGraph graph Claims owns end to end.

Chapter 2's question is where this graph should run. As a subgraph it reads the claim
from the shared ClaimState and its steps land in the Claims checkpoint. Chapter 2's
opening incident puts the same graph behind its own A2A server (meridian.incident), and
the Federation Test refuses that seam because one owner sits on both sides of it.

The default Meridian graph does not include it, so no other chapter's numbers move:
POLICY_WORKERS is a variant that runs the cover check first.
"""

from typing import Any, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meridian.inprocess import IN_PROCESS_WORKERS, ClaimGraph, Node, build_graph

# The policy administration system's answer for each sample claim: the policy wording
# as it stood when the cover check read it. Revision 4 of POL-2002 added an endorsement
# that excludes frame damage.
POLICIES: dict[str, dict[str, Any]] = {
    "CLM-1001": {"policy_id": "POL-1001", "revision": 2,
                 "covers": ["bumper", "door", "frame"], "endorsements": []},
    "CLM-1002": {"policy_id": "POL-2002", "revision": 4,
                 "covers": ["bumper", "door"], "endorsements": ["E-12 frame excluded"]},
}


class PolicyInput(TypedDict):
    claim_id: str
    damaged_parts: list[str]


class PolicyOutput(TypedDict, total=False):
    policy: dict[str, Any]
    covered_parts: list[str]
    completed: list[str]


class PolicyState(PolicyInput, PolicyOutput):
    """Shares its keys with ClaimState. `completed` has no reducer here: the subgraph
    starts from the two input keys, so it reports only its own step to the parent."""


def fetch_policy(state: PolicyState) -> PolicyOutput:
    return {"policy": POLICIES[state["claim_id"]]}


def check_cover(state: PolicyState) -> PolicyOutput:
    covers = state["policy"]["covers"]
    return {"covered_parts": [p for p in state["damaged_parts"] if p in covers],
            "completed": ["policy_lookup"]}


def build_policy_graph(checkpointer: BaseCheckpointSaver[Any] | None = None,
                       ) -> CompiledStateGraph[PolicyState, None, PolicyInput,
                                               PolicyOutput]:
    """Two steps worth checkpointing: which policy was read, then what it covers.

    Compiled without a checkpointer it is a subgraph that inherits its parent's; the
    incident's server compiles it with one of its own.
    """
    graph = StateGraph(PolicyState, input_schema=PolicyInput,
                       output_schema=PolicyOutput)
    graph.add_node("fetch_policy", fetch_policy)
    graph.add_node("check_cover", check_cover)
    graph.add_edge(START, "fetch_policy")
    graph.add_edge("fetch_policy", "check_cover")
    graph.add_edge("check_cover", END)
    return graph.compile(checkpointer=checkpointer)


POLICY_WORKERS: dict[str, Node] = {
    # A compiled graph added as a node is a subgraph: shared keys, the parent's
    # checkpointer. The cast is for the Node protocol, which a Pregel does not declare.
    "policy_lookup": cast(Node, build_policy_graph()),
    **IN_PROCESS_WORKERS,
}


def build_policy_inprocess_graph() -> ClaimGraph:
    return build_graph(POLICY_WORKERS)
