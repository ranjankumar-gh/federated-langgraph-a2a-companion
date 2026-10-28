"""Chapter 2's opening incident: Claims' own policy lookup behind its own A2A server.

Deliberately wrong. The graph is meridian.policy's, unchanged; only where it runs moved.
It gets its own card, its own release, its own task store and its own checkpointer, and
the Claims graph reaches it through a RemoteAgentNode like any partner. Nobody but
Claims calls it. The Federation Test refuses this seam, and Chapter 2's tests measure what it
costs against the same graph run as a subgraph.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver
from starlette.types import ASGIApp

from meridian.inprocess import IN_PROCESS_WORKERS, ClaimGraph, Node, build_graph
from meridian.policy import build_policy_graph
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    log_crossing,
)
from seamkit.server import GraphExecutor, a2a_app, agent_card

POLICY_URL = "http://policy.claims.meridian.test/"

# The fields this seam re-sends on every task: what the subgraph read in place.
POLICY_SEAM = SeamSpec("policy_lookup", "Claims", "lookup_policy",
                       sends=("claim_id", "damaged_parts"),
                       receives={"covered_parts": "covered_parts"})


@dataclass
class PolicyServer:
    """The policy lookup as Claims deployed it in the incident: a second runtime."""

    release: str = "policy-2026.10"
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)

    def __post_init__(self) -> None:
        self.graph = build_policy_graph(self.checkpointer)
        self.executor = GraphExecutor(
            self.graph, "policy_lookup",
            to_input=lambda data: data,
            to_artifact=lambda out: {"covered_parts": out["covered_parts"]})
        self.card = agent_card("Claims policy lookup", POLICY_URL, self.release,
                               {"lookup_policy": "Check damaged parts against cover"})
        self.app: ASGIApp = a2a_app(self.card, self.executor)


def incident_workers(policy: PartnerConnection,
                     log: Callable[[CrossingRecord], None] = log_crossing,
                     others: Mapping[str, Node] = IN_PROCESS_WORKERS,
                     ) -> dict[str, Node]:
    """The incident's workers: the policy lookup remote, everything else unchanged."""
    return {"policy_lookup": RemoteAgentNode(POLICY_SEAM, policy, log), **others}


def build_incident_graph(policy: PartnerConnection,
                         log: Callable[[CrossingRecord], None] = log_crossing,
                         ) -> ClaimGraph:
    return build_graph(incident_workers(policy, log))
