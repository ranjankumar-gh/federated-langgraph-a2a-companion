"""Claims intake exposed as an A2A agent: the mapping seen from the server side.

The same GraphExecutor that serves Risk serves the Claims graph. A caller's contextId
becomes a Claims thread_id; a Claims interrupt (Risk's police-report question, passed up
by RemoteAgentNode) becomes input-required to that caller; the decision is the artifact.

The lifespan opens the partner connections when the app starts and closes their HTTP
clients when it stops, so a cold start fetches every partner card once, not per claim.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from a2a.types import AgentCard
from langgraph.checkpoint.memory import InMemorySaver
from starlette.applications import Starlette
from starlette.types import ASGIApp

from meridian.federated import build_a2a_graph, connect_partners
from seamkit.remote import PartnerConnection
from seamkit.server import GraphExecutor, a2a_app, agent_card

CLAIMS_URL = "http://claims.meridian.test/"


def claim_artifact(out: dict[str, Any]) -> dict[str, Any]:
    keep = ("claim_id", "decision", "payment_ref")
    return {k: out[k] for k in keep if k in out}


@dataclass
class ClaimsAgent:
    card: AgentCard
    executor: GraphExecutor
    app: Starlette


def claims_agent(connections: Mapping[str, PartnerConnection]) -> ClaimsAgent:
    graph = build_a2a_graph(connections, InMemorySaver())
    executor = GraphExecutor(graph, "claim_decision", to_input=dict,
                             to_artifact=claim_artifact)
    card = agent_card("Meridian claims intake", CLAIMS_URL, "2026.10",
                      {"settle_claim": "Screen, estimate, and pay or refer one claim"})
    return ClaimsAgent(card, executor, a2a_app(card, executor))


def claims_app(partner_apps: Mapping[str, tuple[str, ASGIApp]]) -> Starlette:
    """The deployable shape: partners connect in the lifespan, once per process."""
    holder: dict[str, ClaimsAgent] = {}

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        connections = await connect_partners(partner_apps)
        holder["agent"] = claims_agent(connections)
        app.mount("/", holder["agent"].app)
        try:
            yield
        finally:
            for conn in connections.values():
                await conn.client.close()

    return Starlette(lifespan=lifespan)
