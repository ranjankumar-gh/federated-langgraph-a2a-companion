"""Meridian federated over A2A: Chapter 1's graph with three RemoteAgentNodes.

build_graph and claims_intake are Chapter 1's, unchanged except for the removed
fraud_score default. Only the workers mapping differs: each worker is now a node that
sends one A2A task to a partner's server.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from starlette.types import ASGIApp

from meridian.finance_server import FINANCE_URL, FinancePartner
from meridian.inprocess import ClaimGraph, build_graph
from meridian.risk_server import RISK_URL, RiskPartner
from northgate.server import NORTHGATE_URL, NorthgatePartner
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    connect,
    log_crossing,
)

SEAMS: dict[str, SeamSpec] = {
    "fraud_screening": SeamSpec(
        "fraud_screening", "Risk", "screen_fraud",
        sends=("claim_id", "amount", "days_to_report", "prior_claims"),
        receives={"fraud_score": "fraud_score"},
        asks_caller=True),  # Risk's police-report question goes to a Claims adjuster
    "repair_estimate": SeamSpec(
        "repair_estimate", "Northgate", "estimate_repair",
        sends=("claim_id", "damaged_parts"),
        receives={"estimate": "repair_estimate"}),
    "payments": SeamSpec(
        "payments", "Finance", "pay_claim",
        sends=("claim_id", "amount", "repair_estimate"),
        receives={"payment_ref": "payment_ref"}),
}

# The owner map the seam inventory reads, derived from the seams rather than retyped.
OWNERS_A2A = {"claims_intake": "Claims", **{s.name: s.owner for s in SEAMS.values()}}

# A per-partner workload token: it says the caller is the Claims service and authorises
# nothing on anyone's behalf. Chapter 6 replaces it with a delegation.
WORKLOAD_TOKENS = {"Risk": "claims-workload-risk", "Northgate": "claims-workload-ng",
                   "Finance": "claims-workload-fin"}


@dataclass
class A2APartners:
    """The three partner deployments, each an ASGI app with its own state."""

    risk: RiskPartner = field(default_factory=RiskPartner)
    northgate: NorthgatePartner = field(default_factory=NorthgatePartner)
    finance: FinancePartner = field(default_factory=FinancePartner)

    def apps(self) -> dict[str, tuple[str, ASGIApp]]:
        return {"fraud_screening": (RISK_URL, self.risk.app),
                "repair_estimate": (NORTHGATE_URL, self.northgate.app),
                "payments": (FINANCE_URL, self.finance.app)}


def in_process_http(url: str, app: ASGIApp, token: str) -> httpx.AsyncClient:
    """The real SDK client over an ASGI transport: the full wire, no sockets."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=url,
                             headers={"Authorization": f"Bearer {token}"})


async def connect_partners(
        apps: Mapping[str, tuple[str, ASGIApp]],
        http: Mapping[str, httpx.AsyncClient] | None = None,
) -> dict[str, PartnerConnection]:
    """Fetch each partner's card and build a client. `http` overrides a transport."""
    out: dict[str, PartnerConnection] = {}
    for seam, (url, app) in apps.items():
        token = WORKLOAD_TOKENS[SEAMS[seam].owner]
        client = (http or {}).get(seam) or in_process_http(url, app, token)
        out[seam] = await connect(url, client)
    return out


def a2a_workers(connections: Mapping[str, PartnerConnection],
                log: Callable[[CrossingRecord], None] = log_crossing,
                ) -> dict[str, RemoteAgentNode]:
    return {seam: RemoteAgentNode(SEAMS[seam], connections[seam], log)
            for seam in SEAMS}


def build_a2a_graph(connections: Mapping[str, PartnerConnection],
                    checkpointer: BaseCheckpointSaver[Any] | None = None,
                    log: Callable[[CrossingRecord], None] = log_crossing,
                    ) -> ClaimGraph:
    return build_graph(a2a_workers(connections, log), checkpointer)


__all__ = ["OWNERS_A2A", "SEAMS", "A2APartners", "CrossingRecord", "build_a2a_graph",
           "connect_partners", "in_process_http"]
