"""Meridian federated over A2A: Chapter 1's graph with three RemoteAgentNodes.

build_graph and claims_intake are Chapter 1's, unchanged except for the removed
fraud_score default. Only the workers mapping differs: each worker is now a node that
sends one A2A task to a partner's server.

Since Chapter 4, SEAMS is derived from the expectations files in meridian/expectations/,
and connecting to a partner checks its published terms against them.

Since Chapter 5, every partner's card is decided by Meridian's card trust policy
(meridian/trust.py) before its contract is checked.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from starlette.types import ASGIApp

from meridian.data_classes import MERIDIAN_POLICY
from meridian.finance_server import FINANCE_URL, FinancePartner
from meridian.inprocess import ClaimGraph, build_graph
from meridian.risk_server import RISK_URL, RiskPartner
from meridian.trust import MERIDIAN_TRUST
from northgate.server import NORTHGATE_URL, NorthgatePartner
from seamkit.contract import Expectations, load_expectations, seam_spec
from seamkit.inventory import SeamInventory
from seamkit.trust import PartnerTrust
from seamkit.remote import (
    CrossingRecord,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    connect,
    log_crossing,
)

EXPECTED = Path(__file__).parent / "expectations"

# Claims' half of each seam's contract, in the router's order: screen, estimate, pay.
EXPECTATIONS: dict[str, Expectations] = {
    name: load_expectations(EXPECTED / f"{name}.toml")
    for name in ("fraud_screening", "repair_estimate", "payments")}

# Chapter 3's seam specs, derived from the expectations rather than typed here.
SEAMS: dict[str, SeamSpec] = {name: seam_spec(e) for name, e in EXPECTATIONS.items()}

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
        trust: Mapping[str, PartnerTrust] = MERIDIAN_TRUST,
) -> dict[str, PartnerConnection]:
    """Fetch each card, decide whether to trust it, check its terms, and build a client.

    `http` overrides a transport. A seam whose card is untrusted, or whose terms are
    breached, still connects; its node refuses to send. The policy is looked up by the
    owner Claims' expectations name, never by anything the card says.
    """
    out: dict[str, PartnerConnection] = {}
    for seam, (url, app) in apps.items():
        owner = EXPECTATIONS[seam].owner
        client = (http or {}).get(seam) or in_process_http(url, app, WORKLOAD_TOKENS[owner])
        out[seam] = await connect(url, client, EXPECTATIONS[seam], MERIDIAN_POLICY,
                                  trust=trust[owner])
    return out


def with_trust(inventory: SeamInventory,
               connections: Mapping[str, PartnerConnection]) -> SeamInventory:
    """Put each seam's card decision first in its authority answer, never by hand."""
    for entry in inventory.entries:
        decision = (conn := connections.get(entry.callee)) and conn.trust
        if not decision:
            continue
        answer = (f"card trusted by {decision.pinned_by}" if decision.trusted
                  else f"card refused: {decision.refused}")
        if decision.inflated:
            answer += f" (unapproved: {', '.join(decision.inflated)})"
        held = entry.answers.get("authority")
        entry.answers["authority"] = answer + (f"; {held}" if held else "")
    return inventory


def with_contract(inventory: SeamInventory,
                  connections: Mapping[str, PartnerConnection]) -> SeamInventory:
    """Fill each seam's contract answer from the check run at connect, never by hand."""
    for entry in inventory.entries:
        conn = connections.get(entry.callee)
        if conn is None:
            continue
        held = f"terms {conn.terms[0].revision}" if conn.terms else "no terms"
        answer = (f"{held} breached ({len(conn.breaches)})" if conn.breaches
                  else f"{held} checked")
        later = sorted({(b.effective, b.revision) for b in conn.upcoming})
        entry.answers["contract"] = answer + "".join(
            f"; {rev} breaks it from {when}" for when, rev in later)
    return inventory


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


__all__ = ["EXPECTATIONS", "with_contract", "with_trust", "OWNERS_A2A", "SEAMS",
           "A2APartners",
           "CrossingRecord", "build_a2a_graph",
           "connect_partners", "in_process_http"]
