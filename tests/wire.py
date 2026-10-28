"""Test plumbing for the A2A wire: in-process transports and a partner outage."""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.base import BaseCheckpointSaver
from starlette.types import ASGIApp

from meridian.federated import SEAMS, A2APartners, build_a2a_graph, connect_partners
from meridian.identity import ADJUSTER_SESSION, CLAIMS_ISSUER, MeridianIssuer
from meridian.inprocess import ClaimGraph, Node
from meridian.review import Reader, keyword_reader
from meridian.trust import MERIDIAN_TRUST
from meridian.risk_server import RiskPartner
from seamkit.delegation import as_peer
from seamkit.push import PushTo
from seamkit.remote import CrossingRecord, PartnerConnection, SeamSpec
from seamkit.trust import PartnerTrust


def thread(tid: str) -> RunnableConfig:
    """A Claims run, started by an adjuster whose portal session it carries."""
    return {"configurable": {"thread_id": tid, "session": ADJUSTER_SESSION}}


@contextmanager
def as_run(config: RunnableConfig) -> Iterator[None]:
    """Make a direct client call outside a graph see a run's config, as a node does."""
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


@dataclass
class Outage(httpx.AsyncBaseTransport):
    """An in-process transport that can be switched off after the card was fetched."""

    app: ASGIApp
    down: bool = False
    workload: str = "claims"

    def __post_init__(self) -> None:
        self.inner = httpx.ASGITransport(app=as_peer(self.app, self.workload))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        return await self.inner.handle_async_request(request)

    def http(self, url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self, base_url=url)


@dataclass
class Federation:
    partners: A2APartners = field(default_factory=A2APartners)
    records: list[CrossingRecord] = field(default_factory=list)
    connections: dict[str, PartnerConnection] = field(default_factory=dict)
    seams: Mapping[str, SeamSpec] = field(default_factory=lambda: SEAMS)
    issuer: MeridianIssuer = field(default_factory=lambda: CLAIMS_ISSUER)
    # Chapter 16: the card trust policy; Claims' CI passes one that pins a stub's key
    trust: Mapping[str, PartnerTrust] = field(default_factory=lambda: MERIDIAN_TRUST)

    async def graph(self, http: dict[str, httpx.AsyncClient] | None = None,
                    checkpointer: BaseCheckpointSaver[Any] | None = None,
                    apps: Mapping[str, tuple[str, ASGIApp]] | None = None,
                    review: Node | None = None,
                    reader: Reader = keyword_reader,
                    bells: Mapping[str, PushTo] | None = None,
                    ) -> ClaimGraph:
        """`apps` replaces a seam's deployment, such as one a registry lookup chose.
        `review` and `reader` replace Claims' review node, or only its model.
        `bells` sends a seam's new tasks with a push config (Chapter 11)."""
        deployed = {**self.partners.apps(), **(apps or {})}
        self.connections = await connect_partners(deployed, http, issuer=self.issuer,
                                                  trust=self.trust)
        return build_a2a_graph(self.connections, checkpointer,
                               log=self.records.append, seams=self.seams,
                               issuer=self.issuer, review=review, reader=reader,
                               bells=bells)


def slow_risk() -> RiskPartner:
    """Risk's 2026.12 release: a records lookup that answers when the test says so."""
    return RiskPartner(release="2026.12", lookups=asyncio.Event())


# Chapter 9's incident: the screening seam returns as soon as Risk has the task.
RETURNS_EARLY = {**SEAMS, "fraud_screening": replace(SEAMS["fraud_screening"],
                                                     return_immediately=True)}


def new_tasks(risk: RiskPartner) -> list[dict[str, Any]]:
    """Risk's own log of tasks it was asked to start, as opposed to answers it got."""
    return [r for r in risk.executor.received if not r["resumed"]]
