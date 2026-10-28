"""Test plumbing for the A2A wire: in-process transports and a partner outage."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from starlette.types import ASGIApp

from meridian.federated import A2APartners, build_a2a_graph, connect_partners
from meridian.identity import ADJUSTER_SESSION
from meridian.inprocess import ClaimGraph
from seamkit.delegation import as_peer
from seamkit.remote import CrossingRecord, PartnerConnection


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

    async def graph(self, http: dict[str, httpx.AsyncClient] | None = None,
                    apps: Mapping[str, tuple[str, ASGIApp]] | None = None,
                    ) -> ClaimGraph:
        """`apps` replaces a seam's deployment, such as one a registry lookup chose."""
        deployed = {**self.partners.apps(), **(apps or {})}
        self.connections = await connect_partners(deployed, http)
        return build_a2a_graph(self.connections, log=self.records.append)
