"""Test plumbing for the A2A wire: in-process transports and a partner outage."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx
from langchain_core.runnables import RunnableConfig
from starlette.types import ASGIApp

from meridian.federated import A2APartners, build_a2a_graph, connect_partners
from meridian.inprocess import ClaimGraph
from seamkit.remote import CrossingRecord, PartnerConnection


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


@dataclass
class Outage(httpx.AsyncBaseTransport):
    """An in-process transport that can be switched off after the card was fetched."""

    app: ASGIApp
    down: bool = False

    def __post_init__(self) -> None:
        self.inner = httpx.ASGITransport(app=self.app)

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
