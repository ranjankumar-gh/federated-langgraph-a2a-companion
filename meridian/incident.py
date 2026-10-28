"""The opening incidents of Chapters 3 and 6, kept so their tests can reproduce them.

This node is deliberately wrong. It sends Risk a screening task and, when Risk asks
for a police report, calls interrupt() in the same node. LangGraph re-runs an
interrupted node from its first line on resume, so the send runs again: Risk receives a
second task in a fresh context, and the first one waits for an answer forever.
RemoteAgentNode is the fix.

Chapter 6's incident is the authority on the wire before delegation chains: per-partner
workload tokens that say who is calling and nothing about on whose behalf, a Northgate
that accepts any valid Meridian service token from any Meridian workload, and a Risk
tool client that forwards the Authorization header of the request that started its run.
The delegation chain (seamkit.delegation) is the fix.
"""

from collections.abc import Generator, Mapping
from typing import Any

import httpx
from a2a.client import ClientCallContext
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import Role, SendMessageRequest, Task, TaskState
from langgraph.types import interrupt

from seamkit.delegation import as_peer, run_value
from seamkit.remote import PartnerConnection, connect, question_of

RISK_FIELDS = ("claim_id", "amount", "days_to_report", "prior_claims")


async def _send(partner: PartnerConnection, data: dict[str, Any],
                context_id: str | None = None, task_id: str | None = None) -> Task:
    message = new_data_message(data, role=Role.ROLE_USER, context_id=context_id,
                               task_id=task_id)
    message.metadata.update({"skill": "screen_fraud"})
    task = Task()
    async for event in partner.client.send_message(
            SendMessageRequest(message=message), context=ClientCallContext(timeout=10)):
        task = event.task
    return task


def send_then_ask(partner: PartnerConnection) -> Any:
    async def fraud_screening(state: Mapping[str, Any]) -> dict[str, Any]:
        task = await _send(partner, {k: state[k] for k in RISK_FIELDS})  # re-runs
        if task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
            reply = interrupt(question_of(task))
            task = await _send(partner, reply, task.context_id, task.id)
        answer = get_data_parts(task.artifacts[0].parts)[0]
        return {"fraud_score": answer["fraud_score"], "completed": ["fraud_screening"]}

    return fraud_screening


# Chapter 6 ----------------------------------------------------------------------------

# Chapter 3's per-partner workload tokens: each says the caller is the Claims service,
# and authorises nothing on anyone's behalf.
WORKLOAD_TOKENS = {"Risk": "claims-workload-risk", "Northgate": "claims-workload-ng",
                   "Finance": "claims-workload-fin"}
OWNER_OF_SEAM = {"fraud_screening": "Risk", "repair_estimate": "Northgate",
                 "payments": "Finance"}


def introspect(token: str) -> str:
    """Meridian's answer to "is this a valid service token?": its owner, or ""."""
    return "Claims" if token in WORKLOAD_TOKENS.values() else ""


def tokened_http(url: str, app: Any, token: str, workload: str) -> httpx.AsyncClient:
    """Chapter 3's client: a Meridian workload, presenting a bearer workload token."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=as_peer(app, workload)),
                             base_url=url, headers={"Authorization": f"Bearer {token}"})


async def connect_with_tokens(
        apps: Mapping[str, tuple[str, Any]]) -> dict[str, PartnerConnection]:
    """Claims' partner connections as Chapter 3 built them, before delegation."""
    out: dict[str, PartnerConnection] = {}
    for seam, (url, app) in apps.items():
        token = WORKLOAD_TOKENS[OWNER_OF_SEAM[seam]]
        out[seam] = await connect(url, tokened_http(url, app, token, "claims"))
    return out


class ForwardingAuth(httpx.Auth):
    """The incident: the client sends on whatever credential its run arrived with."""

    def auth_flow(self, request: httpx.Request,
                  ) -> Generator[httpx.Request, httpx.Response, None]:
        inbound = run_value("authorization")
        if inbound:
            request.headers["Authorization"] = inbound  # the confused deputy, one line
        yield request
