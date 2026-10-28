"""The opening incident of Chapter 3, kept so its test can reproduce it.

This node is deliberately wrong. It sends Risk a screening task and, when Risk asks
for a police report, calls interrupt() in the same node. LangGraph re-runs an
interrupted node from its first line on resume, so the send runs again: Risk receives a
second task in a fresh context, and the first one waits for an answer forever.
RemoteAgentNode is the fix.
"""

from collections.abc import Mapping
from typing import Any

from a2a.client import ClientCallContext
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import Role, SendMessageRequest, Task, TaskState
from langgraph.types import interrupt

from seamkit.remote import PartnerConnection, question_of

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
