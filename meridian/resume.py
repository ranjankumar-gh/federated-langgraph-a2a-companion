"""Claims' resume gate: a task's authority is re-checked before the node reuses it.

Chapter 9's RemoteAgentNode reads an open task again on every later entry, and resumes
one that paused. Each of those is an act point on Claims' side. The gate runs first:
it takes the chain id recorded with the seam's task (the entry's authority), never the
chain the run holds now, and asks Meridian's issuer about it.

A revoked chain id means the task ends: the gate asks the partner to cancel it, under
a root the issuer mints already revoked, and raises DelegationRevoked, which sends the
claim to an adjuster like every SeamError. A task in auth-required is renewed with a
follow-up on the same task, under this run's chain, only if that chain acts for the
same person and covers the same claim; otherwise the gate raises PartnerNeedsAuth.

RemoteAgentNode.__call__ is unchanged: the gate is a subclass that checks, then calls it.
Since Chapter 8 the gate's parent is FilteredNode, which adds the ingress filter's
egress check before a send and the provenance tag after an answer.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from a2a.client import A2AClientError, ClientCallContext
from a2a.helpers import new_data_message
from a2a.types import CancelTaskRequest, Role, SendMessageConfiguration
from a2a.types import SendMessageRequest
from a2a.utils.errors import A2AError

from meridian.identity import CLAIMS_ISSUER, MeridianIssuer
from seamkit.delegation import run_values
from seamkit.errors import DelegationRefused, DelegationRevoked, PartnerNeedsAuth
from seamkit.ledger import TERMINAL
from seamkit.remote import FilteredNode

logger = logging.getLogger("meridian.resume")


@dataclass
class ResumeGate(FilteredNode):
    """An A2A call node with Claims' resume act point in front of it."""

    issuer: MeridianIssuer = field(default_factory=lambda: CLAIMS_ISSUER)

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        entry = state.get("ledger", {}).get(self.seam.name)
        if entry and entry["task_id"] and entry["state"] not in TERMINAL:
            await self.recheck(entry["task_id"], entry["authority"], entry["state"],
                               str(state.get("claim_id", "")))
        return await super().__call__(state)

    async def recheck(self, task_id: str, chain: str, state: str, claim: str) -> None:
        """Before the node reads or resumes a task: does its authority still hold?"""
        if why := self.issuer.revoked_id(chain):  # the chain it was sent under
            await self._cancel(task_id, self.issuer.subject_of(chain))
            raise DelegationRevoked(self.seam.name, why, task_id)
        if state == "auth-required":
            held = self.issuer.run_chain()[0]  # this run's root, re-exchanged if expired
            if held.subject != self.issuer.subject_of(chain) or claim not in held.claims:
                raise PartnerNeedsAuth(self.seam.name, "a renewal needs the same "
                                       "person and the same claim", task_id)
            await self._renew(task_id)  # the partner re-checks the new chain

    async def _cancel(self, task_id: str, subject: str) -> None:
        """Ask the partner to cancel. Success is not guaranteed, so nothing waits on it."""
        context = ClientCallContext(timeout=self.seam.timeout)
        try:
            with run_values(cancel_for=subject):  # minted under a cancel root
                await self.partner.client.cancel_task(CancelTaskRequest(id=task_id),
                                                      context=context)
            outcome = "canceled"
        except (A2AClientError, A2AError, DelegationRefused) as exc:
            outcome = type(exc).__name__  # TaskNotCancelableError: already final
            logger.warning("cancel of %s on %s: %s", task_id, self.seam.name, exc)
        self.log(self._record("cancel", ())(task_id=task_id, state=outcome))

    async def _renew(self, task_id: str) -> None:
        """A follow-up on the same task, sent under the run's chain, and waited for."""
        task = await self._get(task_id)
        message = new_data_message({}, role=Role.ROLE_USER, task_id=task_id,
                                   context_id=task.context_id)
        message.metadata.update({"skill": self.seam.skill})
        request = SendMessageRequest(message=message, configuration=(
            SendMessageConfiguration(return_immediately=False)))
        async for _ in self.partner.client.send_message(
                request, context=ClientCallContext(timeout=self.seam.timeout)):
            pass
        self.log(self._record("renew", ())(task_id=task_id))
