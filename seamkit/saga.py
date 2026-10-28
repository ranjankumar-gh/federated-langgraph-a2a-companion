"""The saga plan and the compensator: undoing a step that committed at another owner.

A caller can rewind its own checkpoint. It cannot rewind a partner's, so the only way to
undo a step that committed on the other side of a seam is another task: a request for
the compensating skill the partner declared, which the partner may honour, honour in
part, or refuse. The saga plan is that arrangement written down before the first claim
crosses: one row per forward step, naming the step's owner, its compensating skill, how
long Claims needs the compensation honoured, and who settles what it cannot undo.

`check_plan` holds the plan against the terms each partner publishes. The compensator
is a node in the claim's own thread. When the claim's decision is reversed it walks the
ledger newest first, one action per superstep, so every intent is checkpointed before
the call it records:

- a step whose task has not committed is cancelled, and read again;
- a step that committed, with a compensating skill, gets a compensation entry, and then
  the compensation is sent under a key derived from the forward key;
- a step with nothing to undo is recorded as such, and nothing is sent.

Every call the compensator makes for a step goes under one link (`UndoAuth`), narrowed
to the compensating skill and to the one task it undoes. Anything the compensation did
not undo becomes a finding for people. Nothing here retries a refusal.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from a2a.helpers import get_data_parts

from seamkit.contract import Breach
from seamkit.delegation import DelegationAuth, Grant, run_value, run_values
from seamkit.errors import DelegationRefused, PartnerFailed, SeamError
from seamkit.ingress import DETAIL_MAX_CHARS, quote
from seamkit.ledger import TERMINAL, LedgerEntry, compensation, observe, state_name
from seamkit.reconcile import Finding, log_finding, not_found
from seamkit.remote import PartnerConnection, RemoteAgentNode
from seamkit.cards import card_digest

HOUR = 3600
OPEN = frozenset({"submitted", "working", "input-required", "auth-required"})
FINDINGS = frozenset({"residue", "refused", "uncompensated", "lost"})


@dataclass(frozen=True)
class SagaStep:
    """One forward step, and what Claims and its owner agreed happens on an undo."""

    seam: str  # the forward step, by seam name
    skill: str
    owner: str
    compensated_by: str  # the owner's compensating skill; "" when nothing is sent
    window: int  # seconds after the step commits that Claims may still reverse it
    settles: str  # who settles what the compensation cannot undo, or why none is sent


@dataclass(frozen=True)
class SagaPlan:
    """The forward steps in claim order. Compensation runs them newest first."""

    steps: tuple[SagaStep, ...]

    def step(self, seam: str) -> SagaStep:
        return next(s for s in self.steps if s.seam == seam)


def check_plan(plan: SagaPlan,
               connections: Mapping[str, PartnerConnection]) -> list[Breach]:
    """Every row the terms in force do not honour: a promise missing, or too short."""
    breaches = []
    for step in plan.steps:
        conn = connections.get(step.seam)
        if not step.compensated_by or conn is None:
            continue
        terms = conn.terms[0] if conn.terms else None
        held = terms.skills.get(step.skill) if terms else None
        found = []
        if held is None or held.compensated_by != step.compensated_by:
            found.append(f"{step.skill} is not undone by {step.compensated_by}")
        elif (held.compensate_seconds or 0) < step.window:
            found.append(f"{step.compensated_by} is honoured "
                         f"{held.compensate_seconds} s, and the plan needs {step.window} s")
        revision, effective = (terms.revision, terms.effective) if terms else (
            "none", date.min)
        breaches += [Breach(step.seam, revision, effective, d) for d in found]
    return breaches


def held_to_plan(plan: SagaPlan, connections: Mapping[str, PartnerConnection],
                 ) -> dict[str, PartnerConnection]:
    """The connections, each forward seam the plan does not cover refusing to send.

    A step that commits at a partner, with no undo the partner has promised, is a
    contract breach before it is a compensation problem: the seam is held, as Chapter 4
    holds a seam whose terms break what it reads.
    """
    out = dict(connections)
    for breach in check_plan(plan, connections):
        held = out[breach.seam]
        out[breach.seam] = replace(held, breaches=(*held.breaches, breach))
    return out


class UndoAuth(DelegationAuth):
    """A compensation's link: the undo grant, narrowed to the one task it undoes.

    The parent is a root Meridian's issuer mints for Claims itself, never for the
    person whose chain sent the forward task: by the time a step is undone that chain
    may have expired, or been revoked.
    """

    def grant_now(self) -> Grant:
        task = run_value("undoes")
        if not task:
            raise DelegationRefused(self.audience, "a compensation names no task")
        return replace(self.grant, tasks=frozenset({str(task)}))


def undo_name(key: str) -> str:
    """The ledger key of the compensation of the entry under `key`."""
    return f"{key}.undo"


@dataclass
class Compensator:
    """The saga's undo, as a node in the claim's own thread."""

    plan: SagaPlan
    undo: Mapping[str, RemoteAgentNode]  # forward seam -> the node that sends its undo
    authority: Callable[[], str]  # the compensation link's chain, by id
    log: Callable[[Finding], None] = log_finding
    # forward seam -> its own node: a step with nothing to undo is still cancelled,
    # under the run's chain, if its task has not finished
    forward: Mapping[str, RemoteAgentNode] = field(default_factory=dict)

    def todo(self, state: Mapping[str, Any]) -> list[tuple[str, SagaStep]]:
        """(ledger key, plan row) still to settle, newest first.

        A reversed claim undoes every step it took. An orphan adopted from the
        reconciliation job is undone whatever the claim's decision is.
        """
        ledger, undone = state.get("ledger", {}), state.get("undone", {})
        rows = [(s.seam, s) for s in reversed(self.plan.steps)
                if s.seam in ledger and state.get("decision") == "refer"]
        rows += [(key, self.plan.step(seam))
                 for key, seam in state.get("orphans", {}).items()]
        return [(key, step) for key, step in rows if key not in undone]

    def pending(self, state: Mapping[str, Any]) -> bool:
        return bool(self.todo(state))

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        for key, step in self.todo(state):  # one action per superstep, newest first
            forward: LedgerEntry = state["ledger"][key]
            if not forward["task_id"]:  # a send that may have left: Chapter 12's job
                return self.settle(step, key, "uncompensated", forward,
                                   detail="no task id: the reconciliation job looks "
                                          "its key up")
            with run_values(compensating=state["claim_id"], undoes=forward["task_id"]):
                return await self.undo_one(key, step, forward, state)
        return {}

    async def undo_one(self, key: str, step: SagaStep, forward: LedgerEntry,
                       state: Mapping[str, Any]) -> dict[str, Any]:
        if step.seam not in self.undo:  # nothing to undo: only an open task is ended
            return await self.end_open(key, step, forward)
        node = self.node(key, step)
        undo = state["ledger"].get(node.seam.name)
        if undo is None:  # nothing sent yet: the forward task, read fresh, decides
            try:
                task = await node.read(forward["task_id"])
            except PartnerFailed as exc:
                if not_found(exc):  # the partner lost it: nothing can name it now
                    return self.settle(step, key, "lost", forward, detail=exc.detail)
                raise
            now = state_name(task.status.state)
            if now in OPEN:  # not committed: cancel it, as Chapter 10's horizon does
                await node.cancel(task.id)
                task = await node.read(task.id)
                seen = {"ledger": {key: observe(forward, task, node.partner.release)}}
                if state_name(task.status.state) == "completed":
                    return seen  # it committed meanwhile: the next pass compensates
                return {**seen, **self.settle(step, key, "cancelled", forward)}
            if now != "completed":  # failed, rejected, canceled: nothing committed
                return self.settle(step, key, "not committed", forward)
            if not step.compensated_by:  # nothing to undo, or a cost accepted
                return self.settle(step, key, step.settles, forward)
            return {"ledger": {node.seam.name: compensation(
                forward, node.seam.name, step.compensated_by,
                card_digest(node.partner.card), self.authority(),
                node.partner.release)}}  # the intent, checkpointed before the send
        if undo["state"] in TERMINAL:  # answered on an earlier pass; only record it
            verdict = {"completed": "compensated", "rejected": "refused"}.get(
                undo["state"], "uncompensated")
            return self.settle(step, key, verdict, forward, undo)
        return await self.send(key, step, forward, undo, node, state)

    async def send(self, key: str, step: SagaStep, forward: LedgerEntry,
                   undo: LedgerEntry, node: RemoteAgentNode,
                   state: Mapping[str, Any]) -> dict[str, Any]:
        """Send the compensation under its entry's key, and settle on the answer."""
        asked = {"claim_id": state["claim_id"], "task": forward["task_id"],
                 "ledger": {node.seam.name: undo}}
        try:
            update = await node(asked)  # Chapter 12's retry, under the derived key
        except SeamError as exc:  # no answer, or refused before it left Claims
            return self.settle(step, key, "uncompensated", forward, undo,
                               detail=f"{type(exc).__name__}: {exc.detail}")
        seen: LedgerEntry = update["ledger"][node.seam.name]
        ledger = {"ledger": {node.seam.name: seen}}
        if seen["state"] == "completed":
            owed = float(update.get("owed", 0.0))
            residue = str(update.get("residue", ""))
            verdict = "residue" if owed or residue else "compensated"
            detail = f"owed {owed:,.0f}; {residue}" if verdict == "residue" else ""
            return {**ledger, **self.settle(step, key, verdict, forward, seen, detail)}
        if seen["state"] == "rejected":  # the partner's answer: never asked again
            return {**ledger, **self.settle(step, key, "refused", forward, seen,
                                            await self.reason(node, seen))}
        if seen["state"] in TERMINAL:
            return {**ledger, **self.settle(step, key, "uncompensated", forward, seen)}
        return ledger  # recorded unknown or working: the next pass reads it again

    async def end_open(self, key: str, step: SagaStep,
                       forward: LedgerEntry) -> dict[str, Any]:
        """A step the plan undoes with nothing: settled, or cancelled if still open."""
        if forward["state"] == "completed":
            return self.settle(step, key, step.settles, forward)
        if forward["state"] in TERMINAL or step.seam not in self.forward:
            return self.settle(step, key, "not committed", forward)
        node = self.forward[step.seam]
        await node.cancel(forward["task_id"])
        task = await node.read(forward["task_id"])
        seen = {"ledger": {key: observe(forward, task, node.partner.release)}}
        if state_name(task.status.state) == "completed":
            return seen  # it committed meanwhile: the next pass settles it
        return {**seen, **self.settle(step, key, "cancelled", forward)}

    def node(self, key: str, step: SagaStep) -> RemoteAgentNode:
        """The undo node for this entry; an adopted orphan's undo has its own name."""
        node = self.undo[step.seam]
        if key == step.seam:
            return node
        return replace(node, seam=replace(node.seam, name=undo_name(key)))

    async def reason(self, node: RemoteAgentNode, seen: LedgerEntry) -> str:
        """The partner's reason for a refusal: partner text, cleaned and capped."""
        task = await node.read(seen["task_id"])
        message = task.status.message
        parts = get_data_parts(message.parts) if message.parts else []
        said = str(parts[0].get("why", "")) if parts else ""
        return str(quote(said, DETAIL_MAX_CHARS))

    def settle(self, step: SagaStep, key: str, verdict: str, forward: LedgerEntry,
               undo: LedgerEntry | None = None, detail: str = "") -> dict[str, Any]:
        """Record how one step ended; anything left over is a finding for people."""
        if verdict in FINDINGS:
            self.log(Finding(
                step.seam, verdict, (undo or forward)["idempotency_key"],
                forward["task_id"], str(run_value("thread_id") or ""),
                (undo or forward)["seen_at"], forward["authority"],
                (undo or forward)["partner_release"],
                "; ".join(d for d in (detail, f"settled by {step.settles}") if d),
                undo["task_id"] if undo else ""))
        return {"undone": {key: verdict}}


__all__ = ["FINDINGS", "OPEN", "Compensator", "SagaPlan", "SagaStep", "UndoAuth",
           "check_plan", "held_to_plan", "undo_name"]
