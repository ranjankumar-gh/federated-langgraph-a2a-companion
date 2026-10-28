"""The task ledger: the caller's own record of every task it has sent across a seam.

Each owner keeps a checkpoint the other cannot read, and the A2A task is the one object
both can name. The ledger is where the caller writes down what it knows about that
object: the ids, the key it sent under, the card and the credential it was sent with,
and the state the partner last reported, with the time it was seen. An entry is a dated
belief about the partner's task. It is never the task.

Entries are plain dictionaries so that a checkpoint written by one Claims release can be
read by the next, whatever classes that release has.
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypedDict

from a2a.types import Task, TaskState

from seamkit.errors import StaleLedgerWrite

# A task in one of these states will never change again on the partner's side.
TERMINAL = frozenset({"completed", "failed", "canceled", "rejected"})


class LedgerEntry(TypedDict):
    """Claims' record of one task on one seam: a dated belief, never the task."""

    seam: str  # the seam's name, which the router and the inventory already use
    skill: str  # the skill id on the partner's card
    idempotency_key: str  # minted before the send, and sent as the message id
    context_id: str  # partner-issued; empty until the partner answers
    task_id: str
    card_digest: str  # the card the task was sent under
    authority: str  # the credential presented, by name, never its value
    partner_release: str  # the release that reported `state`
    state: str  # last known: "intended", "unknown", or the partner's task state
    seen_at: str  # when Claims saw `state`, ISO 8601 in UTC
    wrote: list[str]  # the state fields this task's artifact wrote
    earlier: list[str]  # task ids this seam sent before this one, oldest first
    version: int  # every write bumps it; the reducer checks it


def merge_ledger(current: Mapping[str, LedgerEntry],
                 update: Mapping[str, LedgerEntry]) -> dict[str, LedgerEntry]:
    """The ClaimState.ledger reducer: compare-and-set per seam, with a terminal floor."""
    merged = dict(current)
    for seam, new in update.items():
        old = merged.get(seam)
        if old is not None:
            if new["version"] != old["version"] + 1:  # the writer read an older entry
                raise StaleLedgerWrite(seam, f"v{new['version']} over v{old['version']}",
                                       old["task_id"])
            same_task = new["idempotency_key"] == old["idempotency_key"]
            if same_task and old["state"] in TERMINAL and new["state"] != old["state"]:
                raise StaleLedgerWrite(seam, f"{old['state']} is final", old["task_id"])
            if not same_task and old["state"] not in TERMINAL:
                raise StaleLedgerWrite(seam, "a second task while one is open",
                                       old["task_id"])
        merged[seam] = new
    return merged


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def state_name(state: int) -> str:
    """TASK_STATE_INPUT_REQUIRED -> "input-required", the specification's spelling."""
    name = str(TaskState.Name(state))
    return name.removeprefix("TASK_STATE_").lower().replace("_", "-")


def intended(seam: str, skill: str, card_digest: str, authority: str,
             release: str) -> LedgerEntry:
    """The first version of an entry, written before anything is sent."""
    return LedgerEntry(seam=seam, skill=skill, idempotency_key=uuid.uuid4().hex,
                       context_id="", task_id="", card_digest=card_digest,
                       authority=authority, partner_release=release, state="intended",
                       seen_at=now(), wrote=[], earlier=[], version=1)


def observe(entry: LedgerEntry, task: Task, release: str) -> LedgerEntry:
    """The next version of an entry, from a task its writer saw. Every writer uses it."""
    return {**entry, "context_id": task.context_id, "task_id": task.id,
            "state": state_name(task.status.state), "partner_release": release,
            "seen_at": now(), "version": entry["version"] + 1}


def unanswered(entry: LedgerEntry) -> LedgerEntry:
    """A send that left and got no answer: the partner may or may not hold a task."""
    return {**entry, "state": "unknown", "seen_at": now(),
            "version": entry["version"] + 1}


def follow_on(entry: LedgerEntry) -> LedgerEntry:
    """A new task in the same context, once this entry's task is final."""
    return {**entry, "idempotency_key": uuid.uuid4().hex, "task_id": "",
            "state": "intended", "seen_at": now(), "wrote": [],
            "earlier": [*entry["earlier"], entry["task_id"]],
            "version": entry["version"] + 1}
