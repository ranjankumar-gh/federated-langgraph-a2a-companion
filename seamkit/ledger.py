"""The task ledger: the caller's own record of every task it has sent across a seam.

Each owner keeps a checkpoint the other cannot read, and the A2A task is the one object
both can name. The ledger is where the caller writes down what it knows about that
object: the ids, the key it sent under, the card and the credential it was sent with,
and the state the partner last reported, with the time it was seen. An entry is a dated
belief about the partner's task. It is never the task.

Entries are plain dictionaries so that a checkpoint written by one Claims release can be
read by the next, whatever classes that release has. For the same reason, the keys
Chapter 10 adds for a paused task are NotRequired: an entry an earlier release wrote
still loads, and reads as a task nobody has asked anything on yet.

Since Chapter 13 a compensation is a task like any other, so it has an entry of its
own. The entry names the task it undoes (`compensates`), and its key is derived from
the forward entry's key, so a resent compensation is a repeat, never a second undo. The
forward entry is never rewritten to say it was undone.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, NotRequired, TypedDict

from a2a.helpers import get_data_parts
from a2a.types import Task, TaskState

from seamkit.errors import StaleLedgerWrite
from seamkit.retry import onward_key

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
    # Chapter 10: who a paused task waits on, and since when; the digest of the
    # question it asks; and the digest of the skill's terms it was sent under.
    waiting_on: NotRequired[str]  # "claims/claims_manager", "finance/...", or ""
    since: NotRequired[float]  # when Claims first saw this pause, on CLOCK
    asked: NotRequired[str]  # sha256 of the question as the partner sent it
    terms: NotRequired[str]  # sha256 of the skill's terms in force at the send
    # Chapter 13: the forward task a compensation entry undoes; absent on every other
    compensates: NotRequired[str]
    # Chapter 14: the partner stop under which this thread's own run cancelled the task
    stopped_by: NotRequired[str]
    # Chapter 15: the behaviour version the skill was recorded under when this task
    # was sent, from the seam's accepted canary baseline; "" for a seam with none
    behaviour: NotRequired[str]
    # Chapter 17: the W3C traceparent of the send that opened the task; what Claims
    # last sent on it ({"digest", "fields"}, under the task's nonce); and the digest of
    # the newest seam receipt for it
    traceparent: NotRequired[str]
    sent: NotRequired[dict[str, Any]]
    receipt: NotRequired[str]
    # Chapter 18: the digest of the seam cost ledger's entry for this task's charge
    cost: NotRequired[str]


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


# The clock every entry is stamped with. A test that lets days pass replaces it.
CLOCK: Callable[[], float] = time.time


def now() -> str:
    return datetime.fromtimestamp(CLOCK(), UTC).isoformat(timespec="milliseconds")


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
    seen: LedgerEntry = {
        **entry, "context_id": task.context_id, "task_id": task.id,
        "state": state_name(task.status.state), "partner_release": release,
        "seen_at": now(), "version": entry["version"] + 1}
    return paused(entry, seen, task)


def paused(entry: LedgerEntry, seen: LedgerEntry, task: Task) -> LedgerEntry:
    """Chapter 10: who the task now waits on, since when, and what it asks."""
    message = task.status.message
    parts = get_data_parts(message.parts) if message.parts else []
    said = parts[0] if parts else {}
    to = str(said.get("to") or said.get("waiting_on") or "")
    # Only a new state restarts the clock: a partner that reassigns a waiting task to
    # someone else changes who it waits on, never how long the caller has waited.
    same = entry["state"] == seen["state"]
    asked = question_digest(said) if seen["state"] == "input-required" else ""
    return {**seen, "waiting_on": to, "asked": asked,
            "since": entry.get("since", CLOCK()) if same else CLOCK()}


def moved(old: LedgerEntry, new: LedgerEntry) -> bool:
    """A new state, a new addressee, or a new question: each is worth a record."""
    return (old["state"] != new["state"]
            or old.get("waiting_on", "") != new.get("waiting_on", "")
            or old.get("asked", "") != new.get("asked", ""))


def question_digest(question: Mapping[str, Any]) -> str:
    """The question as the partner sent it, hashed: an approval is bound to this."""
    raw = json.dumps(dict(question), sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def compensation(forward: LedgerEntry, seam: str, skill: str, card_digest: str,
                 authority: str, release: str) -> LedgerEntry:
    """The entry for a compensation of `forward`'s task, recorded before it is sent.

    Its key is derived from the forward key and the forward task, so every resend of
    this compensation, after a crash or a lost answer, repeats the same request.
    """
    key = onward_key(forward["idempotency_key"], f"compensate {forward['task_id']}")
    return {**intended(seam, skill, card_digest, authority, release),
            "idempotency_key": key, "compensates": forward["task_id"]}


def unanswered(entry: LedgerEntry) -> LedgerEntry:
    """A send that left and got no answer: the partner may or may not hold a task."""
    return {**entry, "state": "unknown", "seen_at": now(),
            "version": entry["version"] + 1}


def follow_on(entry: LedgerEntry) -> LedgerEntry:
    """A new task in the same context, once this entry's task is final."""
    return {**entry, "idempotency_key": uuid.uuid4().hex, "task_id": "",
            "state": "intended", "seen_at": now(), "wrote": [],
            "waiting_on": "", "asked": "",
            "earlier": [*entry["earlier"], entry["task_id"]],
            "version": entry["version"] + 1}
