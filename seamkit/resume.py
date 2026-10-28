"""The cross-owner resume state machine: what a caller does with a recorded pause.

A2A's input-required asks the client, never a person. So every paused task on a seam is
read through one table: the state the partner reports, and who the pause waits on,
decide the step. Before the step is taken, a resume re-checks, in a fixed order, what
can have changed while the person was away: the ledger (is there still something to
resume?), the authority (Chapter 7's verdict on the chain the task was sent under), the
card (is it the one the task was sent under, or one with the same terms for the
skill?), and the question (is it the one that was asked?).

`wake` starts a run on a thread whose task changed, and never writes into one with a
run in flight. `sweep` is the polling loop that calls it. Since Chapter 11 a push
receiver calls `wake` as the trigger, and the sweep stays as the backstop under it.
`fork` carries the head's ledger into a fork of a thread, so time travel on the
caller's side never sends a task the partner has already run.
"""

from collections.abc import Iterator, Mapping, MutableSet
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from a2a.helpers import get_data_parts
from a2a.types import Task
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config

from seamkit.errors import SeamError
from seamkit.ledger import TERMINAL, LedgerEntry, question_digest, state_name
from seamkit.remote import RemoteAgentNode, addressee

Step = Literal["read", "ask", "reask", "refuse", "wait", "renew", "hold", "cancel",
               "expire"]

# (state the partner reports now, who the pause waits on) -> the step a resume takes.
# "us" is a role this seam answers for, "them" anyone else at either owner, and
# "nobody" the partner's own machinery. Every other pair, final states included, reads.
TRANSITIONS: dict[tuple[str, str], Step] = {
    ("input-required", "us"): "ask",  # a Claims person, by role, through interrupt()
    ("input-required", "them"): "refuse",  # PartnerNeedsInput; shown to nobody here
    ("working", "them"): "wait",  # the partner's own people, up to the wait horizon
    ("working", "nobody"): "wait",
    ("submitted", "nobody"): "wait",
    ("auth-required", "us"): "renew",  # Chapter 7: the same person, the same claim
}

PAUSED = frozenset({"input-required", "auth-required", "working", "submitted"})


@dataclass(frozen=True)
class Held:
    """What the caller holds now, to compare a recorded task against."""

    card: str  # the digest of the partner's card the connection holds now
    terms: str  # the digest of the skill's terms in force on that card
    revoked: str  # why the chain the task was sent under is dead, or ""
    roles: tuple[str, ...]  # the roles this seam answers for (SeamSpec.roles)
    asks_caller: bool  # Chapter 3's rule, for a seam that lists no roles
    wait_seconds: float | None  # the caller's wait horizon on this seam


def waits_on(task: Task, held: Held) -> str:
    """"us", "them" or "nobody": who the paused task needs, as the caller sees it."""
    state, to = state_name(task.status.state), addressee(task)
    if state == "auth-required":
        return "us"  # a credential only the caller can present
    if state == "input-required":
        mine = to in held.roles if held.roles else held.asks_caller
        return "us" if mine else "them"
    return "them" if to else "nobody"


def resume_step(entry: LedgerEntry, task: Task, held: Held, now: float) -> Step:
    """Re-check a recorded task in a fixed order, then look its step up."""
    state = state_name(task.status.state)
    if entry["state"] in TERMINAL or state in TERMINAL:  # 1. ledger: nothing paused
        return "read"
    if held.revoked:  # 2. authority: Chapter 7's verdict, on the recorded chain
        return "cancel"
    if entry["card_digest"] != held.card and entry.get("terms") != held.terms:
        return "hold"  # 3. card: sent under terms the partner no longer offers
    step = TRANSITIONS.get((state, waits_on(task, held)), "read")
    waited = now - entry.get("since", now)
    if step == "wait" and held.wait_seconds is not None and waited > held.wait_seconds:
        return "expire"  # past the caller's wait horizon
    if step == "ask" and entry.get("asked") not in ("", None, asked(task)):
        return "reask"  # 4. question: an answer to the old one approves the wrong thing
    return step


def asked(task: Task) -> str:
    """The digest of the question the task asks now; "" if it asks none."""
    if state_name(task.status.state) != "input-required":
        return ""
    parts = get_data_parts(task.status.message.parts)
    return question_digest(parts[0]) if parts else ""


# Waking a thread ----------------------------------------------------------------------

@contextmanager
def as_thread(config: RunnableConfig) -> Iterator[None]:
    """A read outside a run, under the thread's config, so it carries its session."""
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


async def wake(graph: Any, config: RunnableConfig, running: MutableSet[str]) -> str:
    """Start a run on a thread. Never write into a thread that has one in flight.

    Returns how the run ended: "done", a SeamError's name, "asking" for a thread that
    stopped on a Claims person's interrupt, or "busy" if a run was already in flight.
    """
    thread = str(config["configurable"]["thread_id"])
    if thread in running:
        return "busy"  # its own run will read the task; a second writer would race it
    running.add(thread)
    try:
        await graph.ainvoke(None, config)
    except SeamError as exc:  # typed and recorded in the ledger: the run's outcome
        return type(exc).__name__
    finally:
        running.discard(thread)
    stopped = await graph.aget_state(config)
    return "asking" if stopped.interrupts else "done"


def changed(entry: LedgerEntry, task: Task) -> bool:
    """Whether the partner's task moved since the entry was written."""
    return (state_name(task.status.state) != entry["state"]
            or addressee(task) != entry.get("waiting_on", "")
            or asked(task) != entry.get("asked", ""))


async def sweep(graph: Any, threads: Mapping[str, RunnableConfig],
                nodes: Mapping[str, RemoteAgentNode],
                running: MutableSet[str]) -> dict[str, str]:
    """Read every paused task the ledger records; wake each thread whose task moved.

    The poller, kept under push as the backstop (Chapter 11). It reads the partner's
    task, never the partner's checkpoint, and writes nothing itself: the run it starts
    does. A seam with no node in `nodes` is skipped, so each backstop sweeps its own.
    """
    woken: dict[str, str] = {}
    for thread, config in threads.items():
        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            continue  # waiting on a Claims person: that person's answer resumes it
        ledger: Mapping[str, LedgerEntry] = snapshot.values.get("ledger", {})
        for seam, entry in ledger.items():
            if seam not in nodes:  # a backstop sweeps its own seam (Chapter 11)
                continue
            if entry["state"] not in PAUSED or not entry["task_id"]:
                continue
            with as_thread(config):
                task = await nodes[seam].read(entry["task_id"])
            if changed(entry, task):
                woken[thread] = await wake(graph, config, running)
                break
    return woken


# Time travel stops at the seam -------------------------------------------------------

async def fork(graph: Any, config: RunnableConfig,
               checkpoint_id: str) -> RunnableConfig:
    """A fork of the thread at `checkpoint_id` that knows every task sent since.

    The ledger lives in the checkpoint, so a fork rewinds it with everything else, and
    a seam whose entry is gone sends a new task. The head's entries are written into the
    fork first, each one version above the fork's own, so its node reads them instead.
    """
    head: Mapping[str, LedgerEntry] = (
        await graph.aget_state(config)).values.get("ledger", {})
    past: RunnableConfig = {"configurable": {**config["configurable"],
                                             "checkpoint_ns": "",
                                             "checkpoint_id": checkpoint_id}}
    old: Mapping[str, LedgerEntry] = (
        await graph.aget_state(past)).values.get("ledger", {})
    carried = {seam: {**entry, "version": old[seam]["version"] + 1 if seam in old
                      else entry["version"]}
               for seam, entry in head.items() if old.get(seam) != entry}
    if not carried:
        return past
    forked: RunnableConfig = await graph.aupdate_state(past, {"ledger": carried})
    return {"configurable": {**config["configurable"], **forked["configurable"]}}
