"""The reconciliation job: the caller's ledger against what each partner says.

Two owners each keep half the record of a seam's tasks, and nothing on the wire says
when the halves have diverged. The job compares them on a schedule:

- inward, from every unresolved entry: read the task it names, or look its key up in
  the partner's task list when it names none;
- outward, from the partner's list: match every task the partner holds for this caller
  to the entry that names it, now or earlier.

Each disagreement gets one verdict and one action. The job never writes the ledger: a
thread's own run is its only writer (Chapter 9). It rings the thread's wake queue
(Chapter 11), whose run reads the task and records what it saw, or it files a finding
for a person, because some divergences only the partner can settle.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime

from a2a.types import Task
from a2a.utils.errors import TaskNotFoundError
from langchain_core.runnables import RunnableConfig

import seamkit.ledger
from seamkit.errors import PartnerFailed, PartnerUnavailable
from seamkit.ledger import TERMINAL, LedgerEntry, state_name
from seamkit.push import WakeQueue
from seamkit.remote import RemoteAgentNode
from seamkit.resume import as_thread, changed

# Each verdict, and the one thing the job does about it.
ACTIONS: Mapping[str, str] = {
    "agrees": "none",
    "behind": "wake",  # the partner's task moved: the thread's run reads it
    "found": "wake",  # the key opened a task the entry never recorded: the run adopts
    "absent": "finding",  # no task under the key; a list can skip one, so no resend
    "lost": "finding",  # the partner has no task for a recorded id: never resend
    "orphan": "finding",  # a partner task no entry names (Chapter 13 compensates)
    "unreadable": "retry",  # the partner did not answer: next run, then a person
    # Chapter 13: what a compensation could not undo; each one is for people
    "residue": "finding",  # undone, with something left owed or kept
    "refused": "finding",  # the partner said no: never asked again
    "uncompensated": "finding",  # no answer, or no task to name: the job takes over
}


@dataclass(frozen=True)
class Finding:
    """One disagreement: enough for both teams to settle one case."""

    seam: str
    verdict: str
    idempotency_key: str  # joins the finding to the entry and its crossing records
    task_id: str
    thread: str = ""
    seen_at: str = ""  # when Claims last saw the task, if it ever did
    authority: str = ""  # the chain the task was sent under, by id
    partner_release: str = ""
    detail: str = ""
    compensation: str = ""  # Chapter 13: the compensation task, if one was sent


reconcile_logger = logging.getLogger("seamkit.reconcile")


def log_finding(finding: Finding) -> None:
    reconcile_logger.warning("reconciliation finding",
                             extra={"finding": asdict(finding)})


def first_key(task: Task) -> str:
    """The id of the message that opened a task: the caller's key, if it sent one."""
    return task.history[0].message_id if task.history else ""


def skill_of(task: Task) -> str:
    first = task.history[0] if task.history else None
    if first is None or "skill" not in first.metadata:
        return ""
    return str(first.metadata["skill"])


def not_found(exc: PartnerFailed) -> bool:
    return isinstance(exc.__cause__, TaskNotFoundError)


@dataclass
class Reconciler:
    """Compares one Claims deployment's ledgers with its partners, never writing one."""

    queue: WakeQueue
    threads: Callable[[], Mapping[str, RunnableConfig]]  # every thread with a ledger
    nodes: Mapping[str, RemoteAgentNode]  # seam -> its node, for reads and lists
    config: RunnableConfig  # the job's own session: it lists, and reads nothing
    grace: float = 600.0  # seconds: an intended entry older than this may have left
    page_after: int = 3  # runs a task may stay unreadable before a person is paged
    log: Callable[[Finding], None] = log_finding
    misses: dict[tuple[str, str], int] = field(default_factory=dict)

    async def run(self) -> list[Finding]:
        """One pass each way. Returns every verdict but `agrees`, after acting on it."""
        threads = self.threads()
        ledgers = {thread: await self._ledger(config)
                   for thread, config in threads.items()}
        verdicts: list[Finding] = []
        for thread, ledger in ledgers.items():
            for seam, entry in ledger.items():
                if seam in self.nodes and self.unresolved(entry):
                    verdicts.append(await self.inward(thread, threads[thread], entry))
        for seam in self.nodes:
            verdicts += await self.outward(seam, ledgers)
        for finding in verdicts:
            self._act(finding)
        return [f for f in verdicts if f.verdict != "agrees"]

    def unresolved(self, entry: LedgerEntry) -> bool:
        """An entry whose task the job must look at: open, unknown, or maybe sent."""
        if entry["state"] in TERMINAL:
            return False
        if entry["state"] == "intended":  # a run in flight, or a crash after the send
            seen = datetime.fromisoformat(entry["seen_at"]).timestamp()
            return seamkit.ledger.CLOCK() - seen > self.grace
        return True

    async def inward(self, thread: str, config: RunnableConfig,
                     entry: LedgerEntry) -> Finding:
        """From one entry to the partner's task: read it, or look its key up."""
        node = self.nodes[entry["seam"]]

        def verdict(name: str, task_id: str = "", detail: str = "") -> Finding:
            return Finding(entry["seam"], name, entry["idempotency_key"],
                           task_id or entry["task_id"], thread, entry["seen_at"],
                           entry["authority"], entry["partner_release"], detail)

        try:
            with as_thread(config):  # the claim's own chain, as the backstop reads
                task = (await node.read(entry["task_id"]) if entry["task_id"]
                        else await node.lookup(entry))
        except PartnerUnavailable as exc:
            return verdict("unreadable", detail=exc.detail)
        except PartnerFailed as exc:
            if not_found(exc):  # a recorded id the partner no longer holds
                return verdict("lost", detail=f"the entry says {entry['state']}")
            return verdict("unreadable", detail=exc.detail)
        if task is None:
            return verdict("absent", detail=f"no task under the key ({entry['state']})")
        if not entry["task_id"]:
            return verdict("found", task.id, state_name(task.status.state))
        if changed(entry, task):
            return verdict("behind", detail=state_name(task.status.state))
        return verdict("agrees")

    async def outward(self, seam: str, ledgers: Mapping[str, Mapping[str, LedgerEntry]],
                      ) -> list[Finding]:
        """From the partner's list to the ledger: every task it holds must be named."""
        node = self.nodes[seam]
        named: set[str] = set()
        keys: dict[str, tuple[str, LedgerEntry]] = {}
        for thread, ledger in ledgers.items():
            for name, kept in ledger.items():  # an adopted orphan has its own entry
                named.add(kept.get("compensates", ""))  # Chapter 13: undone tasks too
                if kept["seam"] != seam:
                    continue
                named |= {kept["task_id"], *kept["earlier"]}
                if name == seam:
                    keys[kept["idempotency_key"]] = (thread, kept)
        try:
            with as_thread(self.config):
                listed = await node.tasks()
        except PartnerUnavailable as exc:
            return [Finding(seam, "unreadable", "", "", detail=f"list: {exc.detail}")]
        orphans = []
        for task in listed:
            if task.id in named or skill_of(task) != node.seam.skill:
                continue  # an entry names it, now or earlier; or it is another seam's
            key = first_key(task)
            thread, entry = keys.get(key, ("", None))
            if entry is not None and not entry["task_id"]:
                continue  # the entry is still recording it: the inward pass finds it
            why = ("the entry with its key names " + entry["task_id"] if entry
                   else "no entry has its key")
            orphans.append(Finding(seam, "orphan", key, task.id, thread,
                                   entry["seen_at"] if entry else "",
                                   entry["authority"] if entry else "",
                                   node.partner.release,
                                   f"{state_name(task.status.state)}; {why}"))
        return orphans

    def _act(self, finding: Finding) -> None:
        action = ACTIONS[finding.verdict]
        if action == "wake":  # the run reads the task and writes what it saw
            self.queue.ring(finding.thread)
        elif action == "finding":
            self.log(finding)
        elif action == "retry":
            index = (finding.seam, finding.task_id or finding.idempotency_key)
            self.misses[index] = self.misses.get(index, 0) + 1
            if self.misses[index] >= self.page_after:  # a person looks at it now
                self.log(finding)

    async def _ledger(self, config: RunnableConfig) -> Mapping[str, LedgerEntry]:
        snapshot = await self.queue.graph.aget_state(config)
        ledger: Mapping[str, LedgerEntry] = snapshot.values.get("ledger", {})
        return ledger
