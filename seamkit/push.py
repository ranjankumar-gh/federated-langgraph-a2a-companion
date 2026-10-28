"""Push notifications as a doorbell: a push says which task to read, never what it says.

A2A's push authentication is the caller's own secret echoed back. The partner sends the
token the caller put on the task's push config, and no signature of the partner's covers
the body. So the receiver never reads the body into state. It checks that the sender
may ring for this one task, puts one pending wake on the thread, and answers at once.
The run that wake starts reads the task under the caller's authority, through the
ingress filter, like every other read (Chapter 10's `wake`, unchanged).

The token is derived, never stored: an HMAC of the seam and the ledger entry's
idempotency key, which exists before the send because the entry is recorded first.

A push is lost silently: the SDK release the companion pins makes one attempt per push
and logs a failure. So every push seam keeps a backstop poll, Chapter 10's `sweep` at a
cadence set from the seam's wait horizon, and the backstop bounds how long a lost push
can go unnoticed.
"""

import hashlib
import hmac
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from a2a.types import TaskPushNotificationConfig
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from seamkit.ledger import TERMINAL, LedgerEntry
from seamkit.remote import RemoteAgentNode
from seamkit.resume import sweep, wake

# The notification token header the SDK's push sender sets (the A2A reference appendix
# gives its wire spelling). Starlette reads headers case-insensitively.
TOKEN_HEADER = "x-a2a-notification-token"

push_logger = logging.getLogger("seamkit.push")


def push_token(key: bytes, seam: str, idempotency_key: str) -> str:
    """The token for one task: derived from the key the send carries, never stored."""
    message = f"{seam}\n{idempotency_key}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def rings_for(key: bytes, entry: LedgerEntry, token: str) -> bool:
    """Whether `token` is the one this entry's task was sent with: constant time."""
    expected = push_token(key, entry["seam"], entry["idempotency_key"])
    return hmac.compare_digest(expected, token)


@dataclass(frozen=True)
class PushTo:
    """Where a partner rings for a seam's tasks, and the key their tokens come from."""

    url: str  # the receiver's base URL; the path names the thread and the seam
    key: bytes  # Claims' secret; it never leaves Claims

    def __call__(self, seam: str, key: str | None) -> TaskPushNotificationConfig | None:
        if not key:  # a follow-up on a task that already has its bell, or a tool call
            return None
        thread = get_config()["configurable"]["thread_id"]
        return TaskPushNotificationConfig(url=f"{self.url}/{thread}/{seam}",
                                          token=push_token(self.key, seam, key))


# One pending wake per thread ----------------------------------------------------------

@dataclass
class WakeQueue:
    """At most one pending wake per thread, drained only when no run is in flight.

    It replaces Chapter 10's `running` set, which each caller of `wake` passed in: the
    receiver, the backstop, and every run Claims starts share this one.
    """

    graph: Any
    configs: Callable[[str], RunnableConfig]  # thread id -> the thread's run config
    running: set[str] = field(default_factory=set)
    pending: dict[str, None] = field(default_factory=dict)  # an ordered set of threads

    def ring(self, thread: str) -> None:
        self.pending[thread] = None  # a second ring for the same thread adds nothing

    async def drain(self) -> dict[str, str]:
        """Wake every pending thread with no run in flight; how each run ended."""
        woken: dict[str, str] = {}
        for thread in list(self.pending):
            if thread in self.running:
                continue  # its own run is reading the task; drain again after it ends
            del self.pending[thread]
            woken[thread] = await wake(self.graph, self.configs(thread), self.running)
        return woken

    async def run(self, values: Any, config: RunnableConfig) -> Any:
        """Start a run as a person or an API call does, and hold the thread meanwhile."""
        thread = str(config["configurable"]["thread_id"])
        self.running.add(thread)
        try:
            return await self.graph.ainvoke(values, config)
        finally:
            self.running.discard(thread)

    async def entry(self, thread: str, seam: str) -> LedgerEntry | None:
        """The thread's entry for a seam, as last checkpointed: read, never written."""
        snapshot = await self.graph.aget_state(self.configs(thread))
        ledger: Mapping[str, LedgerEntry] = snapshot.values.get("ledger", {})
        return ledger.get(seam)


# The receiver -------------------------------------------------------------------------

@dataclass(frozen=True)
class PushRecord:
    """One record per push received: the verdict, never the body."""

    seam: str
    thread: str
    task_id: str  # what the push claimed, as compared; "" if it named none
    status: int
    verdict: str


def log_push(record: PushRecord) -> None:
    push_logger.info("push received", extra={"push": asdict(record)})


@dataclass
class RateLimit:
    """At most `per_window` pushes per partner in any `window` seconds."""

    per_window: int
    window: float
    now: Callable[[], float]
    seen: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, partner: str) -> bool:
        t = self.now()
        recent = [s for s in self.seen.get(partner, []) if t - s < self.window]
        self.seen[partner] = recent
        if len(recent) >= self.per_window:
            return False
        recent.append(t)
        return True


def pushed_task_id(body: bytes) -> str:
    """The task id a push names, read to compare and for nothing else."""
    try:
        event = json.loads(body or b"{}")
    except ValueError:
        return ""
    if not isinstance(event, dict):
        return ""
    if isinstance(task := event.get("task"), dict):
        return str(task.get("id", ""))
    for kind in ("statusUpdate", "artifactUpdate"):
        if isinstance(update := event.get(kind), dict):
            return str(update.get("taskId", ""))
    return ""


def receiver(key: bytes, queue: WakeQueue, limit: RateLimit,
             log: Callable[[PushRecord], None] = log_push,
             path: str = "/push") -> Starlette:
    """Claims' push endpoint, POST {path}/{thread}/{seam}: verified, then queued."""

    async def verify(request: Request) -> JSONResponse:
        thread, seam = request.path_params["thread"], request.path_params["seam"]
        claimed = pushed_task_id(await request.body())

        def answer(status: int, verdict: str) -> JSONResponse:
            log(PushRecord(seam, thread, claimed, status, verdict))
            return JSONResponse({"verdict": verdict}, status_code=status)

        if not limit.allow(seam):  # 1. per partner, before any work is done
            return answer(429, "rate limited")
        entry = await queue.entry(thread, seam)
        token = request.headers.get(TOKEN_HEADER, "")
        if entry is None or not rings_for(key, entry, token):  # 2. this task's token
            return answer(401, "token")
        if entry["task_id"] and claimed != entry["task_id"]:  # 3. the task we sent
            return answer(404, "task")
        if entry["state"] in TERMINAL:  # final in the ledger: nothing left to read
            return answer(200, "final")
        queue.ring(thread)  # 4. one pending wake; the body is never read into state
        return answer(202, "queued")

    route = Route(path + "/{thread}/{seam}", verify, methods=["POST"])
    return Starlette(routes=[route])


# The backstop poll --------------------------------------------------------------------

@dataclass
class Backstop:
    """Chapter 10's sweep, kept under push for one seam, at that seam's own cadence."""

    queue: WakeQueue
    threads: Callable[[], Mapping[str, RunnableConfig]]
    node: RemoteAgentNode  # the seam's node: the sweep reads only its tasks
    every: float  # seconds: the seam's backstop_seconds
    last: float = float("-inf")

    async def tick(self, now: float) -> dict[str, str]:
        """Sweep once a cadence has passed since the last sweep; how each run ended."""
        if now - self.last < self.every:
            return {}
        self.last = now
        return await sweep(self.queue.graph, self.threads(),
                           {self.node.seam.name: self.node}, self.queue.running)


# A doorbell for a list, not a task ----------------------------------------------------

def doorbell(ring: Callable[[], None]) -> Starlette:
    """POST / rings; the body is never read. What the ring triggers is a pull."""

    async def rung(request: Request) -> JSONResponse:
        ring()
        return JSONResponse({"verdict": "rung"}, status_code=202)

    return Starlette(routes=[Route("/", rung, methods=["POST"])])
