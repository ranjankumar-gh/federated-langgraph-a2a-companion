"""Long-running tasks: the forged push, the doorbell receiver, one wake per thread, the
backstop poll, a dropped stream, Northgate's revocation doorbell, and the matrix."""

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.helpers import new_data_message
from a2a.types import (
    AgentCapabilities,
    TaskPushNotificationConfig,
    GetTaskRequest,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskState,
)
from langgraph.checkpoint.memory import InMemorySaver
from starlette.types import ASGIApp, Receive, Scope, Send

import seamkit.ledger
from meridian.drill import T0, DrillFederation
from meridian.federated import (
    EXPECTATIONS,
    SEAMS,
    A2APartners,
    a2a_workers,
    push_seams,
)
from meridian.finance_server import FinancePartner
from meridian.identity import ADJUSTER, MeridianIssuer, revocation_sources
from meridian.incident import letter_receiver
from meridian.inprocess import ClaimGraph
from meridian.risk_server import RiskPartner
from meridian.samples import clean_claim, suspect_claim
from northgate.server import DOORBELL_PATH, NORTHGATE_URL, NorthgatePartner
from seamkit.delivery import MODES, uncovered
from seamkit.drill import Clock, probe
from seamkit.errors import TaskUnfinished
from seamkit.ledger import LedgerEntry
from seamkit.push import (
    TOKEN_HEADER,
    Backstop,
    PushRecord,
    PushTo,
    RateLimit,
    WakeQueue,
    push_token,
    receiver,
)
from seamkit.server import StoreWriter
from tests.wire import Federation, thread

pytestmark = pytest.mark.anyio

SEAM = "repair_estimate"
MINUTE = 60.0
RECEIVER = "http://claims.example.test"
KEY = b"claims-push-key-2026"  # Claims' secret; a deployment reads it from its vault
SEAM_TOKEN = "northgate-push-2026"  # the incident's one token for every task
# The estimate seam alone, as Chapter 7's drill runs it, on push: the send returns once
# the task exists, and Northgate rings.
PUSHED = {SEAM: push_seams()[SEAM]}


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(T0)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    yield shared


@dataclass
class Late:
    """Claims' receiver, set once Claims' graph exists; Northgate is built first."""

    app: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert self.app is not None
        await self.app(scope, receive, send)


@dataclass
class Bell(httpx.AsyncBaseTransport):
    """The route from Northgate's push sender to Claims. `down`: a Claims deploy."""

    late: Late
    clock: Clock
    down: bool = False
    attempts: list[tuple[float, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.inner = httpx.ASGITransport(app=self.late)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            self.attempts.append((self.clock(), 503))
            return httpx.Response(503, request=request)
        response = await self.inner.handle_async_request(request)
        self.attempts.append((self.clock(), response.status_code))
        return response


@dataclass
class Pushed:
    """Meridian on one clock, Northgate's assessor desk, and Claims' receiver."""

    clock: Clock
    letter: bool = False  # True: the incident's receiver and its one token
    pushes: list[PushRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issuer = MeridianIssuer(now=self.clock)
        self.assessors = asyncio.Event()
        self.late = Late()
        self.bell = Bell(self.late, self.clock)
        lists = revocation_sources(self.issuer)
        self.northgate = NorthgatePartner(
            assessors=self.assessors, revocations=lists, now=self.clock,
            push=httpx.AsyncClient(transport=self.bell, base_url=RECEIVER))
        self.fed = Federation(
            A2APartners(risk=RiskPartner(revocations=lists, now=self.clock),
                        northgate=self.northgate,
                        finance=FinancePartner(revocations=lists, now=self.clock)),
            seams=PUSHED, issuer=self.issuer)

    async def start(self) -> tuple[ClaimGraph, WakeQueue]:
        bell = OneToken if self.letter else PushTo  # the incident: one token per seam
        bells = {SEAM: bell(f"{RECEIVER}/push", KEY)}
        self.saver = InMemorySaver()
        graph = await self.fed.graph(checkpointer=self.saver, bells=bells)
        # Northgate's estimates are proven under this card (Chapter 5's gradient), so
        # its unproven ceiling does not stop CLM-1002's 8,300 before this chapter does.
        record = self.fed.connections[SEAM].record("estimate_repair")
        record.clean.update(f"earlier-{n}" for n in range(20))
        self.queue = WakeQueue(graph, thread)
        self.late.app = (letter_receiver(SEAM_TOKEN, graph, thread, "v7",
                                         self.queue.running) if self.letter
                         else receiver(KEY, self.queue, RateLimit(20, MINUTE,
                                                                  self.clock),
                                       log=self.pushes.append))
        return graph, self.queue

    async def order(self, tid: str) -> LedgerEntry:
        """10:00: the adjuster orders CLM-1002's estimate; Northgate starts on it."""
        with pytest.raises(TaskUnfinished):
            await self.queue.run(suspect_claim(), thread(tid))
        await self.queue.drain()  # Northgate's first pushes, rung during that run
        return await self.entry(tid)

    async def entry(self, tid: str) -> LedgerEntry:
        found = await self.queue.entry(tid, SEAM)
        assert found is not None
        return found

    async def assessor_finishes(self) -> None:
        """10:21: the assessor is done; wait until Northgate has said so."""
        self.clock.t = T0 + 21 * MINUTE
        self.assessors.set()
        executor = self.northgate.executor
        for _ in range(400):
            if executor.acts:  # the commit act point ran: renewed or released
                await asyncio.sleep(0.02)  # and its push has been sent
                return
            await asyncio.sleep(0.005)
        raise TimeoutError("Northgate never reached its commit act point")

    def reads(self) -> int:
        return sum(1 for r in self.fed.records if r.call == "get")


@dataclass(frozen=True)
class OneToken(PushTo):
    """The incident's push config: the seam's one token, on every task it sends."""

    def __call__(self, seam: str, key: str | None) -> TaskPushNotificationConfig | None:
        config = super().__call__(seam, key)
        if config is not None:
            config.token = SEAM_TOKEN
        return config


_THREAD = "clm-1002"


def forged(entry: LedgerEntry, state: str = "TASK_STATE_FAILED") -> bytes:
    """What the forger posts: CLM-1002's real task id, a state Northgate never sent."""
    return json.dumps({"statusUpdate": {
        "taskId": entry["task_id"], "contextId": entry["context_id"],
        "status": {"state": state}}}).encode()


async def post(meridian: Pushed, token: str, body: bytes,
               tid: str = _THREAD) -> httpx.Response:
    """A push from outside Northgate, straight at Claims' receiver."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=meridian.late),
                                 base_url=RECEIVER) as http:
        return await http.post(f"/push/{tid}/{SEAM}", content=body,
                               headers={TOKEN_HEADER: token})


# The incident -------------------------------------------------------------------------

async def test_the_incident_a_forged_failed_locks_out_the_real_estimate(
        clock: Clock) -> None:
    meridian = Pushed(clock, letter=True)
    await meridian.start()
    entry = await meridian.order(_THREAD)
    clock.t = T0 + 2 * MINUTE  # a POST Northgate never sent, with the seam's token
    response = await post(meridian, SEAM_TOKEN, forged(entry))
    # The body became state; the wake's run read Northgate's task, still working, and
    # the ledger's own floor refused that read.
    assert response.json() == {"verdict": "recorded", "woken": "StaleLedgerWrite"}
    assert await recorded_state(meridian) == "failed"
    await meridian.assessor_finishes()  # Northgate asks for a fresh chain to commit
    assert meridian.bell.attempts[-1] == (T0 + 21 * MINUTE, 409)  # "failed is final"
    assert await recorded_state(meridian) == "failed"
    executor = meridian.northgate.executor
    assert executor.pending and not executor.billed  # the estimate never left Northgate
    # The investigator sees a failed estimate and orders it again: a second review.
    await meridian.queue.run(suspect_claim(), thread("clm-1002-again"))
    reviews = [r for r in executor.received if r["skill"] == "estimate_repair"]
    assert [r["request"]["claim_id"] for r in reviews] == ["CLM-1002", "CLM-1002"]
    assert list(executor.billed.values()) == [3.5]  # the second one; the first waits
    assert entry["task_id"] in executor.pending  # a finished estimate nobody will read


async def recorded_state(meridian: Pushed) -> str:
    """The ledger as checkpointed: the thread's own state no longer even reads, because
    the refused write is still pending in its checkpoint and raises again."""
    saved = await meridian.saver.aget_tuple(thread(_THREAD))
    assert saved is not None
    return str(saved.checkpoint["channel_values"]["ledger"][SEAM]["state"])


# The doorbell receiver ----------------------------------------------------------------

async def test_a_forged_push_with_a_real_token_costs_two_reads_and_writes_nothing(
        clock: Clock) -> None:
    meridian = Pushed(clock)
    await meridian.start()
    entry = await meridian.order(_THREAD)
    clock.t = T0 + 2 * MINUTE
    leaked = push_token(KEY, SEAM, entry["idempotency_key"])  # as if it leaked
    before = meridian.reads()
    assert (await post(meridian, leaked, forged(entry))).status_code == 202
    assert await meridian.queue.drain() == {_THREAD: "TaskUnfinished"}
    assert meridian.reads() - before == 2  # the gate's read and the node's read
    assert await meridian.entry(_THREAD) == entry  # nothing written: still working
    await meridian.assessor_finishes()  # the real push rings, and the run reads
    assert await meridian.queue.drain() == {_THREAD: "done"}
    done = await meridian.entry(_THREAD)
    assert done["state"] == "completed" and done["wrote"]
    assert list(meridian.northgate.executor.billed.values()) == [3.5]


async def test_a_token_opens_one_task_and_only_the_task_it_was_sent_with(
        clock: Clock) -> None:
    meridian = Pushed(clock)
    await meridian.start()
    first = await meridian.order(_THREAD)
    second = await meridian.order("clm-1002-b")
    mine = push_token(KEY, SEAM, first["idempotency_key"])
    assert (await post(meridian, mine, forged(second), "clm-1002-b")).status_code == 401
    older = push_token(KEY, SEAM, "a-key-from-an-earlier-task")
    assert (await post(meridian, older, forged(first))).status_code == 401
    wrong = forged({**first, "task_id": second["task_id"]})
    assert (await post(meridian, mine, wrong)).status_code == 404
    assert meridian.queue.pending == {}
    assert [p.verdict for p in meridian.pushes[-3:]] == ["token", "token", "task"]


async def test_duplicates_coalesce_and_a_push_during_a_run_waits_for_it(
        clock: Clock) -> None:
    meridian = Pushed(clock)
    await meridian.start()
    entry = await meridian.order(_THREAD)
    token = push_token(KEY, SEAM, entry["idempotency_key"])
    meridian.queue.running.add(_THREAD)  # the adjuster's own run is in flight
    for _ in range(3):  # the partner retried, or rang once per event
        assert (await post(meridian, token, forged(entry))).status_code == 202
    assert await meridian.queue.drain() == {}  # never a second writer on the thread
    assert list(meridian.queue.pending) == [_THREAD]  # three rings, one pending wake
    meridian.queue.running.discard(_THREAD)
    assert await meridian.queue.drain() == {_THREAD: "TaskUnfinished"}
    assert meridian.queue.pending == {}


async def test_the_receiver_limits_each_partner(clock: Clock) -> None:
    meridian = Pushed(clock)
    await meridian.start()
    entry = await meridian.order(_THREAD)
    token = push_token(KEY, SEAM, entry["idempotency_key"])
    statuses = [(await post(meridian, token, forged(entry))).status_code
                for _ in range(25)]
    assert 429 in statuses and statuses.count(202) < 20


# The backstop poll --------------------------------------------------------------------

async def test_a_lost_push_is_found_within_one_backstop_cadence(clock: Clock) -> None:
    meridian = Pushed(clock)
    graph, queue = await meridian.start()
    await meridian.order(_THREAD)
    nodes = a2a_workers(meridian.fed.connections, seams=PUSHED, issuer=meridian.issuer)
    every = float(EXPECTATIONS[SEAM].backstop_seconds or 0)
    backstop = Backstop(queue, lambda: {_THREAD: thread(_THREAD)}, nodes[SEAM], every)
    found: dict[float, dict[str, str]] = {}
    while clock() <= T0 + 20 * MINUTE:  # ticks at 0, 5 ... 20: nothing moved
        if woken := await backstop.tick(clock()):
            found[clock()] = woken
        clock.advance(MINUTE)
    meridian.bell.down = True  # 10:20, a Claims deploy: the receiver answers 503
    await meridian.assessor_finishes()  # Northgate asks for a fresh chain, once
    lost = len(meridian.bell.attempts)
    await asyncio.sleep(0.05)
    assert len(meridian.bell.attempts) == lost  # one attempt per push, no retry
    meridian.bell.down = False
    while not found:
        if woken := await backstop.tick(clock()):
            found[clock()] = woken
        clock.advance(MINUTE)
    [(when, woken)] = found.items()
    assert woken == {_THREAD: "done"}  # read auth-required, renewed, read completed
    assert when == T0 + 25 * MINUTE  # the next tick after the lost push at 10:21
    assert when - (T0 + 21 * MINUTE) <= every  # at most one cadence late
    assert (await meridian.entry(_THREAD))["state"] == "completed"


# A stream is a view, not a record ----------------------------------------------------

async def test_a_dropped_stream_leaves_the_task_running() -> None:
    risk = RiskPartner(release="2026.12", lookups=asyncio.Event())
    card = risk.card
    card.capabilities.CopyFrom(AgentCapabilities(streaming=True))  # a streaming Risk
    store = InMemoryTaskStore()
    handler = DefaultRequestHandler(agent_executor=risk.executor, task_store=store,
                                    agent_card=card)
    message = new_data_message(dict(clean_claim()), role=Role.ROLE_USER)
    message.metadata.update({"skill": "screen_fraud"})
    context = ServerCallContext()
    stream = handler.on_message_send_stream(SendMessageRequest(message=message),
                                            context)
    first = await anext(stream)
    assert isinstance(first, Task)  # the task id arrives before anything can be lost
    await stream.aclose()  # the caller's connection drops mid-screening
    again = handler.on_subscribe_to_task(SubscribeToTaskRequest(id=first.id), context)
    assert isinstance(await anext(again), Task)  # a subscribe starts with the task
    await again.aclose()
    await risk.finish_lookups()
    task = await handler.on_get_task(GetTaskRequest(id=first.id), context)
    assert task is not None and task.status.state == TaskState.TASK_STATE_COMPLETED
    assert len(risk.executor.received) == 1  # read, never sent again


# Northgate's revocation doorbell ------------------------------------------------------

async def test_the_revocation_doorbell_pulls_now_and_a_forged_one_revokes_nobody(
        ) -> None:
    fed = DrillFederation()
    await fed.connect()
    chain, http = fed.held_chains()["Northgate"], fed.probes()["Northgate"]
    northgate = fed.partners.northgate
    source = next(iter(northgate.sources.values()))
    fed.clock.t = T0 + 3 * MINUTE
    assert await probe(http, chain) == ""  # Northgate pulls list 1 and accepts
    fed.issuer.revoke(ADJUSTER, T0 + 3 * MINUTE + 10)  # list 2 is signed at 10:03:10
    fed.clock.t = T0 + 3 * MINUTE + 20
    assert await probe(http, chain) == ""  # a 20-second-old copy: still a zombie
    bell = httpx.AsyncClient(transport=httpx.ASGITransport(app=northgate.app),
                             base_url=NORTHGATE_URL)
    assert (await bell.post(DOORBELL_PATH, content=b"anything")).status_code == 202
    assert "revoked" in await probe(http, chain)  # pulled list 2, refused at once
    pulls = len(source.pulls)
    for _ in range(50):  # a flood of forged rings inside one gap
        await bell.post(DOORBELL_PATH)
    assert len(source.pulls) == pulls  # costs nothing until the gap passes
    fed.clock.advance(source.gap)
    await bell.post(DOORBELL_PATH, content=b'{"serial": 0, "subjects": {}}')
    assert source.latest is not None and source.latest.serial == 2  # the real list
    assert "revoked" in await probe(http, chain)


# Finance's inbox and the push sender -------------------------------------------------

async def test_a_task_store_write_outside_a_request_pushes_only_if_told_to() -> None:
    sent: list[str] = []

    class Recorder:
        async def send_notification(self, task_id: str, event: Any) -> None:
            sent.append(type(event).__name__)

    from a2a.server.tasks import TaskUpdater
    for push in (None, Recorder()):
        store = InMemoryTaskStore()
        writer = StoreWriter(store, "t-1", "c-1", push)  # type: ignore[arg-type]
        updater = TaskUpdater(writer, "t-1", "c-1")
        await writer.enqueue_event(Task(id="t-1", context_id="c-1"))
        await updater.complete()
    assert sent == ["Task", "TaskStatusUpdateEvent"]  # only the second writer rang


# The delivery-mode matrix -------------------------------------------------------------

async def test_every_meridian_seam_has_a_mode_and_a_backstop() -> None:
    fed = Federation()
    await fed.graph()
    cards = {seam: conn.card for seam, conn in fed.connections.items()}
    assert uncovered(EXPECTATIONS, cards) == []
    assert {e.delivery for e in EXPECTATIONS.values()} <= set(MODES)
    silent = {**EXPECTATIONS, SEAM: _without_backstop()}
    assert uncovered(silent, cards) == [f"{SEAM}: push with no backstop_seconds"]


def _without_backstop() -> Any:
    from dataclasses import replace
    return replace(EXPECTATIONS[SEAM], backstop_seconds=None)


def test_the_push_seams_return_once_the_task_exists() -> None:
    seams = push_seams()
    assert seams[SEAM].return_immediately and seams["payments"].return_immediately
    assert not seams["fraud_screening"].return_immediately
    assert not SEAMS[SEAM].return_immediately  # the earlier chapters' blocking send
