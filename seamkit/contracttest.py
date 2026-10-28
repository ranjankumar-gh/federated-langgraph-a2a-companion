"""Seam contract tests: the caller's expectations, run against any deployment.

A contract test is written by the caller, from its expectations and its recordings,
and it asserts two things about an answer: its outcome class, and the type, unit and
range of every field the caller reads. Never the value. A model-backed partner does not
answer the same way twice, and a legitimate price change is not a broken contract.

The same file runs in two places. The caller runs it against its stub partner, which
proves only that the stub still answers as the recordings say. The partner runs it
against its own release candidate, before release, and that run is the only test that
ever sees the partner's next release. `verify` needs the suite file and this module:
none of the caller's graph, state or credentials.

`conformance` is the short list of protocol behaviours the caller's code relies on. It
is not the protocol's own conformance kit, which tests the wire and not this seam.
"""

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from a2a.client import A2ACardResolver, A2AClientError, Client, ClientCallContext
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskPushNotificationConfig,
    TaskState,
)
from a2a.utils.errors import A2AError

from seamkit.contract import Expectations, Field, violations

if TYPE_CHECKING:
    from stub.recording import Recording

# Each task state as the caller's code reads it: whose fault, and what follows.
CLASSES = {"completed": "answered", "input-required": "asks", "rejected": "rejected",
           "auth-required": "asks authority", "failed": "failed",
           "canceled": "failed"}


def outcome_of(task: Task) -> str:
    """A task's outcome class: "answered", "asks <field>", "rejected", "failed"."""
    state = TaskState.Name(task.status.state).removeprefix("TASK_STATE_")
    found = CLASSES.get(state.lower().replace("_", "-"), "malformed")
    if found != "asks":
        return found
    said = task.status.message.parts
    parts = get_data_parts(said) if said else []
    return f"asks {parts[0].get('needs', '')}" if parts else "asks"


@dataclass(frozen=True)
class ContractTest:
    """One recorded request, the outcome class Claims depends on, and what it reads."""

    name: str
    skill: str
    request: Mapping[str, Any]
    outcome: str
    reads: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Failure:
    test: str
    detail: str


def suite(expects: Expectations,
          recordings: Sequence["Recording"]) -> list[ContractTest]:
    """One test per answer Claims has seen or accepted on the seam's skill.

    A scripted answer is a guess, so it is never asserted against the partner; it is
    on the list of questions to ask the partner instead (`unconfirmed`).
    """
    reads = {k: _plain(f) for k, f in expects.reads.items()}
    tests = []
    for recording in recordings:
        for ex in recording.exchanges:
            if ex.skill != expects.skill or ex.source == "scripted":
                continue
            outcome = CLASSES[ex.state]
            if outcome == "asks":
                outcome = f"asks {ex.message.get('needs', '')}"
            tests.append(ContractTest(
                f"{ex.request.get('claim_id', '?')}: {outcome}", ex.skill,
                dict(ex.request), outcome, reads if outcome == "answered" else {}))
    return tests


def unconfirmed(recordings: Sequence["Recording"],
                scripted: Sequence[Any]) -> list[str]:
    """The outcomes Claims tests only against its own guess: ask the partner."""
    seen = {(e.skill, e.state) for r in recordings for e in r.exchanges}
    return sorted({f"{e.skill}: {e.state}" for e in scripted
                   if (e.skill, e.state) not in seen})


def _plain(f: Field) -> dict[str, Any]:
    return {k: v for k, v in asdict(f).items() if k != "name" and v not in ("", None)}


def export(seam: str, tests: Sequence[ContractTest], path: Path) -> None:
    """The file Claims publishes to the partner: its expectations, as tests."""
    body = {"seam": seam, "tests": [asdict(t) for t in tests]}
    path.write_text(dumps(body) + "\n", encoding="utf-8")


def dumps(body: Mapping[str, Any]) -> str:
    """Indented JSON with each list of plain values on one line, for a reviewer."""
    text = json.dumps(body, indent=2)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]",
                  lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)


def load_suite(path: Path) -> list[ContractTest]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ContractTest(**t) for t in raw["tests"]]


async def client_for(http: httpx.AsyncClient, url: str) -> Client:
    card = await A2ACardResolver(http, url).get_agent_card()
    return ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)


async def send(client: Client, skill: str, data: Mapping[str, Any],
               key: str | None = None, task: Task | None = None,
               push: TaskPushNotificationConfig | None = None) -> Task:
    message = new_data_message(dict(data), role=Role.ROLE_USER,
                               context_id=task.context_id if task else None,
                               task_id=task.id if task else None)
    message.message_id = key or str(uuid.uuid4())
    message.metadata.update({"skill": skill})
    request = SendMessageRequest(
        message=message,
        configuration=SendMessageConfiguration(task_push_notification_config=push))
    got = Task()
    async for event in client.send_message(request,
                                           context=ClientCallContext(timeout=10)):
        if event.HasField("task"):
            got = event.task
    return got


async def verify(http: httpx.AsyncClient, url: str,
                 tests: Sequence[ContractTest]) -> list[Failure]:
    """Run a suite against one deployment of the partner: what did it get wrong?"""
    client, failures = await client_for(http, url), []
    for t in tests:
        task = await send(client, t.skill, t.request)
        if (got := outcome_of(task)) != t.outcome:
            failures.append(Failure(t.name, f"expected {t.outcome}, got {got}"))
            continue
        answer = {k: v for a in task.artifacts for p in get_data_parts(a.parts)
                  for k, v in p.items()}
        shapes = {k: Field(k, **spec) for k, spec in t.reads.items()}
        wrong = [f"lacks {k}" for k in shapes if k not in answer]
        if wrong := wrong + violations(answer, shapes):
            failures.append(Failure(t.name, "; ".join(wrong)))
    return failures


async def conformance(http: httpx.AsyncClient, url: str, skill: str,
                      answered: Mapping[str, Any],
                      asks: tuple[Mapping[str, Any], Mapping[str, Any]],
                      push_url: str = "") -> list[str]:
    """The protocol behaviours Claims' code relies on, checked on one deployment.

    `answered` is a request the partner completes, and `asks` a request it pauses on
    with the reply that resumes it. Returns what did not hold; [] is conformant.
    """
    broken: list[str] = []
    try:
        client = await client_for(http, url)
    except (A2AClientError, A2AError, httpx.HTTPError) as exc:
        return [f"no card at its well-known path: {type(exc).__name__}"]
    done = await send(client, skill, answered, key="conformance-1")
    if outcome_of(done) != "answered":
        broken.append(f"send: {outcome_of(done)}, not answered")
    read = await client.get_task(GetTaskRequest(id=done.id))
    if (read.id, read.status.state) != (done.id, done.status.state):
        broken.append("read: the task read back is not the task sent")
    try:
        await client.cancel_task(CancelTaskRequest(id=done.id))
        broken.append("cancel: a finished task was cancelled")
    except (A2AClientError, A2AError):
        pass  # refused, as Chapter 13's compensator relies on
    paused = await send(client, skill, asks[0])
    resumed = await send(client, skill, asks[1], task=paused)
    if not (outcome_of(paused).startswith("asks") and resumed.id == paused.id
            and outcome_of(resumed) == "answered"):
        broken.append("resume: a reply to a paused task did not finish that task")
    again = await send(client, skill, answered, key="conformance-1")
    if again.id != done.id:
        broken.append("key: a repeat inside the window opened a second task")
    if push_url:
        config = TaskPushNotificationConfig(url=push_url, token="conformance")
        if (await send(client, skill, answered, push=config)).id == "":
            broken.append("push: a send with a push config was not accepted")
    return broken


__all__ = ["CLASSES", "ContractTest", "Failure", "conformance", "export",
           "load_suite", "outcome_of", "suite", "unconfirmed", "verify"]
