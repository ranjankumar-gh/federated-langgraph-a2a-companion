"""Recordings: what a partner answered to requests the caller sent, and what they were
made under.

An exchange is one send and the task state it ended in. Its `source` says how the
caller knows the answer:

- "recorded": seen from the partner, sent to its staging deployment with a synthetic
  claim (never copied from production, which would put customer data in a repository);
- "canary": a golden task with the answer a person accepted (Chapter 15);
- "scripted": allowed by the partner's terms and never seen, written by the caller.
  A scripted answer is a guess, labelled as one, and it is a question for the partner.

A recording names the card digest, the terms revision and the behaviour version it was
made under. When any of them moves, the recording is stale: it describes a partner that
no longer exists.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from seamkit.canary import Baseline, Golden
from seamkit.cards import card_digest
from seamkit.contract import Expectations
from seamkit.contracttest import dumps
from seamkit.remote import PartnerConnection

SOURCES = frozenset({"recorded", "canary", "scripted"})
CARD_PATH = "/.well-known/agent-card.json"


@dataclass(frozen=True)
class Exchange:
    """One send to a partner skill, and the state and answer its task ended with."""

    skill: str
    request: Mapping[str, Any]  # the data Claims sent, and nothing else
    state: str  # the task state it ended in: "completed", "rejected", ...
    artifact: Mapping[str, Any]  # a completed task's answer; {} for any other state
    source: str  # one of SOURCES
    message: Mapping[str, Any] = field(default_factory=dict)  # a question or a reason

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source {self.source!r} is not one of {sorted(SOURCES)}")


@dataclass(frozen=True)
class Recording:
    """One seam's exchanges, with what the partner was when they were made."""

    seam: str
    release: str  # the card's version field: recorded, never acted on
    revision: str  # the terms revision in force
    card_digest: str
    behaviour: str  # the behaviour version Claims had accepted (Chapter 15)
    made_at: str  # the day the recordings were made, ISO
    exchanges: tuple[Exchange, ...]
    card: Mapping[str, Any]  # the card as fetched, without its signatures


def save(recording: Recording, path: Path) -> None:
    body = {"seam": recording.seam, "release": recording.release,
            "revision": recording.revision, "card_digest": recording.card_digest,
            "behaviour": recording.behaviour, "made_at": recording.made_at,
            "exchanges": [_plain(e) for e in recording.exchanges],
            "card": recording.card}
    path.write_text(dumps(body) + "\n", encoding="utf-8")


def load(path: Path) -> Recording:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Recording(raw["seam"], raw["release"], raw["revision"], raw["card_digest"],
                     raw["behaviour"], raw["made_at"],
                     tuple(Exchange(**e) for e in raw["exchanges"]), raw["card"])


def _plain(exchange: Exchange) -> dict[str, Any]:
    return {"skill": exchange.skill, "request": dict(exchange.request),
            "state": exchange.state, "artifact": dict(exchange.artifact),
            "message": dict(exchange.message), "source": exchange.source}


def state_of(wire: str) -> str:
    """"TASK_STATE_INPUT_REQUIRED" as the book writes it: "input-required"."""
    return wire.removeprefix("TASK_STATE_").lower().replace("_", "-")


def _data(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        out.update(part.get("data", {}))
    return out


class Recorder(httpx.AsyncBaseTransport):
    """A transport that keeps what Claims sent and what each task ended as.

    Put it under Claims' client to a partner's staging deployment, send synthetic
    claims, and the exchanges come out in the order they were made. It also keeps the
    card it fetched, which is what a stub serves.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.exchanges: list[Exchange] = []
        self.card: dict[str, Any] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        body = await response.aread()
        if request.method == "GET" and request.url.path == CARD_PATH:
            self.card = {k: v for k, v in json.loads(body).items() if k != "signatures"}
        elif request.method == "POST":
            self._keep(json.loads(request.content), json.loads(body))
        return httpx.Response(response.status_code, headers=response.headers,
                              content=body, request=request)

    def _keep(self, sent: Mapping[str, Any], got: Mapping[str, Any]) -> None:
        if sent.get("method") != "SendMessage" or "task" not in got.get("result", {}):
            return
        message, task = sent["params"]["message"], got["result"]["task"]
        state = state_of(task["status"]["state"])
        artifact = _data([p for a in task.get("artifacts", []) for p in a["parts"]])
        said = _data(task["status"].get("message", {}).get("parts", []))
        self.exchanges.append(Exchange(
            message.get("metadata", {}).get("skill", ""), _data(message["parts"]),
            state, artifact if state == "completed" else {}, "recorded", said))


def from_baseline(goldens: Sequence[Golden], baseline: Baseline,
                  expects: Expectations,
                  answered: Mapping[str, Any] | None = None) -> tuple[Exchange, ...]:
    """Chapter 15's goldens as canary exchanges: each request with the accepted answer.

    The baseline holds the answers in Claims' state fields; an exchange holds them as
    the partner's artifact, so each field goes back through the seam's field map.
    `answered` adds the outputs Claims does not judge, as the partner sent them.
    """
    theirs = {ours: key for key, ours in expects.into.items()}
    return tuple(
        Exchange(expects.skill, dict(g.request), "completed",
                 {**(answered or {}),
                  **{theirs[f]: v for f, v in baseline.answers[g.id].items()}},
                 "canary")
        for g in goldens)


def stale(recording: Recording, connection: PartnerConnection) -> list[str]:
    """What moved since the recording was made; [] while it still describes the
    partner Claims is connected to."""
    revision = connection.terms[0].revision if connection.terms else "none"
    skill = recording.exchanges[0].skill if recording.exchanges else ""
    moved = [("card digest", recording.card_digest, card_digest(connection.card)),
             ("terms revision", recording.revision, revision),
             ("behaviour version", recording.behaviour,
              connection.behaviour.get(skill, recording.behaviour))]
    return [f"{what} {was[:19]} is now {now[:19]}" for what, was, now in moved
            if was != now]


__all__ = ["SOURCES", "Exchange", "Recorder", "Recording", "from_baseline", "load",
           "save", "stale", "state_of"]
