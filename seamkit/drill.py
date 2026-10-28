"""The delegation revocation drill's instruments: a clock, a probe, and the report row.

The drill revokes a person mid-task on a running federation and reports, for each owner
on her chains, when it first refused her authority, whether it acted after the horizon
its seam contract allows, and how every open task under her chains ended. The scenario
is the federation's own (meridian/drill.py replays Chapter 7's incident); these pieces
are what any federation's drill needs.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from seamkit.delegation import HEADER, Chain, encode


@dataclass
class Clock:
    """The one clock every owner reads in a drill, so minutes can pass in a test."""

    t: float

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass(frozen=True)
class DrillRow:
    """One owner's line in the drill's report. Times are seconds after the revocation."""

    owner: str
    first_refusal: float | None  # None: it never refused her authority again
    refused_with: str  # the reason the owner gave
    acted_after: tuple[str, ...]  # acts it accepted after the horizon it agreed
    tasks: Mapping[str, str] = field(default_factory=dict)  # open task -> how it ended
    agreed: float | None = None  # the horizon its seam contract states; None: no term

    @property
    def observed(self) -> float | None:
        """The horizon the drill saw: how long a revoked chain still worked there."""
        return self.first_refusal


async def probe(http: httpx.AsyncClient, chain: Chain) -> str:
    """Present `chain` once, as a read of a task that does not exist.

    Returns the guard's refusal, or "" if the chain was accepted: a read that reaches
    the A2A handler is answered "task not found", which is the guard saying yes.
    """
    body = {"jsonrpc": "2.0", "id": "probe", "method": "GetTask",
            "params": {"id": "drill-probe"}}
    response = await http.post("/", content=json.dumps(body),
                               headers={HEADER: encode(chain), "A2A-Version": "1.0",
                                        "content-type": "application/json"})
    if response.status_code != 403:
        return ""
    return str(response.json()["error"]["message"]).removeprefix("delegation refused: ")
