"""The partner stop drill: stop one partner on a running federation, and report.

For every thread that had work at the stopped owner, the report says how each of its
tasks there ended, which degraded mode the claim took, and how the compensator settled
each step. `forward_sends` is the bound the stop promises: after the stop, no send
that opens work leaves for that owner. Compensations and cancels may, because the stop
leaves the undo door open. The clock is the one Chapter 7's drill introduced (`Clock`
in seamkit.drill), so hours can pass in a test.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from meridian.saga import SagaGraph
from seamkit.remote import CrossingRecord


@dataclass(frozen=True)
class StopRow:
    """One thread's line in the stop drill's report."""

    thread: str
    tasks: Mapping[str, str]  # seam at the stopped owner -> how its task ended
    modes: Mapping[str, str]  # seam -> the degraded mode the claim took
    undone: Mapping[str, str]  # step -> how the compensator settled it


async def stop_report(graph: SagaGraph, threads: Mapping[str, RunnableConfig],
                      seams: Iterable[str]) -> list[StopRow]:
    """Read each thread's checkpoint after the stop: read only, as the backstop does."""
    rows, owned = [], list(seams)
    for tid, config in threads.items():
        values = (await graph.aget_state(config)).values
        ledger = values.get("ledger", {})
        tasks = {seam: entry["state"] + (" (by the stop)" if entry.get("stopped_by")
                                         else "")
                 for seam in owned if (entry := ledger.get(seam))}
        rows.append(StopRow(tid, tasks, dict(values.get("degraded", {})),
                            dict(values.get("undone", {}))))
    return rows


def forward_sends(records: Sequence[CrossingRecord], owner: str) -> list[CrossingRecord]:
    """Sends to `owner` that open work: every send but a compensation's."""
    return [r for r in records if r.partner == owner and r.call == "send"
            and not r.seam.endswith(".undo") and r.error != "BreakerOpen"]


__all__ = ["StopRow", "forward_sends", "stop_report"]
