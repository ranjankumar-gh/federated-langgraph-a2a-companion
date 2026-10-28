"""The delivery-mode matrix: how a caller hears that a long-running task moved.

There are three ways, and each loses updates in its own way. A caller that holds the
connection (a stream, or a blocking send, which is a stream of one answer) loses
everything after a disconnect, and sees the break. A push is lost silently. A poll
loses nothing but time, and costs the partner a read per task per tick. Every row names
the backstop that covers its loss, and every seam's expectations name its row.

The partner's terms say nothing here: how Claims hears is Claims' choice. The partner's
card says only whether it can push at all.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from a2a.types import AgentCard

from seamkit.contract import Expectations


@dataclass(frozen=True)
class Mode:
    """One row of the matrix."""

    holds: str  # who holds the connection while the task runs
    loses: str  # what is lost, and when
    noticed: str  # how the caller notices the loss
    partner_cost: str  # what the mode costs the partner per task
    backstop: str  # what covers the loss


MODES: dict[str, Mode] = {
    "stream": Mode("the caller, for the whole run", "every event after a disconnect",
                   "the connection breaks", "one open connection per task",
                   "the next run reads the task the ledger recorded"),
    "push": Mode("nobody between events", "a push the receiver missed",
                 "never, without a poll", "one POST per event",
                 "a backstop poll every backstop_seconds"),
    "poll": Mode("nobody", "nothing but time", "at most one cadence late",
                 "one read per task per tick", "the poll itself"),
}


def uncovered(expects: Mapping[str, Expectations],
              cards: Mapping[str, AgentCard]) -> list[str]:
    """Every seam with no mode, or a mode whose loss nothing covers."""
    found: list[str] = []
    for seam, e in expects.items():
        if e.delivery not in MODES:
            found.append(f"{seam}: no delivery mode ({e.delivery!r})")
            continue
        polled = e.delivery in ("push", "poll")
        if polled and e.backstop_seconds is None:
            found.append(f"{seam}: {e.delivery} with no backstop_seconds")
        if (polled and e.backstop_seconds is not None and e.wait_seconds is not None
                and e.backstop_seconds >= e.wait_seconds):
            found.append(f"{seam}: a backstop of {e.backstop_seconds} s can miss the "
                         f"{e.wait_seconds} s wait horizon")
        card = cards.get(seam)
        if e.delivery == "push" and card and not card.capabilities.push_notifications:
            found.append(f"{seam}: push, but the partner's card cannot push")
    return found
