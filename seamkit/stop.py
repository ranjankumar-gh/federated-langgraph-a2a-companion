"""The partner stop: a person's decision to stop calling one partner.

A caller cannot kill a partner's agent. It can stop opening work there, end its own open
tasks, and undo what it already asked for. A stop does those three things, in that
order, and nothing else:

- it pins every breaker on the stopped owner's forward seams open, with no trial, so no
  send that opens work leaves for that owner until a person releases the stop;
- it reads each thread's ledger, as the backstop poll does, and rings every thread with
  an open task at that owner (or, for a stop that unwinds, any task there at all);
- each rung thread's own run ends its task and takes the seam's degraded mode, or
  aborts, and the compensator undoes what the saga plan says to undo.

The stop never writes a ledger: one writer per thread (Chapter 9) holds under a stop as
it does under a push. It closes the forward door and leaves the undo door open: reads,
cancels, and the compensating skills have no breaker the stop pins, unless a deployment
wires one, which is the mistake the stop drill keeps as a test.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from seamkit.breaker import SeamBreaker
from seamkit.ledger import TERMINAL
from seamkit.push import WakeQueue


@dataclass(frozen=True)
class Stop:
    """Who stopped which owner, when, why, and whether committed work is undone."""

    owner: str
    reason: str
    by: str  # the person on call the seam inventory names for this owner's seams
    at: float
    unwind: bool = False  # undo committed work at the owner on undecided claims

    @property
    def name(self) -> str:
        return f"{self.owner} stopped by {self.by} at {self.at:.0f}: {self.reason}"


@dataclass
class PartnerStops:
    """The stops in force, over the breakers of the caller's process."""

    breakers: Mapping[str, SeamBreaker]  # seam -> its breaker, forward seams only
    stopped: dict[str, Stop] = field(default_factory=dict)  # owner -> its stop

    async def stop(self, stop: Stop, queue: WakeQueue,
                   threads: Iterable[str]) -> list[str]:
        """Close the forward door to one owner; ring every thread with work there."""
        self.stopped[stop.owner] = stop
        seams = [s for s, b in self.breakers.items() if b.owner == stop.owner]
        for seam in seams:  # no send opens work at the owner from here on
            self.breakers[seam].pin(stop.name, stop.at)
        rung = []
        for thread in threads:  # read each ledger as the backstop does; never write
            for seam in seams:
                entry = await queue.entry(thread, seam)
                if entry and (stop.unwind or entry["state"] not in TERMINAL):
                    queue.ring(thread)  # the thread's own run ends its own task
                    rung.append(thread)
                    break
        return rung

    def release(self, owner: str, now: float) -> None:
        """A person ends the stop: the next send on each trial skill tests the owner."""
        self.stopped.pop(owner, None)
        for breaker in self.breakers.values():
            if breaker.owner == owner:
                breaker.unpin(now)

    def unwinding(self, owner: str) -> Stop | None:
        stop = self.stopped.get(owner)
        return stop if stop and stop.unwind else None


__all__ = ["PartnerStops", "Stop"]
