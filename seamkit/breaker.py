"""The seam breaker: a circuit breaker the caller keeps for one partner skill.

Across an owner line the caller sees only what the wire keeps: a task state and a
message. So the breaker counts only the partner's failures to answer: no answer inside
the seam's timeout (PartnerUnavailable), a task that ended failed or canceled on the
partner's side (PartnerFailed), and an answer later than the seam's late threshold. A
rejection, a breach, an untrusted card or a malformed answer is an answer, or Claims'
own policy, and other mechanisms already hold the seam for it.

The breaker lives in the caller's process, per seam, never in a checkpoint: it is about
the partner, not about any one claim. While it is open the seam opens no new work there
and resends nothing. After `open_seconds` one real task may test the partner (the trial),
and only on the skill the policy names. A breaker whose own skill never trials follows
the breaker of the skill that does. A partner stop pins it open until a person releases
it. The in-flight cap bounds how many sends may wait on the partner at once.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from seamkit.errors import PartnerFailed, PartnerUnavailable, SeamError

COUNTED: tuple[type[SeamError], ...] = (PartnerUnavailable, PartnerFailed)
MODES = frozenset({"hold", "substitute", "abort"})


@dataclass(frozen=True)
class BreakerPolicy:
    """One seam's breaker, as its expectations file states it."""

    late_seconds: float  # an answer later than this counts against the partner
    failures: int  # counted outcomes inside window_seconds that open the breaker
    window_seconds: float
    open_seconds: float  # how long it stays open before one trial task
    trial: str  # the skill whose next real task may test the partner; "" never
    max_in_flight: int  # sends on this seam that may wait on the partner at once
    mode: str  # what a claim that needs the seam does while it is open: MODES


@dataclass
class SeamBreaker:
    """The breaker for one seam's skill, in the caller's process."""

    seam: str
    owner: str
    skill: str
    policy: BreakerPolicy
    follows: "SeamBreaker | None" = None  # the breaker whose trial decides this one
    state: str = "closed"  # "closed", "open" or "half-open"
    opened: float = 0.0
    counted: list[float] = field(default_factory=list)  # when each counted outcome came
    in_flight: int = 0
    trial_out: bool = False
    pinned: str = ""  # the partner stop holding it open; "" if none
    moves: list[tuple[float, str]] = field(default_factory=list)  # (when, new state)

    def refuses(self, now: float) -> str:
        """Why a send that opens work may not leave now, or "" if it may. No effects."""
        if self.pinned:
            return f"stopped: {self.pinned}"
        if self.state == "open" and now - self.opened < self.policy.open_seconds:
            return f"open for {self.opened + self.policy.open_seconds - now:.0f} s more"
        if self.follows is not None and self.follows.state != "closed":
            return f"waits on {self.policy.trial} to close"  # the leader's evidence
        if self.state != "closed" and self.policy.trial != self.skill:
            if self.follows is None:
                return "waits on a person to close"
        elif self.state != "closed" and self.trial_out:
            return "half-open, its one trial in flight"
        if self.in_flight >= self.policy.max_in_flight:
            return f"{self.in_flight} tasks already in flight"
        return ""

    def allow(self, now: float) -> str:
        """Admit one send, or say why not: the check before anything crosses."""
        if why := self.refuses(now):
            return why
        if self.state != "closed" and self.policy.trial != self.skill:
            self.move(now, "closed")  # the skill it follows has passed its trial
        elif self.state != "closed":  # this send is the trial: somebody's real task
            self.move(now, "half-open")
            self.trial_out = True
        self.in_flight += 1
        return ""

    def record(self, error: type[SeamError] | None, elapsed: float, now: float) -> None:
        """The outcome of a send `allow` admitted: an answer, or the error it raised."""
        late = error is None and elapsed > self.policy.late_seconds
        self.observe(late or (error is not None and issubclass(error, COUNTED)), now)

    def failed_later(self, now: float) -> None:
        """A task whose send returned before it finished has ended failed: count it.

        Chapter 16: on a seam whose send returns once the task exists, the failure is
        read, never sent, and `record` only ever saw an answer in time.
        """
        self.in_flight += 1  # the send's own outcome already took its place back
        self.observe(True, now)

    def observe(self, failed: bool, now: float) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if self.trial_out:  # the trial decides, either way
            self.trial_out = False
            self.move(now, "open" if failed else "closed")
            return
        if not failed:
            return
        self.counted = [t for t in self.counted if now - t < self.policy.window_seconds]
        self.counted.append(now)
        if self.state == "closed" and len(self.counted) >= self.policy.failures:
            self.move(now, "open")

    def move(self, now: float, state: str) -> None:
        if state == "open":
            self.opened = now
        if state == "closed":
            self.counted = []
        self.state = state
        self.moves.append((now, state))

    def pin(self, stop: str, now: float) -> None:
        """A partner stop: open, no trial, until a person releases it."""
        self.pinned = stop
        self.move(now, "open")

    def unpin(self, now: float) -> None:
        """Released: the next send on the trial skill tests the partner at once."""
        self.pinned = ""
        self.opened = now - self.policy.open_seconds


def breakers_for(policies: Mapping[str, tuple[str, str, BreakerPolicy]],
                 ) -> dict[str, SeamBreaker]:
    """One breaker per seam (seam -> owner, skill, policy); followers wired to leaders."""
    out = {seam: SeamBreaker(seam, owner, skill, policy)
           for seam, (owner, skill, policy) in policies.items()}
    for breaker in out.values():
        if breaker.policy.trial and breaker.policy.trial != breaker.skill:
            breaker.follows = next(
                (b for b in out.values() if b.owner == breaker.owner
                 and b.skill == breaker.policy.trial), None)
    return out


__all__ = ["COUNTED", "MODES", "BreakerPolicy", "SeamBreaker", "breakers_for"]
