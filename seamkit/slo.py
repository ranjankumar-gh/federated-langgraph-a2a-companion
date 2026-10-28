"""Seam SLOs and error budgets: a promise one team makes another, measured by the caller.

A seam's SLO is a target share of sends answered well inside a window, on the clock
every owner reads: answered, not failed, and inside the seconds the partner promised.
The error budget is what the target leaves: at 95% over 28 days, five sends in every
hundred may go wrong before the partner has broken its promise.

The caller measures it from its own crossing records, and only sends that reached the
partner count. A send Claims refused before it left (a breaker, an untrusted card, a
breached contract, a held skill) is Claims' own protection at work, and it spends
nothing: a breaker is not an SLO (Chapter 14).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seamkit.remote import CrossingRecord

# Sends Claims refused itself: nothing reached the partner, so nothing is spent.
CLAIMS_REFUSED = frozenset({"BreakerOpen", "CardUntrusted", "ContractBreach",
                            "BehaviourDrift", "DelegationRefused", "UntrustedEgress",
                            "BudgetSpent"})  # Chapter 18: Claims stopped paying


@dataclass(frozen=True)
class SeamSLO:
    """One seam's objective, from its expectations file's [slo] table."""

    target: float  # the share of sends that must be answered well
    seconds: float  # inside this long; the partner's promised p95
    window_seconds: float  # measured over this window, on the shared clock
    alert: float = 0.5  # the share of the budget spent that raises a signal


@dataclass(frozen=True)
class Budget:
    """How much of one seam's error budget the window has spent."""

    seam: str
    sends: int
    bad: int
    allowed: float  # bad sends the target allows in this window

    @property
    def spent(self) -> float:
        return self.bad / self.allowed if self.allowed else float(self.bad > 0)

    def __str__(self) -> str:
        return (f"{self.seam}: {self.bad} of {self.sends} sends missed the SLO, "
                f"{self.spent:.0%} of the error budget")


def burn(seam: str, slo: SeamSLO, records: Sequence["CrossingRecord"],
         now: float) -> Budget:
    """The window's error budget, spent by sends the partner answered badly."""
    sent = [r for r in records if r.seam == seam and r.call == "send"
            and now - r.at <= slo.window_seconds and r.error not in CLAIMS_REFUSED]
    bad = sum(1 for r in sent if r.error or r.state == "TASK_STATE_FAILED"
              or (r.seconds or 0.0) > slo.seconds)
    return Budget(seam, len(sent), bad, (1 - slo.target) * len(sent))


def slo_of(raw: dict[str, Any]) -> SeamSLO:
    """An expectations file's [slo] table."""
    return SeamSLO(float(raw["target"]), float(raw["seconds"]),
                   float(raw["window_days"]) * 86_400, float(raw.get("alert", 0.5)))


__all__ = ["CLAIMS_REFUSED", "Budget", "SeamSLO", "burn", "slo_of"]
