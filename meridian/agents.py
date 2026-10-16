"""Meridian's agents as plain rules.

Chapter 1 makes no model calls. What breaks at an ownership boundary is ownership, not
model behaviour, so the rules are deterministic and the breakage tests decide the same
way every run. The same functions run in-process and inside each partner's runtime: only
who owns them changes.
"""

from collections.abc import Callable, Mapping
from typing import Any

Claim = Mapping[str, Any]

# Each signal adds 14 points to a base of 40, so three signals score 82: over the Claims
# team's referral threshold of 70.
FRAUD_SIGNALS: dict[str, Callable[[Claim], bool]] = {
    "reported_late": lambda c: int(c["days_to_report"]) > 30,
    "high_amount": lambda c: float(c["amount"]) > 10_000,
    "repeat_claimant": lambda c: int(c["prior_claims"]) >= 2,
}
BASE_SCORE = 40.0
POINTS_PER_SIGNAL = 14.0

PART_PRICES: dict[str, float] = {"bumper": 1_200.0, "door": 1_800.0, "frame": 6_500.0}


def screen_fraud(claim: Claim) -> tuple[float, list[str]]:
    """Risk's fraud screen: a score on a 0-100 scale and the signals behind it."""
    signals = [name for name, fires in FRAUD_SIGNALS.items() if fires(claim)]
    score = min(100.0, BASE_SCORE + POINTS_PER_SIGNAL * len(signals))
    return score, signals


def estimate_repair(claim: Claim) -> float:
    """Northgate's repair estimate: list price for every damaged part."""
    return sum(PART_PRICES[part] for part in claim["damaged_parts"])


def pay(claim: Claim) -> str:
    """Finance's payment: never more than the estimate, never more than the claim."""
    amount = min(float(claim["amount"]), float(claim["repair_estimate"]))
    return f"PAY-{claim['claim_id']}-{amount:.0f}"
