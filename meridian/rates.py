"""Meridian's internal rate card, as Finance publishes it inside Meridian (Chapter 18).

Risk prices its usage reports by it, Claims prices its ledger by it, and the monthly
chargeback moves Risk's model spend on motor claims to the Claims team's cost centre
at the same rate, so the three cannot disagree about what a token costs.
"""

from pathlib import Path

from seamkit.costs import RateCard, load_rates

RATES: RateCard = load_rates(Path(__file__).parent / "rates.toml")

__all__ = ["RATES"]
