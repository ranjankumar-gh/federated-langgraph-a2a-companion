"""Two sample claims every chapter reuses."""

from meridian.inprocess import ClaimState


def clean_claim() -> ClaimState:
    """A claim no fraud signal fires on: it should be paid."""
    return {"claim_id": "CLM-1001", "amount": 3_000.0, "days_to_report": 2,
            "prior_claims": 0, "damaged_parts": ["bumper", "door"]}


def suspect_claim() -> ClaimState:
    """A claim three signals fire on (score 82): it should be referred, not paid."""
    return {"claim_id": "CLM-1002", "amount": 14_500.0, "days_to_report": 45,
            "prior_claims": 3, "damaged_parts": ["frame", "door"]}
