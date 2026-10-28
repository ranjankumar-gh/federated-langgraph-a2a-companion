"""Meridian's data classes: every claim field in one named class, and where each may go.

A field crosses a seam only if its class may cross to that owner (this file, Meridian's
decision) and the partner's terms accept the class (the partner's commitment). A field
with no class here is "unclassified", and no unclassified field crosses anywhere.
"""

from seamkit.contract import CrossingPolicy

DATA_CLASSES: dict[str, str] = {
    "claim_id": "claim", "amount": "claim", "days_to_report": "claim",
    "police_report": "claim", "decision": "claim", "payment_ref": "claim",
    "booking_ref": "claim", "task": "claim",  # Ch 13: a booking, and a task undone
    "damaged_parts": "vehicle", "covered_parts": "vehicle", "vin": "vehicle",
    "photo_url": "vehicle",
    "prior_claims": "claim_history",
    "fraud_score": "fraud_assessment", "fraud_signals": "fraud_assessment",
    "repair_estimate": "estimate", "estimate_notes": "estimate",
    "note_findings": "estimate", "payment_memo": "estimate",
    "policy": "policy",
}

# One crossing rule per partner. Northgate is another company: it sees the claim and the
# car, and never the claimant's history, Meridian's fraud view, or the policy.
CROSSES_TO: dict[str, frozenset[str]] = {
    "Risk": frozenset({"claim", "vehicle", "claim_history"}),
    "Northgate": frozenset({"claim", "vehicle"}),
    "Finance": frozenset({"claim", "estimate", "policy"}),
}

# Chapter 19: where a class may be processed. Claim history and the fraud view are
# Meridian's own: a partner that holds them holds them at home. A class not named here
# may be processed in any region the partner declares.
STAYS_IN: dict[str, frozenset[str]] = {
    "claim_history": frozenset({"home"}), "fraud_assessment": frozenset({"home"})}
# A stricter reading, which a legal review may impose: the car's data stays home too.
STAYS_IN_STRICT: dict[str, frozenset[str]] = {**STAYS_IN,
                                              "vehicle": frozenset({"home"})}

MERIDIAN_POLICY = CrossingPolicy(DATA_CLASSES, CROSSES_TO, STAYS_IN)
STRICT_POLICY = CrossingPolicy(DATA_CLASSES, CROSSES_TO, STAYS_IN_STRICT)
