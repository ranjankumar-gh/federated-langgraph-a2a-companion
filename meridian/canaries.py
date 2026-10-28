"""Meridian's behavioural canaries: golden tasks for the skills that commit nothing.

The estimate seam sends Northgate four synthetic claims every six hours, and the
screening seam sends Risk three every hour. Each golden is a claim under a reserved id
(CLM-C...), with no customer data in it, crossing under the same data-class policy as
a real claim. The canary chain's root names the on-call for Claims' seams, and her
caseload is those ids.

Booking and payments get none. `book_repair` and `pay_claim` commit at the partner, so
a golden booking would book a repair and a golden payment would pay; Chapter 4's
`check` refuses a `[canary]` table on either. The peer seam's `estimate_history` is
Risk's to probe, as it is Risk's to call.

The baselines below are the answers the seams' owner accepted when the suites were set
up: Northgate's list prices and Risk's 0-100 scores, which every earlier chapter's
tests also rely on.
"""

from collections.abc import Callable, Mapping

from meridian.federated import EXPECTATIONS
from meridian.identity import CANARY_ONCALL, CANARY_SESSION
from meridian.inprocess import Node
from seamkit.canary import Baseline, CanarySuite, Golden
from seamkit.inventory import SeamInventory
from seamkit.reconcile import Finding, log_finding
from seamkit.remote import PartnerConnection, RemoteAgentNode

ACCEPTED_AT = 1_793_520_000.0  # when the on-call accepted the first baselines

ESTIMATE_GOLDENS = (
    Golden("CLM-C01", {"claim_id": "CLM-C01", "damaged_parts": ["bumper"]},
           ("repair_estimate",)),
    Golden("CLM-C02", {"claim_id": "CLM-C02", "damaged_parts": ["door"]},
           ("repair_estimate",)),
    Golden("CLM-C03", {"claim_id": "CLM-C03", "damaged_parts": ["bumper", "door"]},
           ("repair_estimate",)),
    Golden("CLM-C04", {"claim_id": "CLM-C04", "damaged_parts": ["bumper", "bumper"]},
           ("repair_estimate",)),
)
ESTIMATE_BASELINE = Baseline(
    {"CLM-C01": {"repair_estimate": 1_200.0}, "CLM-C02": {"repair_estimate": 1_800.0},
     "CLM-C03": {"repair_estimate": 3_000.0}, "CLM-C04": {"repair_estimate": 2_400.0}},
    CANARY_ONCALL, ACCEPTED_AT)


def screening(claim_id: str, days: int, prior: int) -> Golden:
    """A synthetic claim under 10,000, so Risk never pauses for a police report."""
    return Golden(claim_id, {"claim_id": claim_id, "amount": 2_000.0,
                             "days_to_report": days, "prior_claims": prior},
                  ("fraud_score",))


SCREENING_GOLDENS = (
    screening("CLM-C11", 2, 0),  # no signal: the base score, 40
    screening("CLM-C12", 45, 0),  # reported late: 54
    screening("CLM-C13", 45, 3),  # reported late by a repeat claimant: 68
)
SCREENING_BASELINE = Baseline(
    {"CLM-C11": {"fraud_score": 40.0}, "CLM-C12": {"fraud_score": 54.0},
     "CLM-C13": {"fraud_score": 68.0}}, CANARY_ONCALL, ACCEPTED_AT)

GOLDENS = {"repair_estimate": (ESTIMATE_GOLDENS, ESTIMATE_BASELINE),
           "fraud_screening": (SCREENING_GOLDENS, SCREENING_BASELINE)}


def canary_suites(nodes: Mapping[str, Node],
                  file: Callable[[Finding], None] = log_finding,
                  ) -> dict[str, CanarySuite]:
    """One suite per seam whose expectations carry a `[canary]` table."""
    suites = {}
    for seam, expects in EXPECTATIONS.items():
        node = nodes.get(seam)
        if expects.canary is None or not isinstance(node, RemoteAgentNode):
            continue
        goldens, baseline = GOLDENS[seam]
        suites[seam] = CanarySuite(seam, expects.canary, goldens, baseline, node,
                                   {"session": CANARY_SESSION}, file)
    return suites


def with_versions(inventory: SeamInventory,
                  connections: Mapping[str, PartnerConnection],
                  suites: Mapping[str, CanarySuite]) -> SeamInventory:
    """Add each seam's versions to its contract answer, after Chapter 4's terms
    entry: the protocol in use, the notice required, and the behaviour version."""
    for entry in inventory.entries:
        conn, expects = connections.get(entry.callee), EXPECTATIONS.get(entry.callee)
        if conn is None or expects is None:
            continue
        suite = suites.get(entry.callee)
        watched = (f"behaviour {conn.behaviour.get(expects.skill, '')}"
                   if suite else "behaviour unwatched")
        if suite and expects.skill in conn.held:
            watched += ", held"
        notice = (f"notice {expects.notice_days} d" if expects.notice_days is not None
                  else "no notice term")
        held = entry.answers.get("contract", "")
        entry.answers["contract"] = (held + f"; protocol {conn.protocol or 'as found'}"
                                     f"; {notice}; {watched}")
    return inventory
