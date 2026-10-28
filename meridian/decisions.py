"""Meridian under the Federation Test: owner facts, five candidate calls, five records.

The facts are per owner and the answers are derived from them. The call list is the
design: the three hub seams the Claims graph routes, the one call Risk's screening may
make to Northgate (the peer seam), and the incident's policy lookup.

Run `python -m meridian.decisions` to rewrite docs/decisions/; a test fails if the
committed records drift from what the facts produce.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from meridian.federated import OWNERS_A2A, SEAMS
from meridian.inprocess import Node
from meridian.policy_server import POLICY_SEAM
from seamkit.federation import Candidate, DecisionRecord, OwnerFacts, federation_test
from seamkit.remote import RemoteAgentNode

FACTS = {
    "Claims": OwnerFacts("Claims", "claims-train", "meridian-idp", "meridian-claims"),
    "Risk": OwnerFacts("Risk", "risk-monthly", "meridian-idp", "meridian-risk"),
    "Finance": OwnerFacts("Finance", "finance-train", "meridian-idp",
                          "meridian-payments"),
    "Northgate": OwnerFacts("Northgate", "northgate", "northgate-idp", "northgate"),
}
# Owners of every node the five calls touch: the hub seams' from the seams themselves.
OWNERS = {**OWNERS_A2A, "estimate_history": "Northgate", "policy_lookup": "Claims"}

# The incident's deployment gave the policy lookup a pipeline of its own. That is a fact
# about one deployment, not about an owner, so it is recorded on the callee only.
POLICY_PIPELINE = "claims-policy-pipeline"


def candidate(caller: str, callee: str, stateful: bool, model_decides: bool,
              facts: Mapping[str, OwnerFacts] = FACTS,
              callee_facts: OwnerFacts | None = None) -> Candidate:
    return Candidate(caller, callee, facts[OWNERS[caller]],
                     callee_facts or facts[OWNERS[callee]], stateful, model_decides)


def meridian_candidates(facts: Mapping[str, OwnerFacts] = FACTS) -> list[Candidate]:
    return [
        candidate("claims_intake", "fraud_screening", True, False, facts),
        candidate("claims_intake", "repair_estimate", True, False, facts),
        candidate("claims_intake", "payments", True, False, facts),
        candidate("fraud_screening", "estimate_history", True, True, facts),
        candidate("claims_intake", "policy_lookup", True, False, facts,
                  callee_facts=replace(facts["Claims"], cadence=POLICY_PIPELINE)),
    ]


# What each call re-sends per task. The hub seams' fields come from their SeamSpecs; the
# policy lookup's bytes were measured on the incident build (tests/test_ch02_costs.py).
SENDS = {**{name: spec.sends for name, spec in SEAMS.items()},
         "estimate_history": ("claim_id",), "policy_lookup": POLICY_SEAM.sends}
SHAPE = {"fraud_screening": "hub: Claims routes it", "repair_estimate":
         "hub: Claims routes it", "payments": "hub: Claims routes it",
         "estimate_history": "peer: Risk's screening calls Northgate directly; "
                             "the task is in Risk's record, not Claims'",
         "policy_lookup": "subgraph inside the Claims graph"}
PAGED = {"fraud_screening": "risk-oncall", "repair_estimate": "northgate-support, via "
         "the partner manager", "payments": "finance-oncall",
         "estimate_history": "risk-oncall", "policy_lookup": "claims-oncall"}
REVISIT = {
    "fraud_screening": "Risk moves onto the Claims release train, or Claims takes over "
                       "the fraud model.",
    "repair_estimate": "Meridian buys Northgate or brings estimating in-house.",
    "payments": "Finance and Claims merge their release trains and data boundary.",
    "estimate_history": "Risk agrees to share its fraud signals with Claims (route it "
                        "through the hub), or a second peer seam is proposed.",
    "policy_lookup": "another owner calls the policy lookup, or the policy "
                     "administration team takes it over.",
}


def costs(callee: str) -> dict[str, str]:
    sends = SENDS[callee]
    count = f"{len(sends)} field{'s' if len(sends) > 1 else ''}"
    out = {"re-sent per task": f"{count} ({', '.join(sends)})"}
    if callee == "policy_lookup":
        out |= {"latency": "a round trip per claim instead of a function call",
                "request body (CLM-1001)": "310 bytes",
                "lost from the Claims checkpoint": "the policy revision read, and 4 "
                "subgraph checkpoints"}
    else:
        out |= {"latency": "a queue we do not run (per-seam p99, Ch 14)",
                "checkpoint visibility": f"{OWNERS[callee]}'s steps stay in "
                f"{OWNERS[callee]}'s checkpoint"}
    return out


def records(facts: Mapping[str, OwnerFacts] = FACTS) -> list[DecisionRecord]:
    return [DecisionRecord(c, federation_test(c), costs(c.callee), SHAPE[c.callee],
                           PAGED[c.callee], REVISIT[c.callee])
            for c in meridian_candidates(facts)]


def remote_nodes(workers: Mapping[str, Node]) -> set[str]:
    """Which nodes sit behind a seam, read from the workers, never listed by hand."""
    return {name for name, node in workers.items() if isinstance(node, RemoteAgentNode)}


DECISIONS_DIR = Path(__file__).resolve().parent.parent / "docs" / "decisions"


def write_records(directory: Path = DECISIONS_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for record in records():
        path = directory / f"{record.candidate.callee}.md"
        path.write_text(record.render(), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_records():
        print(written)
