"""What one deployment gave for free and the naive federation lost: one test per
property.

Each test passes by showing the break. A later chapter turns each into the test of its
fix.
"""

import json

import pytest
from langchain_core.runnables import RunnableConfig

from meridian.federated_naive import (
    CLAIMS_CREDENTIAL,
    Partners,
    RiskRuntime,
    build_federated_graph,
)
from meridian.inprocess import (
    IN_PROCESS_WORKERS,
    ClaimState,
    build_graph,
    build_inprocess_graph,
)
from meridian.samples import clean_claim, suspect_claim


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


def test_contract_a_rescaled_score_passes_silently() -> None:
    claim = suspect_claim()
    in_process = build_inprocess_graph().invoke(claim, thread("c1"))
    assert in_process["decision"] == "refer"  # 82 > 70

    partners = Partners(risk=RiskRuntime(release="2026.10"))
    federated = build_federated_graph(partners).invoke(
        claim, thread("c2"))
    assert federated["fraud_score"] == 0.82  # same field, same type, new scale
    assert federated["decision"] == "paid"  # and no error anywhere


def test_authority_every_partner_receives_the_claims_credential() -> None:
    partners = Partners()
    build_federated_graph(partners).invoke(clean_claim(), thread("a1"))
    for partner in (partners.risk, partners.northgate, partners.finance):
        assert partner.received[-1]["credential"] == CLAIMS_CREDENTIAL


def test_state_the_partners_work_lives_in_a_checkpoint_we_cannot_read() -> None:
    partners = Partners()
    graph = build_federated_graph(partners)
    graph.invoke(suspect_claim(), thread("s1"))
    ours = graph.get_state(thread("s1")).values
    assert "fraud_signals" not in ours
    theirs = partners.risk.graph.get_state(thread("CLM-1002")).values
    assert theirs["fraud_signals"] == ["reported_late", "high_amount",
                                       "repeat_claimant"]


def test_failure_a_partner_timeout_arrives_as_a_missing_key() -> None:
    def risk_down(state: ClaimState) -> ClaimState:
        raise TimeoutError("fraud screening did not answer")

    in_process = build_graph({**IN_PROCESS_WORKERS, "fraud_screening": risk_down})
    with pytest.raises(TimeoutError):
        in_process.invoke(suspect_claim(), thread("f1"))

    partners = Partners(risk=RiskRuntime(available=False))
    federated = build_federated_graph(partners).invoke(
        suspect_claim(), thread("f2"))
    assert "fraud_score" not in federated
    assert federated["decision"] == "paid"


def test_accountability_our_record_ends_at_the_call() -> None:
    partners = Partners()
    graph = build_federated_graph(partners)
    graph.invoke(clean_claim(), thread("r1"))
    ours = json.dumps([s.values for s in graph.get_state_history(thread("r1"))])
    crossed = partners.northgate.received[-1]
    assert crossed["request"]["claim_id"] == "CLM-1001"
    assert crossed["release"] == "v7"
    assert "v7" not in ours  # which release answered is recorded only on their side
