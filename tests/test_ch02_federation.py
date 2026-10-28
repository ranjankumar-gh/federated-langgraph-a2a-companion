"""The Federation Test on Meridian's five candidate calls, and misplaced() both ways."""

from dataclasses import replace

import pytest

from meridian.decisions import (
    DECISIONS_DIR,
    FACTS,
    OWNERS,
    meridian_candidates,
    records,
    remote_nodes,
)
from meridian.federated import OWNERS_A2A, a2a_workers
from meridian.federated_naive import OWNERS as CH1_OWNERS
from meridian.inprocess import build_graph, build_inprocess_graph
from meridian.policy import build_policy_inprocess_graph
from meridian.policy_server import incident_workers
from seamkit.federation import Finding, federation_test, misplaced
from tests.wire import Federation


def verdicts() -> dict[str, tuple[str, tuple[str, ...]]]:
    return {c.callee: (v.option, v.first)
            for c in meridian_candidates() for v in [federation_test(c)]}


def test_the_hub_seams_and_the_peer_seam_pass_and_policy_lookup_fails() -> None:
    assert verdicts() == {
        "fraud_screening": ("a2a_agent", ("contract", "authority")),
        "repair_estimate": ("a2a_agent", ("contract", "authority")),
        "payments": ("a2a_agent", ("contract", "authority")),
        "estimate_history": ("a2a_agent", ("contract", "authority")),
        "policy_lookup": ("subgraph", ()),
    }


def test_the_trust_domain_question_sorts_the_seams_as_chapter_6_does() -> None:
    crosses = {c.callee: federation_test(c).answers["trust_domain"][0]
               for c in meridian_candidates()}
    assert crosses == {"fraud_screening": False, "payments": False,
                       "repair_estimate": True, "estimate_history": True,
                       "policy_lookup": False}


def test_a_second_pipeline_under_one_owner_is_a_warning_not_a_reason() -> None:
    policy = next(c for c in meridian_candidates() if c.callee == "policy_lookup")
    verdict = federation_test(policy)
    assert verdict.answers["owner"] == (False, "Claims / Claims")
    assert verdict.answers["cadence"][0] is True  # its own pipeline
    assert verdict.warnings == ("cadence",)
    assert set(verdict.refused) == {"mcp_tool", "a2a_agent"}


@pytest.mark.parametrize("stateful, model_decides, option", [
    (True, False, "subgraph"), (True, True, "subgraph"),
    (False, False, "subgraph"), (False, True, "mcp_tool"),
])
def test_one_owner_chooses_by_state_and_by_who_decides(
        stateful: bool, model_decides: bool, option: str) -> None:
    policy = next(c for c in meridian_candidates() if c.callee == "policy_lookup")
    shaped = replace(policy, stateful=stateful, model_decides=model_decides)
    assert federation_test(shaped).option == option


def test_changing_one_fact_changes_every_verdict_it_touches() -> None:
    # Risk joins the Claims release train: the cadence answer flips, the owner does not.
    merged = {**FACTS, "Risk": replace(FACTS["Risk"], cadence="claims-train")}
    before = {r.candidate.callee: r.verdict for r in records()}
    after = {r.candidate.callee: r.verdict for r in records(merged)}
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"fraud_screening", "estimate_history"}
    assert after["fraud_screening"].option == "a2a_agent"  # still another owner


def test_misplaced_finds_chapter_1s_owner_lines_running_in_process() -> None:
    found = misplaced(build_inprocess_graph(), CH1_OWNERS, remote=set())
    assert sorted(f.callee for f in found) == ["fraud_screening", "payments",
                                               "repair_estimate"]
    assert {f.problem for f in found} == {"owner line in-process"}


def test_the_policy_subgraph_is_never_a_finding() -> None:
    # Same owner, in-process: correct. Only Chapter 1's three owner lines are reported.
    found = misplaced(build_policy_inprocess_graph(), OWNERS, remote=set())
    assert {f.callee for f in found} == {"fraud_screening", "repair_estimate",
                                         "payments"}


@pytest.mark.anyio
async def test_misplaced_finds_the_incidents_remote_call_with_one_owner() -> None:
    federation = Federation()
    await federation.graph()
    seams = a2a_workers(federation.connections)
    policy = federation.connections["payments"]  # any connection: the graph is not run
    workers = incident_workers(policy, others=seams)
    found = misplaced(build_graph(workers), OWNERS, remote_nodes(workers))
    assert found == [Finding("claims_intake", "policy_lookup", "remote with one owner")]
    assert set(OWNERS_A2A) <= set(OWNERS)


def test_the_committed_decision_records_match_the_facts() -> None:
    for record in records():
        path = DECISIONS_DIR / f"{record.candidate.callee}.md"
        assert path.read_text(encoding="utf-8") == record.render(), path.name
