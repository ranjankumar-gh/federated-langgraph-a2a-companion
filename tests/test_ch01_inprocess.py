"""Meridian in one deployment settles a claim end to end."""

from langchain_core.runnables import RunnableConfig

from meridian.inprocess import build_inprocess_graph
from meridian.samples import clean_claim, suspect_claim


def thread(tid: str) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


def test_clean_claim_is_paid_against_the_estimate() -> None:
    out = build_inprocess_graph().invoke(clean_claim(), thread("t1"))
    assert out["decision"] == "paid"
    assert out["repair_estimate"] == 3_000.0
    assert out["payment_ref"] == "PAY-CLM-1001-3000"
    assert out["completed"] == ["fraud_screening", "repair_estimate", "payments"]


def test_suspect_claim_is_referred_before_any_payment() -> None:
    out = build_inprocess_graph().invoke(suspect_claim(), thread("t2"))
    assert out["fraud_score"] == 82.0
    assert out["decision"] == "refer"
    assert "payment_ref" not in out


def test_one_checkpoint_holds_every_agents_work() -> None:
    graph = build_inprocess_graph()
    graph.invoke(suspect_claim(), thread("t3"))
    values = graph.get_state(thread("t3")).values
    assert values["fraud_signals"] == ["reported_late", "high_amount",
                                       "repeat_claimant"]
