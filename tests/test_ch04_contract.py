"""The seam contract: published terms, Claims' expectations, and the check between."""

from dataclasses import replace
from datetime import date

import pytest
from a2a.types import TaskState
from langgraph.types import Command

from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import (
    EXPECTATIONS,
    OWNERS_A2A,
    SEAMS,
    A2APartners,
    build_a2a_graph,
    connect_partners,
    with_contract,
)
from meridian.inprocess import CLAIM_FIELDS, ClaimState
from meridian.risk_server import RISK_TERMS, RISK_TERMS_0_1, RiskPartner
from meridian.samples import clean_claim, suspect_claim
from northgate.server import NORTHGATE_V7, NORTHGATE_V8, NorthgatePartner
from seamkit.contract import (
    EXTENSION_URI,
    Field,
    SkillTerms,
    check,
    seam_spec,
    terms_of,
)
from seamkit.errors import ContractBreach, MalformedAnswer, PartnerRejected
from seamkit.inventory import inventory_from_graph
from seamkit.remote import RemoteAgentNode
from seamkit.tools import remote_skill_tool
from tests.test_ch03_mapping import task
from tests.wire import Federation, thread

pytestmark = pytest.mark.anyio

ESTIMATE = EXPECTATIONS["repair_estimate"]


async def test_terms_survive_the_card_fetch() -> None:
    conns = await connect_partners(A2APartners().apps())
    northgate = conns["repair_estimate"]
    [extension] = northgate.card.capabilities.extensions
    assert (extension.uri, extension.required) == (EXTENSION_URI, False)  # data only
    assert terms_of(northgate.card) == (NORTHGATE_V7,) == northgate.terms
    assert terms_of(conns["fraud_screening"].card) == (RISK_TERMS,)


async def test_every_meridian_seam_holds_today() -> None:
    conns = await connect_partners(A2APartners().apps())
    assert {s: (c.breaches, c.upcoming) for s, c in conns.items()} == {
        s: ((), ()) for s in SEAMS}


def test_seams_are_derived_from_the_expectations_not_typed() -> None:
    assert SEAMS["repair_estimate"].sends == ("claim_id", "damaged_parts")
    assert SEAMS["repair_estimate"].receives == {"estimate": "repair_estimate"}
    assert SEAMS["fraud_screening"].asks_caller is True  # it answers police_report
    assert SEAMS["repair_estimate"].asks_caller is False  # it refuses photo_url
    assert list(SEAMS) == ["fraud_screening", "repair_estimate", "payments"]
    assert all(SEAMS[s] == seam_spec(e) for s, e in EXPECTATIONS.items())


async def test_the_incident_v8_rejects_every_estimate_before_the_check() -> None:
    # Chapter 3's build: v8 deployed, the card changed only in words, nothing checked.
    partners = A2APartners(northgate=NorthgatePartner(release="v8", publishes=False))
    federation = Federation(partners)
    graph = await federation.graph()
    unchecked = replace(federation.connections["repair_estimate"], breaches=())
    graph = build_a2a_graph({**federation.connections, "repair_estimate": unchecked})
    with pytest.raises(PartnerRejected):
        await graph.ainvoke(clean_claim(), thread("v8-unchecked"))
    assert "VIN" in partners.northgate.card.skills[0].description  # a person read it
    assert partners.northgate.executor.received  # the task was sent, and rejected


async def test_v8_breaks_the_seam_on_the_day_it_is_announced() -> None:
    northgate = NorthgatePartner(release="v7", announces=NORTHGATE_V8)
    conns = await connect_partners(A2APartners(northgate=northgate).apps())
    seam = conns["repair_estimate"]
    assert seam.breaches == ()  # v7 is in force: claims keep flowing today
    [breach] = seam.upcoming
    assert (breach.revision, breach.effective) == ("v8", date(2026, 11, 2))
    assert breach.detail.endswith("requires vin, which this seam does not send")


async def test_v8_in_force_stops_only_its_seam_and_sends_nothing() -> None:
    partners = A2APartners(northgate=NorthgatePartner(release="v8"))
    federation = Federation(partners)
    graph = await federation.graph()
    with pytest.raises(ContractBreach, match="requires vin"):
        await graph.ainvoke(clean_claim(), thread("v8"))
    assert partners.northgate.executor.received == []  # nothing crossed
    state = (await graph.aget_state(thread("v8"))).values
    assert state["completed"] == ["fraud_screening"]  # the other seams still run
    refused = [r for r in federation.records if r.error == "ContractBreach"]
    assert [r.seam for r in refused] == ["repair_estimate"]


async def test_a_card_without_terms_is_refused_not_trusted() -> None:
    partners = A2APartners(northgate=NorthgatePartner(publishes=False))
    conns = await connect_partners(partners.apps())
    assert [b.detail for b in conns["repair_estimate"].breaches] == [
        "Northgate publishes no terms"]


def test_the_whole_state_send_is_refused_by_data_class() -> None:
    # Chapter 1's naive seam sent every field it had to Northgate.
    every_field = sorted(CLAIM_FIELDS - {"completed", "ledger"})
    whole = replace(ESTIMATE, sends=tuple(every_field))
    found = {b.detail for b in check(NORTHGATE_V7, whole, MERIDIAN_POLICY)}
    assert "estimate_repair would receive fraud_score, which is fraud_assessment data" \
        in found
    assert "estimate_repair would receive prior_claims, which is claim_history data" \
        in found
    assert "estimate_repair would receive repair_estimate, which is estimate data" \
        in found


def test_an_unclassified_field_crosses_nowhere() -> None:
    adds = replace(ESTIMATE, sends=(*ESTIMATE.sends, "adjuster_notes"))
    repair = NORTHGATE_V7.skills["estimate_repair"]
    notes = Field("adjuster_notes", "string", required=False)
    v7 = replace(NORTHGATE_V7, skills={
        **NORTHGATE_V7.skills,
        "estimate_repair": replace(repair, inputs=(*repair.inputs, notes))})
    [breach] = check(v7, adds, MERIDIAN_POLICY)
    assert breach.detail.endswith("adjuster_notes, which is unclassified data")


def test_a_new_question_from_the_partner_is_a_breach() -> None:
    repair = NORTHGATE_V7.skills["estimate_repair"]
    asks_more = replace(NORTHGATE_V7, skills={
        **NORTHGATE_V7.skills,
        "estimate_repair": replace(repair, asks=(*repair.asks, "vin"))})
    [breach] = check(asks_more, ESTIMATE, MERIDIAN_POLICY)
    assert "may ask for vin" in breach.detail


def test_slower_promise_and_changed_skew_are_breaches() -> None:
    repair = NORTHGATE_V7.skills["estimate_repair"]
    slower = replace(NORTHGATE_V7, skew_seconds=120, skills={
        **NORTHGATE_V7.skills, "estimate_repair": replace(repair, p95_seconds=30.0)})
    details = [b.detail for b in check(slower, ESTIMATE, MERIDIAN_POLICY)]
    assert details == [
        "estimate_repair tolerates 120 s of clock skew, not the 30 s agreed",
        "estimate_repair promises 30 s at p95, longer than the 10 s this seam waits"]


async def test_contract_a_card_that_states_0_1_is_refused_at_connect() -> None:
    # Chapter 1's contract test, federated half: Risk's change, published honestly.
    risk = RiskPartner(release="2026.10", rescaled=True, terms=RISK_TERMS_0_1)
    conns = await connect_partners(A2APartners(risk=risk).apps())
    [breach] = conns["fraud_screening"].breaches
    assert breach.detail == (
        "screen_fraud does not promise fraud_score as a required number on 0-100")


async def test_contract_0_82_under_an_unchanged_card_passes_every_check() -> None:
    # Release 2026.10 as it shipped: 0-1 scores, a card still saying 0-100. The check
    # reads declarations, and 0.82 is a valid 0-100 value. Chapter 15's case.
    risk = RiskPartner(release="2026.10", rescaled=True)
    federation = Federation(A2APartners(risk=risk))
    graph = await federation.graph()
    assert all(c.breaches == () for c in federation.connections.values())
    door_only: ClaimState = {**suspect_claim(), "damaged_parts": ["door"]}
    await graph.ainvoke(door_only, thread("c-0.82"))  # Risk asks for the report
    out = await graph.ainvoke(Command(resume={"police_report": "PR-5521"}),
                              thread("c-0.82"))
    assert (out["fraud_score"], out["decision"]) == (0.82, "paid")


async def test_an_artifact_off_the_contracted_scale_is_malformed() -> None:
    conns = await connect_partners(A2APartners().apps())
    node = RemoteAgentNode(SEAMS["fraud_screening"], conns["fraud_screening"])
    done = task(TaskState.TASK_STATE_COMPLETED, {"fraud_score": 820.0})
    with pytest.raises(MalformedAnswer, match="not fraud_score as a required number"):
        node._outcome(done)
    as_text = task(TaskState.TASK_STATE_COMPLETED, {"fraud_score": "82"})
    with pytest.raises(MalformedAnswer):
        node._outcome(as_text)


async def test_the_tool_schema_is_generated_from_the_partner_terms() -> None:
    conns = await connect_partners(A2APartners().apps())
    tool = remote_skill_tool(conns["repair_estimate"], "estimate_history", "Northgate",
                             log=lambda record: None)
    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    inputs = NORTHGATE_V7.skills["estimate_history"].inputs  # Northgate's, not ours
    assert set(schema["properties"]) == {f.name for f in inputs}
    assert set(schema.get("required", [])) == {f.name for f in inputs if f.required}


async def test_a_partner_without_terms_gets_no_tool() -> None:
    partners = A2APartners(northgate=NorthgatePartner(publishes=False))
    conns = await connect_partners(partners.apps())
    with pytest.raises(ContractBreach, match="publishes no terms"):
        remote_skill_tool(conns["repair_estimate"], "estimate_history", "Northgate")


async def test_the_inventory_contract_column_comes_from_the_check() -> None:
    northgate = NorthgatePartner(announces=NORTHGATE_V8)
    federation = Federation(A2APartners(northgate=northgate))
    graph = await federation.graph()
    inventory = with_contract(inventory_from_graph(graph, OWNERS_A2A),
                              federation.connections)
    answers = {e.callee: e.answers["contract"] for e in inventory.entries}
    assert answers == {
        "fraud_screening": "terms 2026.09 checked",
        "repair_estimate": "terms v7 checked; v8 breaks it from 2026-11-02",
        "payments": "terms 2026.10 checked"}


def test_skill_terms_declare_what_later_chapters_enforce() -> None:
    repair = NORTHGATE_V7.skills["estimate_repair"]
    assert isinstance(repair, SkillTerms)
    assert repair.cost_per_task == 3.5  # read from Chapter 18
    assert repair.compensated_by == ""  # nothing undoes an estimate yet: Chapter 13
