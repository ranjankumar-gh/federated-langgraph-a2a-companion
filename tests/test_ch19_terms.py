"""Evidence as a term: what a partner keeps, where it processes, and what check holds it
to, before a single claim crosses."""

from dataclasses import replace
from datetime import date

from meridian.data_classes import MERIDIAN_POLICY, STRICT_POLICY
from meridian.receipts import OWNERSHIP, RISK_OWNERSHIP
from meridian.risk_server import RISK_TERMS_2027_04, RISK_TERMS_2027_06, risk_card
from meridian.saga import EXPECTED
from northgate.server import NORTHGATE_V7_5, NORTHGATE_V7_6
from seamkit.contract import (
    CrossingPolicy,
    EvidenceTerms,
    Expectations,
    Field,
    SeamTerms,
    SkillTerms,
    check,
    load_expectations,
    terms_of,
)
from seamkit.evidence import Gap, gap_signals, with_evidence
from seamkit.inventory import SeamEntry, SeamInventory

ESTIMATE = load_expectations(EXPECTED / "repair_estimate.evidenced.toml")
SCREENING = load_expectations(EXPECTED / "fraud_screening.evidenced.toml")


def details(terms: SeamTerms, expects: Expectations,
            policy: CrossingPolicy = MERIDIAN_POLICY) -> list[str]:
    return [b.detail for b in check(terms, expects, policy)]


def test_the_partners_files_state_the_evidence_terms() -> None:
    assert NORTHGATE_V7_6.effective == date(2027, 7, 1)
    assert NORTHGATE_V7_6.evidence == EvidenceTerms(2555, 10, "daily")
    history = NORTHGATE_V7_6.skills["estimate_history"]
    assert (history.reads, history.regions) == (("claim_id",), ("home",))
    assert RISK_TERMS_2027_06.evidence == EvidenceTerms(2555, 5, "daily")
    assert RISK_TERMS_2027_06.effective == date(2027, 6, 1)
    assert NORTHGATE_V7_5.evidence is None and RISK_TERMS_2027_04.evidence is None
    published = terms_of(risk_card("2027.06", RISK_TERMS_2027_06))[0]
    assert published.evidence == RISK_TERMS_2027_06.evidence  # it rides on the card
    assert published.skills["screen_fraud"].regions == ("home",)


def test_check_holds_the_evidence_terms_to_what_claims_expects() -> None:
    assert details(NORTHGATE_V7_6, ESTIMATE) == []
    assert details(RISK_TERMS_2027_06, SCREENING) == []
    assert details(NORTHGATE_V7_5, ESTIMATE) == [
        "estimate_repair states no evidence terms, and this seam expects them"]
    short = replace(NORTHGATE_V7_6, evidence=EvidenceTerms(365, 30, "none"))
    assert details(short, ESTIMATE) == [
        "estimate_repair keeps evidence 365 days, fewer than the 2555 expected",
        "estimate_repair produces it in 30 days, slower than the 14 expected",
        "estimate_repair does not witness the log head, and this seam expects it "
        "daily"]


def test_a_field_the_skill_does_not_declare_is_refused() -> None:
    sends = replace(ESTIMATE, skill="estimate_history", reads={}, into={},
                    sends=("claim_id", "damaged_parts"))
    refused = details(NORTHGATE_V7_6, sends)
    assert "estimate_history does not declare reading damaged_parts" in refused
    assert not [d for d in details(NORTHGATE_V7_5, sends) if "declare" in d]


def test_the_stricter_policy_keeps_the_car_at_home() -> None:
    taken = (Field("claim_id", "string"), Field("photo_url", "string"))
    image = SkillTerms(inputs=taken, outputs=(), p95_seconds=2.0,
                       reads=("claim_id", "photo_url"), regions=("elsewhere",))
    terms = replace(NORTHGATE_V7_6, skills={"estimate_image": image})
    sends = replace(ESTIMATE, skill="estimate_image", reads={}, into={},
                    sends=("claim_id", "photo_url"))
    assert not [d for d in details(terms, sends) if "would process" in d]
    [refused] = [d for d in details(terms, sends, STRICT_POLICY)
                 if "would process" in d]
    assert refused == ("estimate_image would process photo_url (vehicle) in "
                       "['elsewhere'], not in ['home']")


def test_the_inventory_names_the_evidence_each_seam_keeps() -> None:
    inventory = SeamInventory([SeamEntry("claims_intake", "repair_estimate", "Claims",
                                         "Northgate"),
                               SeamEntry("claims_intake", "fraud_screening", "Claims",
                                         "Risk")])
    inventory.entries[0].answers["accountability"] = "receipts: countersigned"
    with_evidence(inventory, {"repair_estimate": NORTHGATE_V7_6,
                              "fraud_screening": RISK_TERMS_2027_04})
    northgate, risk = (e.answers.get("accountability") for e in inventory.entries)
    assert northgate == ("receipts: countersigned; evidence: kept 2555 days, produced "
                         "within 10, witnessed daily")
    assert risk is None  # 2027.04 states nothing, and the cell stays open


def test_a_gap_is_a_signal_the_ownership_map_routes_to_both_owners() -> None:
    gap = Gap("where the data went", "northgate", "no regions declared")
    [signal] = gap_signals([gap], "repair_estimate")
    assert signal.kind == "evidence"
    assert {p.to for p in OWNERSHIP.route(signal)} == {
        "claims-oncall", "northgate-support, via the partner manager"}
    [peer] = gap_signals([gap], "estimate_history")
    assert {p.to for p in RISK_OWNERSHIP.route(peer)} == {
        "risk-oncall", "northgate-support, via the partner manager"}
