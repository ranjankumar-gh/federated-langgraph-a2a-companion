"""Contract version, behaviour version: the model swap behind an unchanged card, the
behavioural canaries that catch it, Chapter 1's 0-1 score caught the same way, the
notice rule, the protocol choice at connect, and the compatibility matrix.
"""

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from meridian.canaries import (
    ESTIMATE_BASELINE,
    SCREENING_BASELINE,
    canary_suites,
    with_versions,
)
from meridian.data_classes import MERIDIAN_POLICY
from meridian.drill import T0
from meridian.federated import (
    EXPECTATIONS,
    OWNERS_A2A,
    SEAMS,
    A2APartners,
    a2a_workers,
    connect_partners,
    with_contract,
)
from meridian.finance_server import FINANCE_TERMS_2026_15
from meridian.incident import swap_model
from meridian.inprocess import ClaimGraph, ClaimState
from meridian.matrix import COMMITTED, NORTHGATE_REVISIONS, estimate_matrix
from meridian.risk_server import RISK_URL, RiskPartner
from meridian.saga import EXPECTED
from meridian.samples import clean_claim
from meridian.trust import MERIDIAN_TRUST
from northgate.server import (
    NORTHGATE_V7_2,
    NORTHGATE_V7_3,
    NORTHGATE_V7_4,
    NORTHGATE_V8,
)
from seamkit.canary import Band, CanaryPolicy, CanarySuite, Golden, tick
from seamkit.cards import card_digest
from seamkit.compat import diff, load_seen, record_seen
from seamkit.contract import check, held_to, load_expectations
from seamkit.delegation import as_peer
from seamkit.errors import BehaviourDrift, ContractBreach
from seamkit.inventory import inventory_from_graph
from seamkit.reconcile import Finding
from seamkit.remote import RemoteAgentNode
from tests.wire import Federation, as_run, thread

pytestmark = pytest.mark.anyio

HOUR = 3600.0
ESTIMATE = EXPECTATIONS["repair_estimate"]


def proven(fed: Federation) -> None:
    """Northgate's estimates are proven under this card (Chapter 5's gradient)."""
    record = fed.connections["repair_estimate"].record("estimate_repair")
    record.clean.update(f"earlier-{n}" for n in range(20))


def tier(fed: Federation, seam: str, skill: str) -> str:
    conn = fed.connections[seam]
    return conn.gradient(skill).tier(conn.record(skill))


@dataclass
class Watched(Federation):
    """A federation with a canary suite on every seam with a [canary] table."""

    claims: ClaimGraph | None = None

    async def suites(self, findings: list[Finding]) -> dict[str, CanarySuite]:
        self.claims = await self.graph()
        return canary_suites(a2a_workers(self.connections, log=self.records.append),
                             findings.append)

    async def run(self, claim: ClaimState, tid: str) -> dict[str, Any]:
        assert self.claims is not None
        return await self.claims.ainvoke(claim, thread(tid))


# The incident ------------------------------------------------------------------------

async def test_the_incident_a_swapped_model_passes_every_check() -> None:
    fed = Federation()
    graph = await fed.graph()
    proven(fed)
    northgate = fed.partners.northgate
    before = card_digest(northgate.card)
    swap_model(northgate)  # the release, the terms and the card are untouched
    out = await graph.ainvoke(clean_claim(), thread("i15"))
    assert (out["repair_estimate"], out["decision"]) == (2_450.0, "paid")
    conn = fed.connections["repair_estimate"]
    assert conn.trust is not None and conn.trust.digest == before  # Chapter 5 trusts it
    assert (conn.release, conn.terms[0].revision) == ("v7", "v7.2")
    assert conn.breaches == () and conn.upcoming == ()  # Chapter 4 passes it
    assert tier(fed, "repair_estimate", "estimate_repair") == "proven"
    assert not [r for r in fed.records if r.error]  # nothing failed, nothing paged


# The canary suite --------------------------------------------------------------------

async def test_the_next_canary_run_finds_the_swap_and_the_skill_goes_unproven() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    proven(fed)  # the first baseline keeps the record the skill earned
    estimate = suites["repair_estimate"]
    assert (await estimate.go(T0)).kind == "in band"
    assert tier(fed, "repair_estimate", "estimate_repair") == "proven"
    swap_model(fed.partners.northgate)
    verdict = await estimate.go(T0 + 6 * HOUR)
    assert (verdict.kind, verdict.goldens) == ("drift", ("CLM-C02", "CLM-C03"))
    assert verdict.answers["CLM-C03"] == {"repair_estimate": 2_450.0}
    assert tier(fed, "repair_estimate", "estimate_repair") == "unproven"
    conn = fed.connections["repair_estimate"]
    assert conn.gradient("estimate_repair").ceiling(conn.record("estimate_repair")) \
        == 5_000.0  # Chapter 5's ceiling applies again
    [finding] = findings
    assert (finding.verdict, finding.partner_release) == ("canary drift", "v7")
    assert finding.detail == (f"estimate_repair: CLM-C02, CLM-C03 against "
                              f"{ESTIMATE_BASELINE.version}")
    out = await fed.run(clean_claim(), "after")  # claims flow
    assert (out["repair_estimate"], out["decision"]) == (2_450.0, "paid")


async def test_chapter_1s_incident_caught_the_screening_seam_holds() -> None:
    risk = RiskPartner(release="2026.10", rescaled=True)  # 0-1 under a 0-100 card
    fed, findings = Watched(A2APartners(risk=risk)), list[Finding]()
    suites = await fed.suites(findings)
    assert fed.connections["fraud_screening"].breaches == ()  # every check passes
    verdict = await suites["fraud_screening"].go(T0)
    assert verdict.kind == "drift"
    assert verdict.answers["CLM-C13"] == {"fraud_score": 0.68}  # 68, on 0-1
    sent = len(risk.executor.received)
    with pytest.raises(BehaviourDrift, match="screen_fraud is held: canary drift on "
                                             "CLM-C11, CLM-C12, CLM-C13"):
        await fed.run(clean_claim(), "held")
    assert len(risk.executor.received) == sent  # nothing screened, nothing cleared
    assert [f.verdict for f in findings] == ["canary drift"]


async def test_canary_answers_never_earn_trust() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    for run in range(5):  # four goldens a run: 20 clean canary answers
        assert (await suites["repair_estimate"].go(T0 + run * 6 * HOUR)).kind \
            == "in band"
    record = fed.connections["repair_estimate"].record("estimate_repair")
    assert record.clean == set() and record.malformed == set()
    assert tier(fed, "repair_estimate", "estimate_repair") == "unproven"
    seen = suites["repair_estimate"].seen["CLM-C03"]
    assert seen["provenance"]["repair_estimate"]["canary"] == "CLM-C03"  # tagged


async def test_a_refused_canary_is_a_finding_not_drift() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    estimate = suites["repair_estimate"]
    estimate.goldens = (Golden("CLM-C01", {"claim_id": "CLM-C01",
                                           "damaged_parts": ["wing"]},
                               ("repair_estimate",)),)  # a part Northgate never prices
    verdict = await estimate.go(T0)
    assert (verdict.kind, verdict.goldens) == ("refused", ("CLM-C01",))
    assert [f.verdict for f in findings] == ["canary refused"]
    conn = fed.connections["repair_estimate"]
    assert conn.behaviour["estimate_repair"] == ESTIMATE_BASELINE.version  # unchanged
    assert conn.held == {}


async def test_accepting_a_new_baseline_restarts_the_skill_unproven() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    proven(fed)
    estimate = suites["repair_estimate"]
    swap_model(fed.partners.northgate)
    assert (await estimate.go(T0)).kind == "drift"
    accepted = estimate.accept("lead:tl-031", T0 + HOUR)
    conn = fed.connections["repair_estimate"]
    assert accepted.version != ESTIMATE_BASELINE.version
    assert conn.behaviour["estimate_repair"] == accepted.version
    assert estimate.accepted == [ESTIMATE_BASELINE, accepted]  # the old one is kept
    assert tier(fed, "repair_estimate", "estimate_repair") == "unproven"
    assert (await estimate.go(T0 + 6 * HOUR)).kind == "in band"  # the new behaviour
    for n in range(20):  # it earns proven again from real answers, not canaries
        await fed.run(clean_claim(), f"real-{n}")
    assert tier(fed, "repair_estimate", "estimate_repair") == "proven"


async def test_the_detection_bound_is_the_cadence() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    estimate = suites["repair_estimate"]
    await tick([estimate], T0)
    changed = T0 + HOUR  # Northgate swaps its model an hour after a run
    swap_model(fed.partners.northgate)
    found = None
    for hour in range(2, 13):  # the scheduler ticks hourly; the suite runs when due
        ran = await tick([estimate], T0 + hour * HOUR)
        if ran.get("repair_estimate") and ran["repair_estimate"].kind == "drift":
            found = T0 + hour * HOUR
            break
    assert found == T0 + 6 * HOUR and ESTIMATE.canary is not None
    assert found - changed <= ESTIMATE.canary.every_seconds  # within one cadence


async def test_the_ledger_and_crossing_records_carry_the_behaviour_version() -> None:
    fed, findings = Watched(), list[Finding]()
    await fed.suites(findings)
    out = await fed.run(clean_claim(), "bv")
    assert out["ledger"]["repair_estimate"]["behaviour"] == ESTIMATE_BASELINE.version
    assert out["ledger"]["fraud_screening"]["behaviour"] == SCREENING_BASELINE.version
    assert out["ledger"]["payments"]["behaviour"] == ""  # no canary on payments
    sends = [r for r in fed.records if r.seam == "repair_estimate" and r.call == "send"]
    assert sends and all(r.behaviour == ESTIMATE_BASELINE.version for r in sends)


# Committing skills never get a canary -----------------------------------------------

def test_check_refuses_a_canary_on_a_skill_that_commits() -> None:
    canary = CanaryPolicy(3600.0, Band(relative=0.05))
    booking = replace(load_expectations(EXPECTED / "repair_booking.toml"),
                      canary=canary)
    [breach] = [b for b in check(NORTHGATE_V7_4, booking, MERIDIAN_POLICY)
                if "canary" in b.detail]
    assert breach.detail == ("book_repair commits at the partner (undone by "
                             "cancel_booking), so it is never sent a canary")
    payments = replace(EXPECTATIONS["payments"], canary=canary)
    [breach] = [b for b in check(FINANCE_TERMS_2026_15, payments, MERIDIAN_POLICY)
                if "canary" in b.detail]
    assert "pay_claim commits at the partner (undone by void_payment)" in breach.detail


# Declared versions ------------------------------------------------------------------

def test_diff_labels_v8_breaking_and_v7_3_safe() -> None:
    assert [str(c) for c in diff(NORTHGATE_V7_2, NORTHGATE_V8, ESTIMATE)] == [
        "input added vin (required): breaking"]
    assert [str(c) for c in diff(NORTHGATE_V7_2, NORTHGATE_V7_3, ESTIMATE)] == [
        "term changed key_seconds (None to 3600): safe"]
    narrowed = replace(NORTHGATE_V7_2, skills={
        **NORTHGATE_V7_2.skills, "estimate_repair": replace(
            NORTHGATE_V7_2.skills["estimate_repair"],
            outputs=tuple(replace(f, unit="EUR") if f.name == "estimate" else f
                          for f in NORTHGATE_V7_2.skills["estimate_repair"].outputs))})
    [change] = diff(NORTHGATE_V7_2, narrowed, ESTIMATE)
    assert (change.kind, change.field, change.breaking) == (
        "unit changed", "estimate", True)


def test_the_matrix_matches_the_committed_table() -> None:
    # The CI gate: regenerate, and any changed cell fails until a person commits it.
    assert estimate_matrix() == COMMITTED.read_text(encoding="utf-8")
    assert len(NORTHGATE_REVISIONS) == 6


def test_the_notice_rule() -> None:
    seen = {rev: row.first_seen for rev, row in
            load_seen(Path(EXPECTED).parent / "seen" / "repair_estimate.toml").items()}
    _, upcoming = held_to([NORTHGATE_V7_2, NORTHGATE_V8], ESTIMATE, MERIDIAN_POLICY,
                          seen)  # first seen 2026-10-05: 28 days of notice
    assert [b.detail for b in upcoming] == [
        "estimate_repair requires vin, which this seam does not send"]
    late = {**seen, "v8": date(2026, 10, 29)}  # the same revision, four days ahead
    _, upcoming = held_to([NORTHGATE_V7_2, NORTHGATE_V8], ESTIMATE, MERIDIAN_POLICY,
                          late)
    assert upcoming[-1].detail == (
        "estimate_repair breaks this seam 4 days after Claims first saw it, inside the "
        "21 days of notice agreed")
    safe = {**seen, "v7.3": date(2026, 12, 20)}  # a safe revision needs no notice
    assert held_to([NORTHGATE_V7_2, NORTHGATE_V7_3], ESTIMATE, MERIDIAN_POLICY,
                   safe)[1] == []


def test_a_revision_keeps_the_date_claims_first_saw_it(tmp_path: Path) -> None:
    path = tmp_path / "seen.toml"
    record_seen(path, [NORTHGATE_V7_2, NORTHGATE_V8], "v7", date(2026, 10, 5))
    again = record_seen(path, [NORTHGATE_V7_2, NORTHGATE_V8], "v7", date(2026, 10, 30))
    assert again["v8"].first_seen == date(2026, 10, 5)  # the notice it gave stands
    assert load_seen(path) == again


def screened(claim: ClaimState) -> dict[str, Any]:
    """What the screening seam sends of a claim, and nothing else."""
    fields: dict[str, Any] = dict(claim)
    return {k: fields[k] for k in SEAMS["fraud_screening"].sends}


class Recording(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.sent: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        return await self.inner.handle_async_request(request)


async def test_the_protocol_claims_speaks_is_chosen_and_sent() -> None:
    risk = RiskPartner(protocols=("0.3", "1.0"))  # in the middle of a migration
    trust = {**MERIDIAN_TRUST, "Risk": replace(
        MERIDIAN_TRUST["Risk"], digests=frozenset({card_digest(risk.card)}))}
    wire = Recording(httpx.ASGITransport(app=as_peer(risk.app, "claims")))
    http = httpx.AsyncClient(transport=wire, base_url=RISK_URL)
    conns = await connect_partners(A2APartners(risk=risk).apps(),
                                   {"fraud_screening": http}, trust=trust)
    conn = conns["fraud_screening"]
    assert (conn.protocol, conn.breaches) == ("1.0", ())
    node = RemoteAgentNode(SEAMS["fraud_screening"], conn)
    with as_run(thread("p15")):
        await node.call_once(screened(clean_claim()))
    posts = [r for r in wire.sent if r.method == "POST"]
    assert posts and all(r.headers["A2A-Version"] == "1.0" for r in posts)


async def test_a_card_with_no_protocol_claims_speaks_is_a_breach() -> None:
    risk = RiskPartner(protocols=("0.3",))
    trust = {**MERIDIAN_TRUST, "Risk": replace(
        MERIDIAN_TRUST["Risk"], digests=frozenset({card_digest(risk.card)}))}
    conns = await connect_partners(A2APartners(risk=risk).apps(), trust=trust)
    [breach] = conns["fraud_screening"].breaches
    assert breach.detail == "Risk's card offers protocol 0.3, and this seam speaks 1.0"
    node = RemoteAgentNode(SEAMS["fraud_screening"], conns["fraud_screening"])
    with pytest.raises(ContractBreach), as_run(thread("p15-none")):
        await node.call_once(screened(clean_claim()))
    assert risk.executor.received == []  # nothing was sent


# The seam inventory ------------------------------------------------------------------

async def test_the_inventory_contract_column_names_every_version() -> None:
    fed, findings = Watched(), list[Finding]()
    suites = await fed.suites(findings)
    assert fed.claims is not None
    inventory = with_versions(with_contract(inventory_from_graph(
        fed.claims, OWNERS_A2A), fed.connections), fed.connections, suites)
    rows = {e.callee: e.answers["contract"] for e in inventory.entries}
    assert rows["repair_estimate"] == (
        "terms v7.2 checked; protocol 1.0; notice 21 d; "
        f"behaviour {ESTIMATE_BASELINE.version}")
    assert rows["payments"].endswith("protocol as found; no notice term; "
                                     "behaviour unwatched")
