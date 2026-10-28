"""Audit evidence across organisations: the referral nobody could explain, Risk's signed
decision record, the log head a partner witnesses, and the pack an auditor joins."""

import asyncio
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

import seamkit.ledger
from meridian.canaries import SCREENING_BASELINE
from meridian.data_classes import MERIDIAN_POLICY, STRICT_POLICY
from meridian.federated import SEAMS, A2APartners, a2a_workers, connect_partners
from meridian.identity import (
    ADJUSTER,
    ADJUSTER_SESSION,
    ISSUER,
    KEYS,
    MERIDIAN,
    MeridianIssuer,
    revocation_sources,
)
from meridian.inprocess import ClaimState, build_graph
from meridian.receipts import pinned, receipted
from meridian.risk_server import (
    RISK_TERMS_2027_04,
    RISK_TERMS_2027_06,
    RiskPartner,
    ScreeningModel,
)
from meridian.saga import EXPECTED
from northgate import server as northgate_server
from northgate.server import NORTHGATE_V7_5, NORTHGATE_V7_6, NorthgatePartner
from seamkit.approval import Approval, sign_approval
from seamkit.canary import Verdict
from seamkit.contract import EvidenceTerms, SeamTerms, load_expectations
from seamkit.costs import CostEntry, CostLedger
from seamkit.drill import Clock
from seamkit.evidence import (
    DAY,
    AuditPolicy,
    EvidencePack,
    Gap,
    Held,
    Records,
    Watched,
    build_pack,
    gaps,
    head_of,
    join,
    watched_canaries,
    witnessed,
)
from seamkit.receipt import ReceiptLog, Receipts
from seamkit.reconcile import Finding
from tests.wire import thread

pytestmark = pytest.mark.anyio

SCREENING = "fraud_screening"
HISTORY = "estimate_history"
MAY = datetime(2027, 5, 17, 11, 0, tzinfo=UTC).timestamp()
CLAIM = {"claim_id": "CLM-1003", "amount": 14_200.0, "days_to_report": 9,
         "prior_claims": 1}
_WANTS = load_expectations(EXPECTED / "fraud_screening.evidenced.toml").evidence
assert _WANTS is not None
WANTS: EvidenceTerms = _WANTS
RISK_TERMS: dict[str, SeamTerms] = {"2027.04": RISK_TERMS_2027_04,
                                    "2027.06": RISK_TERMS_2027_06}
NORTHGATE_TERMS: dict[str, SeamTerms] = {"v7.5": NORTHGATE_V7_5, "v7.6": NORTHGATE_V7_6}
PINNED = {**pinned(), "claims": KEYS["claims"].public_key(),
          ISSUER: MERIDIAN.keys[ISSUER]}


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    shared = Clock(MAY)
    monkeypatch.setattr(seamkit.ledger, "CLOCK", shared)
    monkeypatch.setitem(northgate_server.BOOKINGS["meridian"], "CLM-1003",
                        ("SHOP-3", ["bumper"]))
    yield shared


class Scored(ScreeningModel):
    """Risk's model on CLM-1003: three signals fire, and the score is 79."""

    def screen(self, claim: dict[str, Any], history: list[dict[str, Any]],
               config: RunnableConfig) -> tuple[float, list[str]]:
        return 79.0, ["reported_late", "high_amount", "repeat_claimant"]


def asks_history(claim: dict[str, Any],
                 seen: list[dict[str, Any]]) -> dict[str, Any] | None:
    return None if seen else {"claim_id": claim["claim_id"]}


@dataclass
class Referral:
    """CLM-1003 referred as suspected fraud: Claims -> Risk -> Northgate, one chain."""

    clock: Clock
    risk: RiskPartner
    northgate: NorthgatePartner
    claims_log: ReceiptLog
    risk_log: ReceiptLog
    chain: str = ""
    score: float = 0.0

    @property
    def context(self) -> str:
        return str(self.risk.executor.received[0]["context_id"])


async def refer(clock: Clock, risk: str, northgate: str,
                directory: Path | None = None) -> Referral:
    records = asyncio.Event()
    records.set()  # the police records service answers at once
    claims_log, risk_log = ReceiptLog(directory), ReceiptLog()
    onward = Receipts(KEYS["risk"], "risk", os.urandom(32), PINNED, risk_log, True)
    partner = RiskPartner(release=risk, lookups=records, terms=RISK_TERMS[risk],
                          model=Scored(), assistant=asks_history, countersigns=True,
                          onward=onward)
    ng = NorthgatePartner(release=northgate)
    issuer = MeridianIssuer(sessions={  # the adjuster is working CLM-1003
        ADJUSTER_SESSION: (ADJUSTER, frozenset({"CLM-1003"}))})
    apps = {SCREENING: A2APartners(risk=partner, northgate=ng).apps()[SCREENING]}
    nodes = a2a_workers(await connect_partners(apps, issuer=issuer),
                        lambda r: None, {SCREENING: SEAMS[SCREENING]}, issuer)
    receipted(list(nodes.values()), claims_log)
    out = await build_graph(nodes, InMemorySaver()).ainvoke(
        cast(ClaimState, dict(CLAIM)), thread("t-1003"))
    ref = Referral(clock, partner, ng, claims_log, risk_log)
    ref.score = float(out["fraud_score"])
    ref.chain = next(claims_log.receipts()).authority
    return ref


def approval(task_id: str) -> Approval:
    return sign_approval(Approval(ISSUER, "adj-117", "claims/adjuster", task_id,
                                  "CLM-1003", 14_200.0, "sha256:question"),
                         KEYS[ISSUER])


def packs(ref: Referral, ask: Sequence[str] = ("prior_claims",), watching: bool = True,
          witness: bool = True) -> list[EvidencePack]:
    """Claims' pack and Risk's pack, each built over its own stores."""
    task = next(ref.claims_log.receipts()).ours.task_id
    witnesses = (ref.risk.witness(SCREENING, *head_of(ref.claims_log, SCREENING)),
                 ) if witness else ()
    ledger = CostLedger()
    ledger.append(CostEntry(SCREENING, "Risk", "t-1003", task, "k", "screen_fraud",
                            "claim", ref.risk.release, 0.62, "", "", ref.clock()))
    settled = Finding(SCREENING, "settled", "k", task, authority=ref.chain,
                      detail="referral confirmed by an adjuster")
    serial = next(iter(revocation_sources().values())).fetch().serial
    canaries = watched_canaries(SCREENING, ref.risk.release,
                                [(ref.clock(), Verdict("in band"))],
                                [SCREENING_BASELINE]) if watching else []
    suite = [Watched("suite", HISTORY, ref.northgate.release, ref.clock(),
                     "contract suite passed")] if watching else []
    claims = Records(
        "claims", "claims", KEYS["claims"], ref.chain, ref.claims_log,
        {SCREENING: Held("risk", ref.risk.terms)}, ledger,
        {"approval": approval(task), "revocation_serial": serial}, canaries,
        [settled], [w for w in witnesses if w])
    record = ref.risk.evidence.records.get(ref.chain)
    handed = ref.northgate.witness(HISTORY, *head_of(ref.risk_log, HISTORY))
    risk = Records(
        "risk", "risk", KEYS["risk"], ref.chain, ref.risk_log,
        {HISTORY: Held("northgate", NORTHGATE_TERMS[ref.northgate.release])},
        CostLedger(), {"record": record} if record else {}, suite, (),
        [handed] if handed else [], ref.risk.evidence.openings.get(ref.chain),
        decides=True)
    return [build_pack("CLM-1003", claims, ask), build_pack("CLM-1003", risk, ask)]


def audit(ref: Referral, stays: Any = MERIDIAN_POLICY) -> AuditPolicy:
    return AuditPolicy(stays, {"claims": WANTS}, {"screen_fraud": "risk"},
                       frozenset({"risk"}), {"claims": ref.claims_log})


# The incident ----------------------------------------------------------------------

async def test_incident_three_questions_are_gaps(clock: Clock) -> None:
    ref = await refer(clock, "2027.04", "v7.5")
    ref.risk.prune(ref.context)  # 30 days after close, the thread is gone
    claims, risk = packs(ref)
    assert (ref.score, claims.crossings[0].terms) == (79.0, "2027.04")
    assert isinstance(claims.decision["approval"], Approval)  # who decided: answered
    assert risk.decision == {} and risk.crossings[0].seam == HISTORY
    assert gaps([claims, risk], PINNED, audit(ref), clock) == [
        Gap("on what", "risk", "thread pruned, no decision record"),
        Gap("where the data went", "northgate", "no regions declared"),
        Gap("is it complete", "claims", "no witness")]


# The fix ----------------------------------------------------------------------------

async def test_fixed_the_joined_packs_leave_nothing_unanswered(clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    ref.risk.prune(ref.context)  # pruned all the same: the record is not the thread
    made = packs(ref)
    [chain] = join(made)
    assert chain.owner == "claims+risk" and len(chain.crossings) >= 2
    assert gaps(made, PINNED, audit(ref), clock) == []
    record = ref.risk.evidence.records[ref.chain]
    assert (record.score, record.basis, record.release) == (79.0, "full", "2027.06")
    assert record.onward and set(record.onward) <= {
        r.digest for r in ref.risk_log.receipts()}


async def test_the_opening_shows_prior_claims_crossed_to_risk_only(
        clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    claims, risk = packs(ref)
    [opening] = risk.openings
    record = ref.risk.evidence.records[ref.chain]
    assert (opening.field, opening.value) == ("prior_claims", 1)
    assert opening.opens(record.inputs["prior_claims"])
    assert set(record.inputs) == set(CLAIM) and record.signals.startswith("sha256:")
    fields = {f for r in claims.crossings for f in r.ours.fields}
    assert "prior_claims" in fields  # Claims sent the history to Risk
    onward = {f for r in risk.crossings for f in r.ours.fields}
    assert onward == {"claim_id"}  # and Risk sent Northgate the claim id alone
    assert claims.openings == ()  # Claims' own pack opens nothing


async def test_a_receipt_dropped_from_the_log_is_caught_by_the_witness(
        clock: Clock, tmp_path: Path) -> None:
    ref = await refer(clock, "2027.06", "v7.6", tmp_path)
    head, count = head_of(ref.claims_log, SCREENING)
    seen = ref.risk.witness(SCREENING, head, count)
    assert seen is not None and witnessed(ref.claims_log, seen)
    path = tmp_path / f"{SCREENING}.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    assert ref.claims_log.broken(SCREENING) == ""  # Chapter 17 sees nothing wrong
    assert not witnessed(ref.claims_log, seen)  # the partner remembers a longer log
    assert seen.verifies(PINNED["risk"])


async def test_erasure_leaves_the_receipts_and_loses_the_opening(clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    ref.risk.evidence.erase(ref.chain)  # the claimant's right to erasure
    made = packs(ref)
    assert all(r.theirs and r.theirs.verifies(PINNED[r.theirs.signer])
               for p in made for r in p.crossings)  # the receipts still verify
    assert ref.risk.evidence.records[ref.chain].verifies(PINNED["risk"])
    assert gaps(made, PINNED, audit(ref), clock) == [
        Gap("on what", "risk", "prior_claims erased")]


async def test_a_record_past_its_retention_is_a_gap(clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    clock.advance(2556 * DAY)  # one day past seven years
    made = packs(ref, watching=False)
    reasons = {g.reason for g in gaps(made, PINNED, audit(ref), clock)}
    assert "past retention: kept 2555 days" in reasons


async def test_the_watching_record_names_the_canaries_and_the_baseline(
        clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    claims, risk = packs(ref)
    assert {w.kind for w in claims.watching} == {"canary", "baseline"}
    assert all(w.release == "2027.06" for w in claims.watching)
    assert [w.kind for w in risk.watching] == ["suite"]
    bare = packs(ref, watching=False)
    assert gaps(bare, PINNED, audit(ref), clock) == [
        Gap("on what", "risk", "no suite run for 2027.06 within 28 days"),
        Gap("on what", "northgate", "no suite run for v7.6 within 28 days")]
    clock.advance(40 * DAY)  # a run taken six weeks after the decision is not near it
    assert gaps(packs(ref), PINNED, audit(ref), clock) == [
        Gap("on what", "risk", "no suite run for 2027.06 within 28 days"),
        Gap("on what", "northgate", "no suite run for v7.6 within 28 days")]


async def test_the_pack_carries_the_approval_the_finding_the_serial_and_the_cost(
        clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    claims, _ = packs(ref)
    assert claims.decision["approval"].approver == "adj-117"
    assert claims.decision["revocation_serial"] >= 1
    [finding] = claims.findings
    assert (finding.verdict, finding.authority) == ("settled", ref.chain)
    [cost] = claims.costs
    assert cost.price == 0.62 and cost.task_id == claims.crossings[0].ours.task_id
    assert claims.signer == "claims" and claims.sig


async def test_a_forged_pack_and_a_missing_pack_are_gaps(clock: Clock) -> None:
    ref = await refer(clock, "2027.06", "v7.6")
    claims, risk = packs(ref)
    forged = replace(risk, subject="CLM-1004")  # edited after Risk signed it
    assert Gap("is it complete", "risk", "the pack's signature does not verify") in (
        gaps([claims, forged], PINNED, audit(ref), clock))
    assert gaps([claims], PINNED, audit(ref, STRICT_POLICY), clock) == [
        Gap("on what", "risk", "no pack produced")]
