"""Revocation: the zombie delegation reproduced, then bounded, owner by owner."""

from dataclasses import replace

import pytest
from a2a.client import A2AClientError
from a2a.types import GetTaskRequest
from langchain_core.runnables import RunnableConfig

from meridian.drill import (
    MINUTE,
    T0,
    DrillFederation,
    adjusters_run,
    revocation_drill,
)
from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import (
    EXPECTATIONS,
    GRANTS,
    OWNERS_A2A,
    SEAMS,
    connect_partners,
    with_revocation,
)
from meridian.identity import (
    ADJUSTER,
    ADJUSTER_SESSION,
    ISSUER,
    KEYS,
    LIST_EVERY,
    LIST_GRACE,
    ROOT_LIFETIME,
    MeridianIssuer,
)
from meridian.inprocess import ClaimGraph
from meridian.resume import ResumeGate
from meridian.samples import suspect_claim
from northgate.server import NORTHGATE_V7, NORTHGATE_V7_2
from seamkit.contract import check
from seamkit.delegation import Verified, mint
from seamkit.drill import Clock, probe
from seamkit.errors import (
    CardUntrusted,
    DelegationRefused,
    DelegationRevoked,
    PartnerNeedsAuth,
    PartnerRejected,
)
from seamkit.inventory import inventory_from_graph
from seamkit.revocation import (
    ListUnavailable,
    RevocationList,
    RevocationSource,
    checked,
    checker,
    revoked,
    sign_list,
    still_holds,
)
from tests.wire import as_run

pytestmark = pytest.mark.anyio

AT = T0 + 3 * MINUTE  # 10:03, the suspension


async def order_estimate(fed: DrillFederation) -> ClaimGraph:
    """10:00: the adjuster orders CLM-1002's estimate; Northgate's assessor has it."""
    estimate, _ = await fed.connect()
    assert await fed.run(estimate, suspect_claim(), "clm-1002") == "TaskUnfinished"
    return estimate


# The incident -------------------------------------------------------------------------

async def test_the_incident_revoked_authority_keeps_acting_at_every_owner() -> None:
    fed = DrillFederation(fixed=False)  # Chapter 6's federation
    rows = {r.owner: r for r in await revocation_drill(fed, ADJUSTER, at=AT)}
    assert fed.outcomes["CLM-1001"] == "paid"  # at 10:06, from Claims' cached root
    assert len(fed.partners.finance.executor.received) == 1
    assert rows["Northgate"].tasks["CLM-1002 estimate"] == "completed, billed 3.5"
    assert rows["Claims"].first_refusal == 11 * MINUTE  # the cache re-exchanges
    assert {rows[o].first_refusal for o in ("Risk", "Northgate", "Finance")} == {
        12 * MINUTE}  # every partner accepted her chain until it expired
    assert all(row.acted_after for row in rows.values())


# The drill ----------------------------------------------------------------------------

async def test_the_drill_no_owner_acts_after_the_revocation_horizon() -> None:
    fed = DrillFederation()
    rows = await revocation_drill(fed, ADJUSTER, at=T0 + 3 * MINUTE)
    for row in rows:
        assert row.acted_after == ()
        assert row.observed is not None and row.agreed is not None
        assert row.observed <= row.agreed
    northgate = {row.owner: row for row in rows}["Northgate"]
    assert northgate.tasks == {"CLM-1002 estimate": "canceled, not billed"}
    assert fed.partners.northgate.executor.billed == {}
    assert fed.outcomes["CLM-1001"] == "DelegationRefused"  # nothing left Claims
    assert fed.partners.finance.executor.received == []


async def test_the_drill_reports_where_each_owner_noticed() -> None:
    fed = DrillFederation()
    rows = {r.owner: r for r in await revocation_drill(fed, ADJUSTER, at=AT)}
    assert rows["Claims"].first_refusal == 0  # at home there is no copy to go stale
    assert rows["Northgate"].first_refusal == 30  # its copy was pulled at 10:02:30
    assert rows["Northgate"].agreed == 90  # terms v7.2
    assert "revoked" in rows["Northgate"].refused_with
    assert fed.outcomes["CLM-1002 at 10:22"] == "DelegationRevoked"


# The list -----------------------------------------------------------------------------

def test_revocation_is_by_person_and_time_not_by_token() -> None:
    clock = Clock(T0)
    issuer = MeridianIssuer(now=clock)
    old = issuer.exchange(ADJUSTER_SESSION)  # a root no list ever named by id
    rl = issuer.revoke(ADJUSTER, at=AT)
    assert revoked(old, rl) == f"{ADJUSTER} revoked (list 1)"
    assert rl.subjects[ADJUSTER] == AT + ROOT_LIFETIME  # the cut, in expiry terms
    reinstated = replace(old[0], expires=AT + ROOT_LIFETIME)  # issued at 10:03 exactly
    assert revoked((reinstated,), rl) == ""
    with pytest.raises(DelegationRefused, match="ended session"):
        issuer.exchange(ADJUSTER_SESSION)


def test_a_partner_reads_the_list_only_with_the_key_it_pinned() -> None:
    clock = Clock(T0)
    issuer = MeridianIssuer(now=clock)
    issuer.revoke(ADJUSTER, at=T0)
    forged = sign_list(RevocationList(ISSUER, 99, T0, T0 + 60, {}), KEYS["claims"])
    source = RevocationSource(ISSUER, KEYS[ISSUER].public_key(), lambda: forged)
    with pytest.raises(DelegationRefused, match="stale"):  # never "nothing revoked"
        source.current(T0)
    good = RevocationSource(ISSUER, KEYS[ISSUER].public_key(), issuer.served_list)
    assert good.current(T0).serial == 1
    good.fetch = lambda: replace(issuer.publish(T0), serial=0)  # an old list replayed
    clock.advance(61)
    assert good.current(clock()).serial == 1


async def test_a_stale_list_refuses_new_work_on_that_seam_only() -> None:
    fed = DrillFederation()
    _, claims = await fed.connect()
    northgate_list = fed.partners.northgate.executor.revocations[ISSUER]

    def down() -> RevocationList:
        raise ListUnavailable("idp.meridian.test")

    northgate_list.fetch = down  # Northgate cannot reach Meridian; Risk can
    fed.clock.advance(LIST_STALE := 60 + 30 + 1)
    fed.issuer.serving = True
    with pytest.raises(PartnerRejected, match="revocation list from meridian-idp"):
        await claims.ainvoke({**suspect_claim(), "claim_id": "CLM-1001",
                              "damaged_parts": ["door"], "amount": 3_000.0,
                              "days_to_report": 2, "prior_claims": 0},
                             adjusters_run("stale"))
    assert LIST_STALE > 90
    screened = fed.partners.risk.guard.log[-1]
    assert screened["refused"] == ""  # the other seams still run


# Act points ---------------------------------------------------------------------------

async def test_cancel_is_the_one_call_a_revoked_chain_may_make() -> None:
    fed = DrillFederation()
    estimate = await order_estimate(fed)
    fed.clock.t = AT
    fed.issuer.revoke(ADJUSTER, AT)
    fed.clock.advance(MINUTE)  # past Northgate's pull cadence
    with pytest.raises(DelegationRevoked):  # Claims' resume gate, on the next read
        await estimate.ainvoke(suspect_claim(), adjusters_run("clm-1002"))
    log = fed.partners.northgate.guard.log
    cancel = [e for e in log if e["method"] == "CancelTask"]
    assert cancel and cancel[-1]["refused"] == ""  # accepted under a revoked chain
    assert fed.partners.northgate.executor.canceled  # the task ended at Northgate
    connections = await connect_partners(fed.partners.apps(), issuer=fed.issuer)
    task = cancel[-1]
    northgate = connections["repair_estimate"]
    cancel_run: RunnableConfig = {"configurable": {"thread_id": "x",
                                                   "cancel_for": ADJUSTER}}
    with as_run(cancel_run), pytest.raises(A2AClientError, match="403"):
        await northgate.client.get_task(GetTaskRequest(id="any"))  # nothing else
    assert task["subject"] == ADJUSTER
    assert fed.partners.northgate.executor.billed == {}


async def test_expiry_is_renewable_by_the_same_person_for_the_same_claim() -> None:
    fed = DrillFederation()
    estimate = await order_estimate(fed)
    fed.clock.t = T0 + 21 * MINUTE  # her 10:00 chain expired at 10:15
    fed.assessors.set()
    await fed.settle()
    [act] = fed.partners.northgate.executor.acts
    assert act["holds"] == "expired"  # auth-required, the estimate kept
    fed.clock.advance(MINUTE)
    with pytest.raises(CardUntrusted, match="8,300"):  # Chapter 5's ceiling, after
        await estimate.ainvoke(suspect_claim(), adjusters_run("clm-1002"))
    resumed = fed.partners.northgate.executor.acts[-1]
    assert (resumed["point"], resumed["holds"]) == ("resume", "ok")
    assert list(fed.partners.northgate.executor.billed.values()) == [3.5]


async def test_a_reassigned_claim_is_not_renewed_under_another_person() -> None:
    fed = DrillFederation()
    estimate = await order_estimate(fed)
    fed.clock.t = T0 + 21 * MINUTE
    fed.assessors.set()
    await fed.settle()
    entry = (await estimate.aget_state(adjusters_run("clm-1002"))).values[
        "ledger"]["repair_estimate"]
    fed.issuer.sessions["session-adj-204"] = ("adjuster:adj-204",
                                               frozenset({"CLM-1002"}))
    gate = ResumeGate(SEAMS["repair_estimate"], fed.connections["repair_estimate"],
                      issuer=fed.issuer)
    theirs: RunnableConfig = {"configurable": {"thread_id": "clm-1002",
                                               "session": "session-adj-204"}}
    with as_run(theirs), pytest.raises(PartnerNeedsAuth, match="same person"):
        await gate.recheck(entry["task_id"], entry["authority"], "auth-required",
                           "CLM-1002")
    assert fed.partners.northgate.executor.billed == {}  # nothing was renewed


def test_still_holds_tells_revoked_from_expired() -> None:
    clock = Clock(T0)
    issuer = MeridianIssuer(now=clock)
    root = issuer.exchange(ADJUSTER_SESSION)
    chain = (*root, mint(root, KEYS["claims"], "claims", "northgate",
                         GRANTS["repair_estimate"], frozenset({"CLM-1002"})))
    held = Verified(chain, "chain:x", ADJUSTER, "meridian")
    sources = {ISSUER: RevocationSource(ISSUER, KEYS[ISSUER].public_key(),
                                        issuer.served_list)}
    assert still_holds(held, sources, clock()) == "ok"
    clock.t = T0 + ROOT_LIFETIME + 61
    assert still_holds(held, sources, clock()) == "expired"
    issuer.revoke(ADJUSTER, AT)
    clock.advance(61)
    assert still_holds(held, sources, clock()) == "revoked"  # revoked wins
    issuer.serving = False
    clock.advance(LIST_EVERY + LIST_GRACE + 61)
    assert still_holds(held, sources, clock()) == "unknown"  # a stale list


def test_risk_cannot_mint_onward_under_a_revoked_chain() -> None:
    clock = Clock(T0)
    issuer = MeridianIssuer(now=clock)
    root = issuer.exchange(ADJUSTER_SESSION)
    check = checker({ISSUER: RevocationSource(ISSUER, KEYS[ISSUER].public_key(),
                                              issuer.served_list)})
    parent = checked(lambda: root, check, clock)
    assert parent() == root
    issuer.revoke(ADJUSTER, AT)
    clock.t = AT + 61
    with pytest.raises(DelegationRevoked, match="revoked"):
        parent()


# The contract and the inventory -------------------------------------------------------

def test_the_seam_contract_holds_the_revocation_term() -> None:
    expects = EXPECTATIONS["repair_estimate"]
    assert expects.revocation_seconds == 120
    assert check(NORTHGATE_V7_2, expects, MERIDIAN_POLICY) == []  # 90 <= 120
    v7 = replace(NORTHGATE_V7, skills=NORTHGATE_V7_2.skills)  # v7, with Ch 8's notes
    [breach] = check(v7, expects, MERIDIAN_POLICY)
    assert breach.detail.endswith("states no revocation horizon")
    slow = replace(NORTHGATE_V7_2, revocation_seconds=900)
    [breach] = check(slow, expects, MERIDIAN_POLICY)
    assert "may act 900 s on a revoked chain" in breach.detail


async def test_the_inventory_names_each_seams_revocation_horizon() -> None:
    fed = DrillFederation()
    _, claims = await fed.connect()
    inventory = with_revocation(inventory_from_graph(claims, OWNERS_A2A),
                                fed.connections)
    answers = {e.callee: e.answers["authority"] for e in inventory.entries}
    assert answers["repair_estimate"].endswith("revoked within 90 s")
    assert answers["fraud_screening"].endswith("no revocation term")


async def test_a_probe_is_refused_once_the_list_names_her() -> None:
    fed = DrillFederation()
    await order_estimate(fed)
    chain = fed.held_chains()["Northgate"]
    http = fed.probes()["Northgate"]
    assert await probe(http, chain) == ""
    fed.clock.t = AT
    fed.issuer.revoke(ADJUSTER, AT)
    fed.clock.advance(MINUTE)
    assert "revoked" in await probe(http, chain)
