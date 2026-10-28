"""The closer: the federation gateway. One owner's seams built once from one
configuration, a refusal to start a seam missing a control, the inbound half at the
partner, and Meridian run end to end with every owner behind its own gateway."""

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from langgraph.types import Command

from meridian.assembled import AUGUST, HERE, assemble, august
from meridian.decisions import FACTS
from meridian.identity import ADJUSTER, KEYS
from meridian.resume import ResumeGate
from meridian.samples import clean_claim
from seamkit.drill import Clock
from seamkit.errors import DelegationRefused
from seamkit.federation import Candidate, OwnerFacts, federation_test
from seamkit.gateway import GatewayConfig, UnguardedSeam
from seamkit.inventory import PROPERTIES
from tests.wire import thread

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def clock() -> Iterator[Clock]:
    with august(Clock(AUGUST)) as shared:
        yield shared


async def test_every_seam_carries_every_control_its_policy_requires(
        clock: Clock) -> None:
    a = await assemble(clock)
    for seam, sc in a.claims.config.seams.items():
        on = {p.control for p in a.claims.points(seam)}
        assert set(a.claims.config.requires) <= on | set(sc.waivers), seam
        node = a.claims.built[seam]
        assert isinstance(node, ResumeGate)  # the node every chapter extended
        assert node.breaker and node.receipts and node.seam.rules is not None
    assert a.claims.built["payments"].costs is None  # waived, with its reason
    assert set(a.claims.waived("payments")) == {"budget", "evidence"}


def without_breaker(tmp_path: Path) -> Path:
    """Claims' gateway directory with the [breaker] table cut from payments.toml."""
    here = shutil.copytree(HERE / "claims", tmp_path / "claims")
    text = (here / "payments.toml").read_text(encoding="utf-8")
    start = text.index("[breaker]")
    (here / "payments.toml").write_text(text[:start] + text[text.index("[retry]"):],
                                        encoding="utf-8")
    shutil.copy(HERE.parent / "ownership.toml", tmp_path / "ownership.toml")
    gateway = (here / "gateway.toml").read_text(encoding="utf-8")
    (here / "gateway.toml").write_text(
        gateway.replace('"../../ownership.toml"', '"../ownership.toml"'),
        encoding="utf-8")
    return here


async def test_a_seam_missing_its_breaker_stops_the_gateway_at_start(
        clock: Clock, tmp_path: Path) -> None:
    a = await assemble(clock)
    a.claims.config = GatewayConfig.load(without_breaker(tmp_path))
    with pytest.raises(UnguardedSeam, match="payments would start without breaker"):
        a.claims.check()
    held = a.claims.config.seams["payments"]
    waived = replace(held, waivers={**held.waivers, "breaker": "a reason to review"})
    a.claims.config = replace(a.claims.config,
                              seams={**a.claims.config.seams, "payments": waived})
    a.claims.check()  # a waiver with a reason a reviewer reads lets it start


async def test_the_inventory_fills_every_column_in_one_call(clock: Clock) -> None:
    a = await assemble(clock)
    inventory = a.claims.inventory("claims_intake")
    assert [e.callee for e in inventory.entries] == [
        "fraud_screening", "repair_estimate", "payments"]
    assert all(not e.open_properties() for e in inventory.entries)
    estimate = inventory.entries[1].answers
    assert estimate["contract"].startswith("contract: terms v7.6 checked")
    assert "revoked within 90 s" in estimate["authority"]
    assert set(estimate) == set(PROPERTIES)
    peer = a.risk.inventory("fraud_screening").entries[0]
    assert "ledger waived" in peer.answers["state"]  # what the peer shape gives up


async def test_the_gateway_caches_no_authority(clock: Clock) -> None:
    a = await assemble(clock)
    out = await a.graph.ainvoke(clean_claim(), thread("r1"))  # Finance asks Claims
    asked = out["__interrupt__"][0].value
    revoked_at, sent = clock(), len(a.finance_partner.executor.received)
    a.issuer.revoke(ADJUSTER, revoked_at)  # suspended while the payment waits
    clock.advance(45)
    with pytest.raises(DelegationRefused):  # refused in Claims, at the next act point
        await a.graph.ainvoke(Command(resume=a.approval(asked)), thread("r1"))
    assert len(a.finance_partner.executor.received) == sent  # nothing reached Finance
    assert clock() - revoked_at < 90  # inside the horizon Meridian's seams agree


async def test_three_claims_end_to_end_through_four_owners(clock: Clock) -> None:
    a = await assemble(clock)
    out = {s["claim_id"]: s for s in (await a.run()).values()}
    assert out["CLM-1001"]["payment_ref"] == "PAY-CLM-1001-3000"
    assert (out["CLM-1002"]["fraud_score"], out["CLM-1002"]["decision"]) == (82, "refer")
    assert (out["CLM-1003"]["fraud_score"], out["CLM-1003"]["screening_basis"],
            out["CLM-1003"]["decision"]) == (79, "full", "refer")
    for state in out.values():  # every answer that moved a ledger entry is receipted
        assert all("receipt" in entry for entry in state["ledger"].values())
    assert "repair_estimate" not in out["CLM-1003"]["ledger"]  # referred first


async def test_the_partners_inbound_gateways_sign_and_meter(clock: Clock) -> None:
    a = await assemble(clock)
    await a.run()
    log, risk_key = a.claims.runtime.receipts, KEYS["risk"].public_key()
    screened = list(log.receipts(["fraud_screening"]))
    assert screened and all(r.theirs and r.theirs.verifies(risk_key) for r in screened)
    paid = [r for r in log.receipts(["payments"]) if r.theirs]
    assert paid and all(r.theirs.signer == "finance" for r in paid if r.theirs)
    costs = a.claims.runtime.costs.entries(["fraud_screening"])
    assert costs and all(c.usage for c in costs)  # Risk's signed usage report


def test_a_gateway_several_owners_share_is_an_owner_on_every_seam() -> None:
    platform = OwnerFacts("Platform", "platform-train", "meridian-idp",
                          "meridian-platform")
    to = federation_test(Candidate("claims_intake", "platform_gateway",
                                   FACTS["Claims"], platform, True, False))
    on = federation_test(Candidate("platform_gateway", "fraud_screening", platform,
                                   FACTS["Risk"], True, False))
    assert (to.option, on.option) == ("a2a_agent", "a2a_agent")  # one seam is two
    assert to.answers["owner"] == (True, "Claims / Platform")
