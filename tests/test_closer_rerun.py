"""The closer: Chapter 2's Federation Test rerun on Meridian with what each seam measured
end to end, and the one seam whose shape does not survive it."""

from typing import Any

import pytest

from meridian.data_classes import MERIDIAN_POLICY
from meridian.decisions import SENDS
from meridian.rerun import RERUN_DIR, Shapes, crossed, measured, reruns
from seamkit.federation import Rerun

pytestmark = pytest.mark.anyio

_RUN: dict[str, Any] = {}


async def shapes() -> Shapes:
    """Both shapes on the same three claims, run once for the module."""
    if "shapes" not in _RUN:
        _RUN["shapes"] = await measured()
    held: Shapes = _RUN["shapes"]
    return held


async def found() -> dict[str, Rerun]:
    return {r.seam: r for r in reruns(await shapes())}


async def test_each_hub_seam_resends_what_its_record_predicted() -> None:
    peer = (await shapes()).measured_peer
    for seam in ("fraud_screening", "repair_estimate", "payments"):
        assert set(peer[seam].fields) == set(SENDS[seam]), seam
        assert peer[seam].ledger == "Claims" and peer[seam].links == 2


async def test_every_owner_line_holds_and_policy_lookup_stays_a_subgraph() -> None:
    held = await found()
    assert all(r.option_holds for r in held.values())
    policy = held["policy_lookup"]
    assert policy.record.verdict.option == "subgraph" and policy.shape_holds


async def test_the_peer_seam_keeps_its_owner_line_and_loses_its_shape() -> None:
    s, held = await shapes(), await found()
    history = held["estimate_history"]
    assert history.option_holds and history.record.verdict.option == "a2a_agent"
    assert crossed(s.peer, "estimate_history").sent == ("claim_id",)
    assert not history.shape_holds  # what crossed was not what the record protected
    assert [seam for seam, r in held.items() if not r.shape_holds] == [
        "estimate_history"]


async def test_the_hub_shape_moves_the_task_and_shortens_the_chain() -> None:
    s = await shapes()
    peer, hub = s.measured_peer, s.measured_hub
    assert (peer["estimate_history"].links, hub["estimate_history"].links) == (3, 2)
    assert (peer["estimate_history"].ledger, hub["estimate_history"].ledger) == (
        "none", "Claims")
    assert not s.hub.risk.config.seams  # Risk sends on nothing: no outbound controls
    owners = s.hub.claims.config.ownership
    assert owners and owners.seams["estimate_history"].callee == "Northgate"
    records = s.hub.risk_partner.evidence.records.values()
    assert records and all(r.onward == () for r in records)  # nothing Risk sent on


async def test_the_move_has_a_price_and_the_rerun_measures_it() -> None:
    s = await shapes()
    peer, hub = s.measured_peer, s.measured_hub
    asked, screened = peer["estimate_history"].tasks, hub["estimate_history"].tasks
    assert (asked, screened) == (2, 3)  # every screened claim, not the ones chosen
    assert hub["estimate_history"].cost_per_task == 0.5  # Northgate's terms
    grew = hub["fraud_screening"].request_bytes - peer["fraud_screening"].request_bytes
    assert grew > 0 and hub["fraud_screening"].tokens > peer["fraud_screening"].tokens
    sent = hub["fraud_screening"].fields
    assert MERIDIAN_POLICY.classes["estimate_history"] == "estimate"
    assert "estimate" not in MERIDIAN_POLICY.allowed["Risk"] and "estimate_history" in sent


async def test_the_rerun_records_match_the_committed_files() -> None:
    for seam, rerun in (await found()).items():
        committed = (RERUN_DIR / f"{seam}.md").read_text(encoding="utf-8")
        assert committed.replace("\r\n", "\n") == rerun.render(), seam
