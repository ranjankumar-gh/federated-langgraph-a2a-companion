"""Delegation chains: the confused deputy reproduced, then refused, link by link."""

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from a2a.types import GetTaskRequest
from a2a.utils.errors import TaskNotFoundError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.runnables import RunnableConfig

from meridian.federated import (
    GRANTS,
    OWNERS_A2A,
    SEAMS,
    A2APartners,
    connect_partners,
    in_process_http,
    with_authority,
)
from meridian.identity import (
    ADJUSTER,
    ADJUSTER_SESSION,
    KEYS,
    MERIDIAN_TRUSTS,
    SESSIONS,
    MeridianIssuer,
)
from meridian.incident import connect_with_tokens, tokened_http
from meridian.risk_server import NORTHGATE_GRANT, RiskPartner
from meridian.samples import clean_claim, suspect_claim
from meridian.trust import NORTHGATE as NORTHGATE_TRUST
from northgate.server import (
    INSURER_B,
    INSURER_B_KEYS,
    NORTHGATE_URL,
    TRUSTS,
    NorthgatePartner,
    estimate_history,
)
from seamkit.delegation import (
    Chain,
    DelegationAuth,
    Grant,
    Link,
    decode,
    digest,
    encode,
    mint,
    narrows,
    sign,
    verify,
)
from seamkit.inventory import inventory_from_graph
from seamkit.errors import DelegationRefused, PartnerNeedsInput, PartnerRejected
from seamkit.remote import RemoteAgentNode, SeamSpec, connect
from tests.wire import Federation, as_run, thread

pytestmark = pytest.mark.anyio

NOW = time.time()


def curious_assistant(claim: dict[str, Any],
                      seen: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The incident's model: the claim's shop history, then the shop's whole book."""
    if not seen:
        return {"claim_id": claim["claim_id"]}
    if len(seen) == 1 and seen[0].get("shop"):
        return {"shop": seen[0]["shop"]}
    return None


def screening_fields(claim: Mapping[str, Any]) -> dict[str, Any]:
    return {k: claim[k] for k in SEAMS["fraud_screening"].sends}


def chain_to_risk(claim: str = "CLM-1002") -> Chain:
    """The chain Claims sends Risk: the adjuster's root, and Claims' link to Risk."""
    root = MeridianIssuer(now=lambda: NOW).exchange(ADJUSTER_SESSION)
    link = mint(root, KEYS["claims"], "claims", "risk", GRANTS["fraud_screening"],
                frozenset([claim]))
    return (*root, link)


def to_northgate(chain: Chain) -> Chain:
    """Risk's link to Northgate, minted the only way Risk's client can."""
    return (*chain, mint(chain, KEYS["risk"], "risk", "northgate", NORTHGATE_GRANT,
                         frozenset(["CLM-1002"])))


# The incident -------------------------------------------------------------------------

async def test_the_incident_a_forwarded_header_is_served_as_claims() -> None:
    partners = A2APartners(risk=RiskPartner(assistant=curious_assistant, forwards=True),
                           northgate=NorthgatePartner(verifies_chains=False))
    risk = (await connect_with_tokens(partners.apps()))["fraud_screening"]
    node = RemoteAgentNode(SEAMS["fraud_screening"], risk, log=lambda r: None)
    with pytest.raises(PartnerNeedsInput):  # Risk then asks for the police report
        await node.call_once(screening_fields(suspect_claim()))
    asked = partners.northgate.executor.received
    assert [r["request"] for r in asked] == [{"claim_id": "CLM-1002"},
                                             {"shop": "SHOP-7"}]
    assert {r["caller"] for r in asked} == {"Claims"}  # Claims never made either call
    book = estimate_history("meridian", {"shop": "SHOP-7"})["estimates"]
    assert len(book) == 8 and len({e["claim_id"] for e in book}) == 7


async def test_a_seam_that_reads_the_tenant_from_the_request_serves_anyone() -> None:
    partners = A2APartners(northgate=NorthgatePartner(verifies_chains=False))
    http = tokened_http(NORTHGATE_URL, partners.northgate.app, "claims-workload-ng",
                        "claims")
    seam = SeamSpec("history", "Northgate", "estimate_history", ("shop", "insurer"),
                    {"estimates": "estimates"})
    node = RemoteAgentNode(seam, await connect(NORTHGATE_URL, http), log=lambda r: None)
    out = await node.call_once({"shop": "SHOP-7", "insurer": "insurer-b"})
    assert {e["claim_id"][:2] for e in out["estimates"]} == {"B-"}  # another insurer's


# The fix ------------------------------------------------------------------------------

async def test_the_fix_risk_mints_and_northgate_refuses_the_whole_book() -> None:
    partners = A2APartners(risk=RiskPartner(assistant=curious_assistant))
    graph = await Federation(partners).graph()
    paused = await graph.ainvoke(suspect_claim(), thread("d1"))  # police report next
    first, book = partners.northgate.guard.log
    assert first["refused"] == ""
    assert first["path"] == ("meridian-idp", "claims", "risk")
    assert "fields ['shop'] not granted" in book["refused"]
    assert book["subject"] == ADJUSTER and book["chain"] == first["chain"]
    assert paused["ledger"]["fraud_screening"]["authority"] == first["chain"]


async def test_the_whole_claim_still_settles_under_one_chain() -> None:
    partners = A2APartners(risk=RiskPartner(assistant=curious_assistant))
    fed = Federation(partners)
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("d2"))
    assert out["decision"] == "paid"
    chains = {entry["authority"] for entry in out["ledger"].values()}
    assert len(chains) == 1  # every seam's task names the same chain
    served = partners.northgate.executor.received
    assert [r["skill"] for r in served] == ["estimate_history", "estimate_repair"]
    assert served[0]["path"] == ("meridian-idp", "claims", "risk")
    assert served[1]["path"] == ("meridian-idp", "claims")


async def test_a_run_without_a_chain_is_refused_not_defaulted() -> None:
    fed = Federation()
    graph = await fed.graph()
    with pytest.raises(DelegationRefused, match="no adjuster session"):
        await graph.ainvoke(clean_claim(), {"configurable": {"thread_id": "m1"}})
    assert fed.partners.risk.executor.received == []  # nothing left Claims
    # A caller that sends no chain at all is refused by the partner, and says so typed.
    partners = A2APartners()
    bare = in_process_http(NORTHGATE_URL, partners.northgate.app, "claims")
    node = RemoteAgentNode(SEAMS["repair_estimate"], await connect(NORTHGATE_URL, bare))
    with pytest.raises(PartnerRejected, match="no delegation chain"):
        await node.call_once({"claim_id": "CLM-1001", "damaged_parts": ["door"]})


# Verification, check by check ---------------------------------------------------------

def test_a_valid_chain_verifies_and_names_its_tenant() -> None:
    verified = verify(to_northgate(chain_to_risk()), "risk", "northgate", TRUSTS, NOW)
    assert verified.tenant == "meridian" and verified.subject == ADJUSTER
    assert verified.path == ("meridian-idp", "claims", "risk")
    assert decode(encode(verified.chain)) == verified.chain


def test_a_link_wider_than_its_parent_is_refused_by_minter_and_receiver() -> None:
    chain = chain_to_risk()
    wider = Grant(frozenset({"estimate_history"}), frozenset({"claim_id", "shop"}), 0)
    with pytest.raises(DelegationRefused, match="widens the fields"):
        mint(chain, KEYS["risk"], "risk", "northgate", wider, frozenset(["CLM-1002"]))
    parent = chain[-1]
    forged = sign(Link("risk", "northgate", ADJUSTER, frozenset(["CLM-1002"]),
                       wider.skills, wider.fields, 0, parent.expires, digest(parent)),
                  KEYS["risk"])  # a Risk that skips mint() and signs it anyway
    with pytest.raises(DelegationRefused, match="risk: widens the fields"):
        verify((*chain, forged), "risk", "northgate", TRUSTS, NOW)


def test_a_middle_link_widened_is_caught_however_narrow_the_last_one_is() -> None:
    root, claims_link = chain_to_risk()
    widened = sign(replace(claims_link, hops=5, sig=""), KEYS["claims"])
    risk_link = sign(Link("risk", "northgate", ADJUSTER, frozenset(["CLM-1002"]),
                          NORTHGATE_GRANT.skills, NORTHGATE_GRANT.fields, 0,
                          widened.expires, digest(widened)), KEYS["risk"])
    assert narrows(widened, risk_link) == []  # the last hop alone looks fine
    with pytest.raises(DelegationRefused, match="does not spend a hop"):
        verify((root, widened, risk_link), "risk", "northgate", TRUSTS, NOW)


def test_a_chain_presented_by_another_workload_is_refused() -> None:
    chain = to_northgate(chain_to_risk())
    with pytest.raises(DelegationRefused, match="presented by finance, held by risk"):
        verify(chain, "finance", "northgate", TRUSTS, NOW)


def test_a_link_signed_with_a_key_nobody_pinned_is_refused() -> None:
    chain = chain_to_risk()
    stranger = Ed25519PrivateKey.generate()
    forged = sign(replace(chain[-1], sig=""), stranger)
    with pytest.raises(DelegationRefused, match="link from claims is not signed"):
        verify((chain[0], forged), "claims", "risk", MERIDIAN_TRUSTS, NOW)


def test_an_expired_chain_is_refused_and_a_link_cannot_outlive_its_parent() -> None:
    chain = to_northgate(chain_to_risk())
    with pytest.raises(DelegationRefused, match="chain expired"):
        verify(chain, "risk", "northgate", TRUSTS, chain[0].expires + 1)
    root, link = chain_to_risk()
    longer = sign(replace(link, expires=link.expires + 3600, sig=""), KEYS["claims"])
    assert narrows(root, longer) == ["outlives its parent"]


def test_no_hop_is_left_below_northgate() -> None:
    chain = to_northgate(chain_to_risk())
    onward = Grant(frozenset({"estimate_history"}), frozenset({"claim_id"}), 0)
    with pytest.raises(DelegationRefused, match="does not spend a hop"):
        mint(chain, KEYS["risk"], "northgate", "supplier", onward,
             frozenset(["CLM-1002"]))


# Caller binding on a shared skill -----------------------------------------------------

async def test_each_insurer_sees_only_its_own_history() -> None:
    partners = A2APartners()
    root = sign(Link(INSURER_B.root, "insurer-b", "adjuster:b-9", frozenset(["B-2201"]),
                     frozenset({"estimate_history"}), frozenset({"claim_id"}), 1,
                     NOW + 600), INSURER_B_KEYS[INSURER_B.root])
    http = in_process_http(NORTHGATE_URL, partners.northgate.app, "insurer-b")
    conn = await connect(NORTHGATE_URL, http, trust=NORTHGATE_TRUST)
    assert conn.trust is not None
    http.auth = DelegationAuth("insurer-b", INSURER_B_KEYS["insurer-b"], "northgate",
                               Grant(root.skills, root.fields, 0), lambda: (root,),
                               conn.card, conn.trust.covers)
    seam = SeamSpec("history", "Northgate", "estimate_history", ("claim_id",),
                    {"estimates": "estimates"})
    out = await RemoteAgentNode(seam, conn, log=lambda r: None).call_once(
        {"claim_id": "B-2201"})
    assert out["estimates"] == [{"part": "frame", "amount": 6_900.0}]  # B's only
    assert partners.northgate.executor.received[-1]["tenant"] == "insurer-b"


async def test_a_meridian_chain_cannot_name_another_insurer() -> None:
    partners = A2APartners()
    http = in_process_http(NORTHGATE_URL, partners.northgate.app, "risk")
    conn = await connect(NORTHGATE_URL, http, trust=NORTHGATE_TRUST)
    assert conn.trust is not None
    http.auth = DelegationAuth("risk", KEYS["risk"], "northgate", NORTHGATE_GRANT,
                               chain_to_risk, conn.card, conn.trust.covers)
    seam = SeamSpec("history", "Northgate", "estimate_history",
                    ("claim_id", "insurer"), {"estimates": "estimates"})
    node = RemoteAgentNode(seam, conn, log=lambda r: None)
    with pytest.raises(PartnerRejected, match=r"fields \['insurer'\] not granted"):
        await node.call_once({"claim_id": "CLM-1002", "insurer": "insurer-b"})


async def test_another_adjusters_chain_cannot_read_the_task() -> None:
    fed = Federation()
    out = await (await fed.graph()).ainvoke(clean_claim(), thread("o1"))
    estimate = out["ledger"]["repair_estimate"]["task_id"]
    other = MeridianIssuer(sessions={**SESSIONS, "session-adj-204": (
        "adjuster:adj-204", frozenset({"CLM-2000"}))})
    northgate = (await connect_partners(fed.partners.apps(), issuer=other))[
        "repair_estimate"]
    theirs: RunnableConfig = {"configurable": {"thread_id": "o2",
                                               "session": "session-adj-204"}}
    with as_run(theirs), pytest.raises(TaskNotFoundError):  # not "forbidden"
        await northgate.client.get_task(GetTaskRequest(id=estimate))
    with as_run(thread("o3")):  # the adjuster it was started for can still read it
        task = await northgate.client.get_task(GetTaskRequest(id=estimate))
    assert task.id == estimate


async def test_an_untrusted_card_receives_no_authority() -> None:
    partners = A2APartners()
    http = in_process_http(NORTHGATE_URL, partners.northgate.app, "claims")
    conn = await connect(NORTHGATE_URL, http)
    http.auth = DelegationAuth("claims", KEYS["claims"], "northgate",
                               GRANTS["repair_estimate"],
                               MeridianIssuer().run_chain, conn.card,
                               trusted=lambda card: False)
    node = RemoteAgentNode(SEAMS["repair_estimate"], conn, log=lambda r: None)
    with as_run(thread("u1")), pytest.raises(DelegationRefused, match="trust policy"):
        await node.call_once({"claim_id": "CLM-1001", "damaged_parts": ["door"]})
    assert partners.northgate.guard.log == []  # nothing was sent


async def test_the_inventory_reads_each_seams_authority_from_its_grant() -> None:
    graph = await Federation().graph()
    inventory = with_authority(inventory_from_graph(graph, OWNERS_A2A))
    answers = {e.callee: e.answers["authority"] for e in inventory.entries}
    assert answers["fraud_screening"] == (
        "chain for screen_fraud, may pass on estimate_history; 1 hop(s) below")
    assert answers["payments"] == "chain for pay_claim; 0 hop(s) below"
    assert all("authority" not in e.open_properties() for e in inventory.entries)
