"""Claims' receipts, its ownership map, and its seam SLOs, as Meridian deploys them.

Claims signs its half of every receipt with its own key, the one Chapter 6's chains are
already signed with. It checks a partner's half with the key it pinned for that partner
out of band: Northgate's card key (Chapter 5), and for Risk and Finance, the owner keys
inside Meridian's trust domain. No new key, and no new trust decision.

From terms v7.5 Northgate countersigns, and the estimate seam's expectations say so
(`repair_estimate.receipted.toml`): an estimate without Northgate's half is a mismatch.
Risk and Finance countersign by a CI rule inside Meridian, which a flag on each
deployment stands in for here.
"""

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from meridian.identity import KEYS
from meridian.saga import EXPECTED, SAGA_EXPECTATIONS, UNDO_EXPECTATIONS
from meridian.trust import SAGA_TRUST
from seamkit.contract import Expectations, load_expectations
from seamkit.inventory import SeamInventory
from seamkit.ownership import OwnershipMap, load_map
from seamkit.receipt import ReceiptLog, ReceiptMismatch, Receipts
from seamkit.remote import RemoteAgentNode
from seamkit.slo import SeamSLO
from seamkit.trust import PartnerTrust

HERE = Path(__file__).parent
OWNERSHIP = load_map(HERE / "ownership.toml")  # Claims' map
RISK_OWNERSHIP = load_map(HERE / "ownership_risk.toml")  # Risk's own, kept by Risk

CLAIMS_KID = "claims"
# The secret Claims' receipt nonces derive from. Generated per process here; in
# production it lives in Claims' key store, and only Claims ever holds it.
RECEIPT_SECRET = os.urandom(32)

RECEIPTED: dict[str, Expectations] = {
    "repair_estimate": load_expectations(EXPECTED / "repair_estimate.receipted.toml")}


def pinned(trust: Mapping[str, PartnerTrust] = SAGA_TRUST,
           ) -> dict[str, Ed25519PublicKey]:
    """Every key a partner's half may be signed with, by key id: what Claims pinned."""
    keys: dict[str, Ed25519PublicKey] = {
        owner: KEYS[owner].public_key() for owner in ("risk", "finance")}
    for policy in trust.values():
        for kid, jwk in policy.keys.items():
            assert isinstance(jwk.key, Ed25519PublicKey)
            keys[kid] = jwk.key
    return keys


def expectations(seam: str) -> Expectations:
    """The seam's expectations as Claims holds them once receipts are in force."""
    return RECEIPTED.get(seam) or {**SAGA_EXPECTATIONS, **{
        f"{s}.undo": e for s, e in UNDO_EXPECTATIONS.items()}}[seam]


def receipted(nodes: Iterable[RemoteAgentNode], log: ReceiptLog,
              mismatched: Callable[[ReceiptMismatch], None] = lambda exc: None,
              trust: Mapping[str, PartnerTrust] = SAGA_TRUST) -> None:
    """Give each node what it needs to cut receipts, all into one log."""
    keys = pinned(trust)
    for node in nodes:
        wants = expectations(node.seam.name).receipts == "countersigned"
        node.receipts = Receipts(KEYS["claims"], CLAIMS_KID, RECEIPT_SECRET, keys, log,
                                 wants, mismatched)


def with_receipts(inventory: SeamInventory, owners: OwnershipMap,
                  kinds: Mapping[str, str], slos: Mapping[str, SeamSLO | None],
                  ) -> SeamInventory:
    """Fill each seam's accountability column, the last one open, and its paged cell."""
    for entry in inventory.entries:
        held = owners.seams.get(entry.callee)
        if held is None:
            continue
        slo = slos.get(entry.callee)
        target = (f"; SLO {slo.target:.0%} in {slo.seconds:g} s over "
                  f"{slo.window_seconds / 86_400:g} days" if slo else "")
        entry.answers["accountability"] = (
            f"receipts: {kinds.get(entry.callee, 'ours')}; owners: "
            f"{held.caller_oncall} / {held.callee_contact}{target}")
        entry.paged = f"{held.caller_oncall}, {held.callee_contact}"
    return inventory


__all__ = ["CLAIMS_KID", "OWNERSHIP", "RECEIPTED", "RECEIPT_SECRET", "RISK_OWNERSHIP",
           "expectations", "pinned", "receipted", "with_receipts"]
