"""Hop attribution: which seam a wrong or late field came across, from one owner's side.

A field's provenance tag names the seam that wrote it (Chapter 8). The ledger entry for
that seam names the key the send went under (Chapter 9), and the crossing records with
that key or that task say what each call got (Chapter 3). So from Claims' side a wrong
field walks back to a seam, and the seam's recorded answer is held to Claims'
expectations. The first hop that failed them is the answer. A field Claims computed
walks back through what it was computed from.

The walk stops at Claims' own seams, because Claims holds nothing beyond them. A wrong
answer from Risk that Risk built from Northgate's is attributed to Risk by Claims, and
Risk's own walk, over Risk's records, carries it on to Northgate. When every hop's
answer met Claims' expectations, the hop is Claims: the expectations let it through.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from seamkit.contract import Expectations, violations
from seamkit.remote import CrossingRecord


@dataclass(frozen=True)
class Hop:
    """Where a walk ended: the owner, the seam on the walker's side, and why."""

    owner: str
    seam: str  # "" for a field the walker's own code wrote
    why: str
    # Chapter 17: the receipt for the answer the hop is about, which both owners
    # signed: what Claims cites, one hop further than its own records reach
    receipt: str = ""


def hop_on(seam: str, owner: str, records: Sequence[CrossingRecord],
           key: str = "", task_id: str = "") -> Hop | None:
    """The first recorded call on `seam` that got no answer, or None."""
    for r in records:
        mine = not (key or task_id) or r.idempotency_key == key or (
            task_id and r.task_id == task_id)
        if r.seam == seam and mine and r.error:
            return Hop(owner, seam, f"{r.error} on the {r.call}")
    return None


def attribute(field: str, state: Mapping[str, Any], records: Sequence[CrossingRecord],
              expects: Mapping[str, Expectations]) -> Hop:
    """The first hop, from Claims' side, whose answer failed Claims' expectations."""
    tag = state.get("provenance", {}).get(field)
    if tag is None or not tag["seam"]:  # Claims computed it: walk its inputs first
        for source in (tag or {}).get("derived_from", []):
            if (hop := attribute(source, state, records, expects)).owner != "Claims":
                return hop
        return Hop("Claims", "", f"{field} was computed by Claims from passing inputs")
    seam, e = tag["seam"], expects[tag["seam"]]
    if tag.get("substitute"):
        return Hop("Claims", seam, f"{field} came from {tag['substitute']}")
    entry = state.get("ledger", {}).get(seam, {})
    key, task_id = entry.get("idempotency_key", ""), tag["task_id"]
    signed = entry.get("receipt", "")
    if failed := hop_on(seam, e.owner, records, key, task_id):
        return replace(failed, receipt=signed)
    answer = {theirs: state[ours] for theirs, ours in e.into.items() if ours in state}
    if wrong := violations(answer, e.reads):
        return Hop(e.owner, seam, "; ".join(wrong), signed)
    return Hop("Claims", seam, f"{e.owner}'s answer met Claims' expectations", signed)


__all__ = ["Hop", "attribute", "hop_on"]
