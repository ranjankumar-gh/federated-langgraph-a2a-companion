"""The seam evidence pack: what each owner can show an auditor about one claim.

Chapter 17 gave every crossing a receipt both owners signed; Chapter 18 gave it a price.
An auditor asks four things of a decision that crossed two companies: who decided, on
what, where the data went, and whether the record is complete. Each owner answers over
its own stores and signs what it hands over, and the pack holds digests and names, never
a claim's values. A value is shown only as an opening, produced when the auditor asks
for one field of one claim, and checked against a digest the owner signed earlier.

The auditor joins the owners' packs by delegation chain and runs `gaps`. Each question a
pack cannot answer comes back as a Gap naming its owner and the reason, which is a
signal like any other and routes through the ownership map as kind "evidence".
"""

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from seamkit.approval import Approval, encode_approval, refuse_approval
from seamkit.canary import Baseline, Verdict
from seamkit.contract import CrossingPolicy, EvidenceTerms, SeamTerms
from seamkit.costs import CostEntry, CostLedger
from seamkit.drill import Clock
from seamkit.inventory import SeamInventory
from seamkit.ownership import Signal
from seamkit.receipt import ReceiptLog, SeamReceipt, digest
from seamkit.reconcile import Finding

DAY = 86_400.0
WINDOW = 28 * DAY  # how far either side of a decision the watching record is read
WHO, ON_WHAT, WENT, COMPLETE = ("who decided", "on what", "where the data went",
                                "is it complete")


def plain(value: Any) -> Any:
    """The JSON form of a record: dataclasses, tuples, sets and dates flattened."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(plain(v) for v in value)
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    return value.isoformat() if isinstance(value, date) else value


def signature(record: Any, key: Ed25519PrivateKey) -> str:
    """An Ed25519 signature over the record's canonical JSON, without its `sig`."""
    unsigned = {k: v for k, v in plain(record).items() if k != "sig"}
    return base64.urlsafe_b64encode(key.sign(rfc8785.dumps(unsigned))).decode()


def genuine(record: Any, key: Ed25519PublicKey) -> bool:
    unsigned = {k: v for k, v in plain(record).items() if k != "sig"}
    try:
        key.verify(base64.urlsafe_b64decode(record.sig), rfc8785.dumps(unsigned))
    except (InvalidSignature, ValueError, binascii.Error):
        return False
    return True


def shut(data: Mapping[str, Any], nonce: str) -> dict[str, str]:
    """One digest per field under the owner's nonce, so a field opens alone."""
    return {name: digest({name: value}, nonce) for name, value in data.items()}


@dataclass(frozen=True)
class Witness:
    """A partner's signature over the head of Claims' log: the count and the digest of
    the last receipt it was shown, taken daily."""

    seam: str
    head: str
    count: int
    at: float
    signer: str
    sig: str = ""

    def verifies(self, key: Ed25519PublicKey) -> bool:
        return genuine(self, key)


def head_of(log: ReceiptLog, seam: str) -> tuple[str, int]:
    """What Claims shows a partner to witness: the last digest and the count."""
    held = list(log.receipts([seam]))
    return (held[-1].digest if held else ""), len(held)


def witness_head(key: Ed25519PrivateKey, kid: str, seam: str, head: str, count: int,
                 at: float) -> Witness:
    """The partner's side: sign the head it was shown. It signs what it saw, no more."""
    unsigned = Witness(seam, head, count, at, kid)
    return replace(unsigned, sig=signature(unsigned, key))


def witnessed(log: ReceiptLog, w: Witness) -> bool:
    """True if the log still reaches the head the partner signed."""
    if w.count < 1 or log.broken(w.seam):  # Chapter 17's check: the chain holds
        return False
    held = list(log.receipts([w.seam]))
    return len(held) >= w.count and held[w.count - 1].digest == w.head


@dataclass(frozen=True)
class Opening:
    """One field of one claim, produced when an auditor asks and never kept in the log.
    An erased value is None with no nonce: the digest stays, and nothing opens it."""

    field: str
    value: Any
    nonce: str

    def opens(self, closed: str) -> bool:
        return self.value is not None and digest(
            {self.field: self.value}, self.nonce) == closed


@dataclass(frozen=True)
class Gap:
    """A question the pack cannot answer, whose it is, and why."""

    question: str
    owner: str
    reason: str


@dataclass(frozen=True)
class Watched:
    """One thing done to check a partner's behaviour: a canary verdict ("canary"), an
    accepted baseline ("baseline") or a contract-test run ("suite")."""

    kind: str
    seam: str
    release: str  # the partner's release the check ran against
    at: float
    detail: str


def watched_canaries(seam: str, release: str, history: Sequence[tuple[float, Verdict]],
                     accepted: Sequence[Baseline]) -> list[Watched]:
    """Chapter 15's record of a seam, as the pack keeps it."""
    return ([Watched("canary", seam, release, at, v.kind) for at, v in history]
            + [Watched("baseline", seam, release, b.accepted_at, b.version)
               for b in accepted])


@dataclass(frozen=True)
class Held:
    """The terms in force on one seam, and the owner who answers for them."""

    owner: str
    terms: SeamTerms


@dataclass(frozen=True)
class EvidencePack:
    """What one owner hands an auditor about one claim, signed by that owner."""

    subject: str  # the claim, as the auditor named it
    chain_id: str
    owner: str
    decision: Mapping[str, Any]  # the owner's signed record, approval or list serial
    crossings: tuple[SeamReceipt, ...]
    openings: tuple[Opening, ...]
    terms: Mapping[str, Held]  # seam -> the terms in force
    watching: tuple[Watched, ...]
    findings: tuple[Finding, ...]
    costs: tuple[CostEntry, ...]
    witnesses: tuple[Witness, ...]
    gaps: tuple[Gap, ...]  # what the owner already knows it cannot show
    signer: str
    sig: str = ""


@dataclass
class Records:
    """Everything one owner keeps about a chain, each in its own store."""

    owner: str
    kid: str
    key: Ed25519PrivateKey
    chain_id: str
    log: ReceiptLog
    terms: Mapping[str, Held]
    costs: CostLedger
    decision: Mapping[str, Any]
    watching: Sequence[Watched] = ()
    findings: Sequence[Finding] = ()
    witnesses: Sequence[Witness] = ()
    openings: Mapping[str, Opening] | None = None  # None: this owner keeps none
    decides: bool = False  # this owner's own decision is part of the question


def build_pack(subject: str, records: Records, ask: Sequence[str]) -> EvidencePack:
    """One owner's pack for one chain. `ask` names the fields to open, no others."""
    chain = records.chain_id
    crossings = tuple(r for r in records.log.receipts() if r.authority == chain)
    seams, tasks = {r.seam for r in crossings}, {r.ours.task_id for r in crossings}
    kept = records.openings
    openings = () if kept is None else tuple(
        kept.get(name) or Opening(name, None, "") for name in ask)
    declared = (Gap(ON_WHAT, records.owner, "thread pruned, no decision record"),
                ) if records.decides and "record" not in records.decision else ()
    pack = EvidencePack(
        subject, chain, records.owner, dict(records.decision), crossings, openings,
        {s: h for s, h in records.terms.items() if s in seams},
        tuple(w for w in records.watching if w.seam in seams),
        tuple(f for f in records.findings if f.authority == chain),
        tuple(c for c in records.costs.entries() if c.task_id in tasks),
        tuple(w for w in records.witnesses if w.seam in seams), declared, records.kid)
    return replace(pack, sig=signature(pack, records.key))


def join(packs: Sequence[EvidencePack]) -> list[EvidencePack]:
    """One pack per delegation chain: the owners' packs side by side, unsigned."""
    chains: dict[str, list[EvidencePack]] = {}
    for pack in packs:
        chains.setdefault(pack.chain_id, []).append(pack)
    return [EvidencePack(
        group[0].subject, chain, "+".join(p.owner for p in group),
        {k: v for p in group for k, v in p.decision.items()},
        tuple(c for p in group for c in p.crossings),
        tuple(o for p in group for o in p.openings),
        {k: v for p in group for k, v in p.terms.items()},
        tuple(w for p in group for w in p.watching),
        tuple(f for p in group for f in p.findings),
        tuple(c for p in group for c in p.costs),
        tuple(w for p in group for w in p.witnesses),
        tuple(g for p in group for g in p.gaps), "joined")
        for chain, group in chains.items()]


@dataclass(frozen=True)
class AuditPolicy:
    """What the auditor holds the packs to, and the logs the owners handed over."""

    crossing: CrossingPolicy  # classes, and where each class may be processed
    wants: Mapping[str, EvidenceTerms]  # caller's key id -> the evidence it expects
    decides: Mapping[str, str]  # skill -> the owner whose record says why it decided
    internal: frozenset[str]  # owners inside the caller, whose regions are its own
    logs: Mapping[str, ReceiptLog]  # caller's key id -> its receipt log


def gaps(packs: Sequence[EvidencePack], pinned: Mapping[str, Ed25519PublicKey],
         policy: AuditPolicy, clock: Clock) -> list[Gap]:
    """What the packs leave unanswered: signatures, openings, logs, regions, age."""
    found = [g for pack in packs for g in _forged(pack, pinned)]
    for chain in join(packs):
        found += [*chain.gaps, *_decided(chain, policy, pinned)]
        for receipt in chain.crossings:
            held = chain.terms.get(receipt.seam)
            found += _went(receipt, held, policy)
            found += _watched(receipt, held, chain.watching)
            found += _aged(receipt, held, policy, clock())
        found += _opened(chain)
        found += _complete(chain, pinned, policy)
    return found


def _stamp(at: str) -> float:
    return datetime.fromisoformat(at).timestamp()


def _forged(pack: EvidencePack, pinned: Mapping[str, Ed25519PublicKey]) -> list[Gap]:
    """A pack, or a partner's half in it, that its pinned key does not vouch for."""
    key = pinned.get(pack.signer)
    found = [] if key and genuine(pack, key) else [
        Gap(COMPLETE, pack.owner, "the pack's signature does not verify")]
    for r in pack.crossings:
        half = r.theirs
        if half and not ((k := pinned.get(half.signer)) and half.verifies(k)):
            found.append(Gap(ON_WHAT, half.signer, f"{r.seam}: the partner's half "
                                                   "does not verify"))
    return found


def _decided(chain: EvidencePack, policy: AuditPolicy,
             pinned: Mapping[str, Ed25519PublicKey]) -> list[Gap]:
    """Who decided: a person's approval, and the decider's signed record of why."""
    found: list[Gap] = []
    for r in chain.crossings:
        owner = policy.decides.get(r.ours.skill)
        record, approval = chain.decision.get("record"), chain.decision.get("approval")
        if owner is None:
            continue
        if owner not in chain.owner.split("+"):
            found.append(Gap(ON_WHAT, owner, "no pack produced"))
        elif record and not ((k := pinned.get(record.signer)) and genuine(record, k)):
            found.append(Gap(ON_WHAT, owner, "the decision record does not verify"))
        if not isinstance(approval, Approval) or refuse_approval(
                {"approval": encode_approval(approval)}, pinned, approval.role,
                approval.task_id, approval.claim, approval.amount, approval.asked):
            found.append(Gap(WHO, "claims", "no approval that verifies"))
    return found


def _went(receipt: SeamReceipt, held: Held | None, policy: AuditPolicy) -> list[Gap]:
    """Where the fields a receipt names were processed: declared, and allowed."""
    if held is None:
        return [Gap(WENT, receipt.seam, "no terms on record")]
    if held.owner in policy.internal:
        return []
    skill = held.terms.skills.get(receipt.ours.skill)
    if skill is None or not skill.regions:
        return [Gap(WENT, held.owner, "no regions declared")]
    found = []
    for name in receipt.ours.fields:
        kind = policy.crossing.classes.get(name, "unclassified")
        home = policy.crossing.stays_in.get(kind, frozenset(skill.regions))
        if not set(skill.regions) <= home:
            found.append(Gap(WENT, held.owner, f"{name} ({kind}) went to "
                                               f"{sorted(skill.regions)}"))
        if skill.reads and name not in skill.reads:
            found.append(Gap(WENT, held.owner, f"{name} is not a field it declares"))
    return found


def _watched(receipt: SeamReceipt, held: Held | None,
             watching: Sequence[Watched]) -> list[Gap]:
    """The answer was checked against the release that gave it, near the decision."""
    at = _stamp(receipt.ours.at)
    if any(w.seam == receipt.seam and w.release == receipt.partner_release
           and w.kind != "baseline" and abs(w.at - at) <= WINDOW for w in watching):
        return []
    return [Gap(ON_WHAT, held.owner if held else receipt.seam,
                f"no suite run for {receipt.partner_release} within 28 days")]


def _aged(receipt: SeamReceipt, held: Held | None, policy: AuditPolicy,
          now: float) -> list[Gap]:
    """A record older than the retention its owner promised is owed no longer."""
    have = (held.terms.evidence if held else None) or policy.wants.get(
        receipt.ours.signer)
    if have is None or now - _stamp(receipt.ours.at) <= have.retain_days * DAY:
        return []
    return [Gap(COMPLETE, held.owner if held else receipt.seam,
                f"past retention: kept {have.retain_days} days")]


def _opened(chain: EvidencePack) -> list[Gap]:
    """Each opening against the digest its owner signed. An erased value is a gap."""
    record, found = chain.decision.get("record"), []
    owner = record.signer if record else chain.owner
    for o in chain.openings:
        closed = record.inputs.get(o.field) if record else None
        if o.value is None:
            found.append(Gap(ON_WHAT, owner, f"{o.field} erased"))
        elif closed is None or not o.opens(closed):
            found.append(Gap(ON_WHAT, owner, f"{o.field} does not open its digest"))
    return found


def _complete(chain: EvidencePack, pinned: Mapping[str, Ed25519PublicKey],
              policy: AuditPolicy) -> list[Gap]:
    """A log a caller expects witnessed must reach the head its partner last signed."""
    found = []
    for caller, seam in sorted({(r.ours.signer, r.seam) for r in chain.crossings}):
        want = policy.wants.get(caller)
        if want is None or want.witness != "daily":
            continue
        seen = sorted((w for w in chain.witnesses if w.seam == seam),
                      key=lambda w: w.at)
        log = policy.logs.get(caller)
        if not seen:
            found.append(Gap(COMPLETE, caller, "no witness"))
        elif not ((k := pinned.get(seen[-1].signer)) and seen[-1].verifies(k)):
            found.append(Gap(COMPLETE, caller, "the witness does not verify"))
        elif log is None or not witnessed(log, seen[-1]):
            found.append(Gap(COMPLETE, caller, "the log no longer reaches the head "
                                               f"{seen[-1].signer} witnessed"))
    return found


def gap_signals(found: Sequence[Gap], seam: str, at: float = 0.0) -> list[Signal]:
    """Each gap as a signal on a seam: the ownership map says who hears it."""
    return [Signal(seam, "evidence", f"{g.question}: {g.owner}: {g.reason}", at)
            for g in found]


def with_evidence(inventory: SeamInventory, terms: Mapping[str, SeamTerms],
                  ) -> SeamInventory:
    """Add each seam's evidence terms to the accountability column."""
    for entry in inventory.entries:
        held = terms.get(entry.callee)
        if held is None or held.evidence is None:
            continue
        kept = held.evidence
        entry.answers["accountability"] = (
            entry.answers.get("accountability", "") + f"; evidence: kept "
            f"{kept.retain_days} days, produced within {kept.produce_within_days}, "
            f"witnessed {'daily' if kept.witness == 'daily' else 'never'}")
    return inventory


__all__ = ["AuditPolicy", "EvidencePack", "Gap", "Held", "Opening", "Records",
           "Watched", "Witness", "build_pack", "gap_signals", "gaps", "genuine",
           "head_of", "join", "plain", "shut", "signature", "watched_canaries",
           "with_evidence", "witness_head", "witnessed"]
