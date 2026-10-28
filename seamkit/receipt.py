"""The seam receipt: one record of one partner answer, signed by both owners.

A crossing record is the caller's word, written by the caller's code into the caller's
log. A receipt is cut at the seam when an answer arrives, and both owners sign the same
two digests: what the caller sent, and what the partner answered. Each digest is sha256
over the canonical JSON (RFC 8785) of the data and a per-task nonce, so a receipt holds
no value and no hash anyone can confirm a guessed value against. Field names are signed
in clear: they are what an auditor asks about.

The partner cuts its half as the answer leaves, in the wrapper around its executor
(seamkit.server.Countersigner), and returns it in the task's metadata under this book's
receipt extension URI. The format is canonical JSON and an Ed25519 signature over named
fields, so a partner on any runtime can produce it without seamkit. The A2A
specification defines signatures for agent cards only; this format is the book's own.

The caller closes the receipt: it checks the partner's half with the card key it pinned
out of band (Chapter 5), compares both digests with its own, signs its half, and appends
the receipt to an append-only log kept apart from the checkpoint. A partner whose terms
declare no receipts gets a one-sided receipt, marked as the caller's word only.
"""

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import rfc8785
from a2a.helpers import get_data_parts
from a2a.types import Task
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from google.protobuf.json_format import MessageToDict
from langgraph.config import get_config

from seamkit.errors import SeamError

RECEIPT_URI = "urn:seamkit:seam-receipt"  # the partner's half, in the task's metadata
NONCE = "receipt_nonce"  # the caller's per-task nonce, in the message's metadata


class ReceiptMismatch(SeamError):
    """The partner's half does not match the caller's: a different request, a different
    answer, a signature that fails, or no half from a partner whose terms promise one.
    The answer is not taken, and both owners are told."""


def digest(data: Mapping[str, Any], nonce: str) -> str:
    """sha256 over the canonical JSON of the data and the task's nonce."""
    raw = rfc8785.dumps({"data": dict(data), "nonce": nonce})
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def nonce_for(secret: bytes, idempotency_key: str) -> str:
    """The caller's nonce for one task, derived so it never needs storing apart."""
    return hmac.new(secret, idempotency_key.encode(), hashlib.sha256).hexdigest()


def thread_of() -> str:
    """The caller's thread the current run belongs to; "" outside a graph run."""
    try:
        return str(get_config().get("configurable", {}).get("thread_id", ""))
    except RuntimeError:
        return ""


def answered(task: Task) -> dict[str, Any]:
    """What a partner said in a task: its artifacts' data and its status message's."""
    message = task.status.message
    return {"artifacts": [p for a in task.artifacts for p in get_data_parts(a.parts)],
            "message": get_data_parts(message.parts) if message.parts else []}


@dataclass(frozen=True)
class ReceiptHalf:
    """One owner's signed statement of one crossing; both halves name the same."""

    skill: str
    task_id: str
    context_id: str
    state: str  # the task state the answer carried
    sent: str  # digest of what was sent, as the signer saw it
    answer: str  # digest of what the partner answered
    fields: tuple[str, ...]  # the names of the fields sent, never their values
    signer: str  # the key id the signature verifies under
    at: str  # when the signer cut it, ISO 8601 in UTC
    sig: str = ""

    def payload(self) -> bytes:
        unsigned = {k: v for k, v in asdict(self).items() if k != "sig"}
        return rfc8785.dumps({**unsigned, "fields": list(self.fields)})

    def signed(self, key: Ed25519PrivateKey) -> "ReceiptHalf":
        sig = base64.urlsafe_b64encode(key.sign(self.payload())).decode()
        return replace(self, sig=sig)

    def verifies(self, key: Ed25519PublicKey) -> bool:
        try:
            key.verify(base64.urlsafe_b64decode(self.sig), self.payload())
        except (InvalidSignature, ValueError, binascii.Error):
            return False
        return True


def half_of(task: Task) -> ReceiptHalf | None:
    """The partner's half, as it came back in the task's metadata; None if absent."""
    raw = MessageToDict(task.metadata).get(RECEIPT_URI) if task.metadata else None
    if not isinstance(raw, dict):
        return None
    try:
        return ReceiptHalf(**{**raw, "fields": tuple(raw.get("fields", []))})
    except TypeError:
        return None


@dataclass(frozen=True)
class SeamReceipt:
    """One partner answer, as both owners signed it, with what the caller knew of it."""

    seam: str
    thread: str  # the caller's own thread: how an auditor finds a claim's receipts
    ours: ReceiptHalf
    theirs: ReceiptHalf | None  # None: a one-sided receipt, the caller's word only
    card_digest: str  # the card the task was sent under (Chapter 5)
    authority: str  # the delegation chain's id, never a token (Chapter 6)
    behaviour: str  # the behaviour version the skill was recorded under (Chapter 15)
    partner_release: str
    terms: str  # the revision of the partner's terms in force
    idempotency_key: str  # joins the receipt to the ledger entry and crossing records
    trace_id: str  # the caller's trace that sent the task
    approval: str = ""  # a signed approval the caller sent, carried whole (Chapter 10)
    outcome: Mapping[str, Any] = field(default_factory=dict)  # a compensation's answer
    previous: str = ""  # the digest of the receipt before it in this seam's log

    @property
    def kind(self) -> str:
        return "countersigned" if self.theirs else "ours"

    def plain(self) -> dict[str, Any]:
        out = asdict(self)
        out["ours"]["fields"] = list(self.ours.fields)
        if self.theirs:
            out["theirs"]["fields"] = list(self.theirs.fields)
        out["outcome"] = dict(self.outcome)
        return out

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(rfc8785.dumps(self.plain())).hexdigest()


def receipt_of(raw: Mapping[str, Any]) -> SeamReceipt:
    def half(h: Mapping[str, Any] | None) -> ReceiptHalf | None:
        return ReceiptHalf(**{**h, "fields": tuple(h["fields"])}) if h else None

    ours = half(raw["ours"])
    assert ours is not None
    return SeamReceipt(**{**raw, "ours": ours, "theirs": half(raw["theirs"])})


def outcome_signed(receipt: SeamReceipt, nonce: str) -> bool:
    """Whether the outcome a receipt carries is the answer the partner signed for.
    The nonce is the task's: Claims derives it again from the key, for an auditor."""
    said = {"artifacts": [dict(receipt.outcome)], "message": []}
    return receipt.theirs is not None and receipt.theirs.answer == digest(said, nonce)


def close(ours: ReceiptHalf, task: Task, pinned: Mapping[str, Ed25519PublicKey],
          countersigned: bool, seam: str) -> ReceiptHalf | None:
    """Check the partner's half against Claims' own: the partner's half, or None."""
    theirs = half_of(task)
    if theirs is None:
        if countersigned:  # the terms promise a half, and none came
            raise ReceiptMismatch(seam, "the answer carries no countersignature",
                                  task.id)
        return None
    key = pinned.get(theirs.signer)
    if key is None or not theirs.verifies(key):
        raise ReceiptMismatch(seam, f"the half is not signed by a key Claims pinned "
                                    f"({theirs.signer})", task.id)
    if theirs.sent != ours.sent or theirs.fields != ours.fields:
        raise ReceiptMismatch(seam, "the partner signed for a request Claims never "
                                    "sent", task.id)
    if theirs.answer != ours.answer:  # hashed as it crossed, before the ingress filter
        raise ReceiptMismatch(seam, "the partner answered something other than what "
                                    "it signed", task.id)
    if (theirs.task_id, theirs.state) != (ours.task_id, ours.state):
        raise ReceiptMismatch(seam, "the half names another task or state", task.id)
    return theirs


@dataclass
class ReceiptLog:
    """Append-only, one chain per seam, kept apart from any checkpoint.

    Each receipt carries the digest of the one before it on its seam, so a receipt
    deleted or edited later breaks the chain. `directory` holds one JSON-lines file
    per seam; None keeps the log in memory, for a test.
    """

    directory: Path | None = None
    kept: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def _lines(self, seam: str) -> list[dict[str, Any]]:
        if self.directory is None:
            return self.kept.setdefault(seam, [])
        path = self.directory / f"{seam}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def append(self, receipt: SeamReceipt) -> SeamReceipt:
        lines = self._lines(receipt.seam)
        last = receipt_of(lines[-1]).digest if lines else ""
        chained = replace(receipt, previous=last)
        if self.directory is None:
            lines.append(chained.plain())
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{receipt.seam}.jsonl").open("a") as fh:
                fh.write(json.dumps(chained.plain(), sort_keys=True) + "\n")
        return chained

    def seams(self) -> list[str]:
        if self.directory is None:
            return sorted(self.kept)
        return sorted(p.stem for p in self.directory.glob("*.jsonl"))

    def receipts(self, seams: Sequence[str] = ()) -> Iterator[SeamReceipt]:
        for seam in seams or self.seams():
            yield from (receipt_of(line) for line in self._lines(seam))

    def broken(self, seam: str) -> str:
        """Where this seam's chain breaks, or "" if every receipt links to the last."""
        last = ""
        for n, line in enumerate(self._lines(seam)):
            receipt = receipt_of(line)
            if receipt.previous != last:
                return f"{seam}: receipt {n} does not follow the one before it"
            last = receipt.digest
        return ""


@dataclass
class Receipts:
    """What a node needs to cut receipts on one seam: Claims' signing key, the partner
    keys Claims pinned, the secret its nonces derive from, and the log.

    `countersigned` is the seam's expectation (Chapter 4's file): an answer with no
    partner half is then a mismatch, not a one-sided receipt. `mismatched` hears every
    mismatch, so the ownership map can send it to both owners.
    """

    key: Ed25519PrivateKey
    kid: str
    secret: bytes
    pinned: Mapping[str, Ed25519PublicKey]
    log: ReceiptLog = field(default_factory=ReceiptLog)
    countersigned: bool = False
    mismatched: Callable[[ReceiptMismatch], None] = field(default=lambda exc: None)

    def nonce(self, idempotency_key: str) -> str:
        return nonce_for(self.secret, idempotency_key)

    def sent(self, data: Mapping[str, Any], idempotency_key: str) -> dict[str, Any]:
        """What the ledger keeps of a message Claims sent: its digest and its names."""
        return {"digest": digest(data, self.nonce(idempotency_key)),
                "fields": sorted(data)}


__all__ = ["NONCE", "RECEIPT_URI", "ReceiptHalf", "ReceiptLog", "ReceiptMismatch",
           "Receipts", "SeamReceipt", "answered", "close", "digest", "half_of",
           "nonce_for", "outcome_signed", "receipt_of", "thread_of"]
