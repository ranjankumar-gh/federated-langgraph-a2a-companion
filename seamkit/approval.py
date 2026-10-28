"""The approval credential: who decided, bound to one decision on one task.

An answer to a partner's question is a value, and a value carries no identity: whoever
resumes the interrupt types it. When the question is an approval, the asker needs to
know who approved, so the approver's identity provider signs a statement of exactly
what was approved: this task, this claim, this amount, this question (by digest), by
this person in this role.

It rides inside the answer, under the caller's ordinary delegation chain. It is never a
chain and never sent as one, so no guard reads it as authority: it authorizes nothing,
and proves one decision. The encoding reuses the delegation chain's (canonical JSON,
an Ed25519 signature by the issuer), so a partner verifies it with a key it already
pinned.
"""

import base64
import binascii
import json
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


@dataclass(frozen=True)
class Approval:
    """One person's yes to one question on one task."""

    issuer: str  # the identity provider that knows who the approver is
    approver: str  # the person, as the issuer names her
    role: str  # "owner/role", as the issuer records it, never as she types it
    task_id: str  # the partner's task the question was asked on
    claim: str
    amount: float
    asked: str  # the question's digest, as the partner sent it
    sig: str = ""


def _payload(approval: Approval) -> bytes:
    unsigned = {k: v for k, v in asdict(approval).items() if k != "sig"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def sign_approval(approval: Approval, key: Ed25519PrivateKey) -> Approval:
    signature = key.sign(_payload(approval))
    return replace(approval, sig=base64.urlsafe_b64encode(signature).decode())


def encode_approval(approval: Approval) -> str:
    return base64.urlsafe_b64encode(json.dumps(asdict(approval)).encode()).decode()


def decode_approval(text: str) -> Approval | None:
    try:
        return Approval(**json.loads(base64.urlsafe_b64decode(text)))
    except (ValueError, TypeError, binascii.Error):
        return None


def refuse_approval(answer: Mapping[str, Any], keys: Mapping[str, Ed25519PublicKey],
                    role: str, task_id: str, claim: str, amount: float, asked: str,
                    already: Collection[str] = ()) -> str:
    """Why this answer is not an approval the asker can rely on, or "" if it is.

    `keys` are the issuers the asker pinned, by name; `already` is who has approved
    this task so far. The answer's own words are never read for who answered.
    """
    approval = decode_approval(str(answer.get("approval", "")))
    if approval is None:
        return "the answer carries no approval credential"
    key = keys.get(approval.issuer)
    if key is None:
        return f"{approval.issuer} is not an issuer this partner pins"
    try:
        key.verify(base64.urlsafe_b64decode(approval.sig), _payload(approval))
    except (InvalidSignature, ValueError, binascii.Error):
        return "the approval's signature does not verify"
    bound = (approval.task_id, approval.claim, approval.amount, approval.asked)
    if bound != (task_id, claim, amount, asked):
        return "the approval is for another task, claim, amount or question"
    if approval.role != role:
        return f"the question was for {role}, not {approval.role}"
    if approval.approver in already:
        return f"{approval.approver} has already approved this task"
    return ""
