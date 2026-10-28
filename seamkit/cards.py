"""A card's canonical bytes, its digest, and the partner-side signer (Chapter 5).

The digest covers exactly the bytes a card signature covers: the card's JSON with its
signatures and empty values removed, canonicalized with JCS (RFC 8785). So "this card"
means one thing to the card trust policy that pins it, the partner that signs it, the
trust decision that logs it, and the task ledger entry that records it.

The SDK computes the same payload in a private helper; a test holds the two equal, so
a change on either side fails a test instead of silently splitting what a digest means.
"""

import hashlib
from typing import Any

import rfc8785
from a2a.types import AgentCard
from a2a.utils.signing import create_agent_card_signer
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from google.protobuf.json_format import MessageToDict
from jwt.algorithms import OKPAlgorithm
from jwt.api_jwk import PyJWK


def canonical(card: AgentCard) -> bytes:
    """The bytes a card signature covers: JCS JSON, no signatures, no empty values."""
    as_json = MessageToDict(card)
    as_json.pop("signatures", None)
    return rfc8785.dumps(_without_empty(as_json))


def card_digest(card: AgentCard) -> str:
    """One string that names one card exactly, over the bytes its signature covers."""
    return "sha256:" + hashlib.sha256(canonical(card)).hexdigest()


def _without_empty(value: Any) -> Any:
    # The specification signs the card with empty strings, lists, and objects dropped.
    if isinstance(value, dict):
        kept = {k: v for k, v in ((k, _without_empty(v)) for k, v in value.items())
                if v is not None}
        return kept or None
    if isinstance(value, list):
        items = [v for v in (_without_empty(v) for v in value) if v is not None]
        return items or None
    return None if value == "" else value


def jwk(key: Ed25519PrivateKey | Ed25519PublicKey) -> PyJWK:
    """An Ed25519 key as the JWK a partner hands over at onboarding (OKP, EdDSA)."""
    return PyJWK.from_json(OKPAlgorithm.to_jwk(key), algorithm="EdDSA")


def sign_card(card: AgentCard, key: Ed25519PrivateKey, kid: str,
              jku: str) -> AgentCard:
    """Append a JWS signature over the card's canonical bytes, as the partner does.

    `jku` is where the partner says its keys can be fetched. A caller that follows it
    proves only that the card and the key came from the same server.
    """
    signer = create_agent_card_signer(
        jwk(key), {"kid": kid, "alg": "EdDSA", "jku": jku, "typ": "JOSE"})
    return signer(card)
