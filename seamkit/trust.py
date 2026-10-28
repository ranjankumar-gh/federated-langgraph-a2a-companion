"""The card trust policy: which fetched cards a caller may send data and authority to.

Fetching a card is discovery. Trusting it is a decision the caller makes against facts
it pinned out of band, per owner: the origins the card may come from and send to, the
keys that may sign it (or, for an owner pinned by digest, the exact cards approved), and
the skills the caller has approved. `decide` makes that decision on every fetched card.

A signature proves who published a card, never how the agent behind it behaves. So each
approved skill also earns its authority under one card digest from the answers the
caller has observed: the trust gradient. A new card starts unproven.
"""

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from a2a.types import AgentCard
from a2a.utils.signing import (
    InvalidSignaturesError,
    NoSignatureError,
    create_signature_verifier,
)
from jwt.api_jwk import PyJWK
from jwt.exceptions import InvalidKeyError

from seamkit.cards import card_digest

trust_logger = logging.getLogger("seamkit.trust")


@dataclass
class TrackRecord:
    """What the caller observed of one skill under one card digest, by task id.

    Task ids, not counts: re-reading a task Claims already saw is not a second answer.
    """

    clean: set[str] = field(default_factory=set)
    malformed: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Gradient:
    """How one skill earns authority under one card: settings, not measurements."""

    output: str = ""  # the answer field whose value the skill decides, if any
    unproven_ceiling: float | None = None  # the most that value may be while unproven
    proven_after: int = 20  # clean answers under one digest

    def tier(self, record: TrackRecord) -> str:
        if record.malformed:  # one malformed answer under this card: nothing is sent
            return "suspended"
        return "proven" if len(record.clean) >= self.proven_after else "unproven"

    def ceiling(self, record: TrackRecord) -> float | None:
        """The most this skill's answer may decide now; None is no limit."""
        return self.unproven_ceiling if self.tier(record) == "unproven" else None


@dataclass(frozen=True)
class PartnerTrust:
    """What the caller pinned, out of band, about one owner's card. Caller-owned."""

    owner: str
    origins: frozenset[str]  # where the card may come from, and where it may send
    skills: frozenset[str]  # the skills this caller has approved for this owner
    keys: Mapping[str, PyJWK] = field(default_factory=dict)  # by key id: a key pin
    digests: frozenset[str] = frozenset()  # exact cards approved: a digest pin
    refetch_seconds: int = 3600  # a refetched card with a new digest is decided again
    gradients: Mapping[str, Gradient] = field(default_factory=dict)  # per skill


@dataclass(frozen=True)
class TrustDecision:
    """The policy's verdict on one fetched card, logged with the card's digest."""

    owner: str
    digest: str
    refused: str = ""  # why the card is not the owner's; "" when it is trusted
    pinned_by: str = ""  # "key" or "digest"
    skills: frozenset[str] = frozenset()  # approved and on this card: callable
    inflated: tuple[str, ...] = ()  # on this card, approved by nobody: never callable

    @property
    def trusted(self) -> bool:
        return not self.refused

    def covers(self, card: AgentCard) -> bool:
        """Whether this decision is about `card`: a trust hook for authority (Ch 6)."""
        return self.trusted and card_digest(card) == self.digest


def origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def decide(card: AgentCard, fetched_from: str, trust: PartnerTrust,
           uses: Collection[str] = ()) -> TrustDecision:
    """Is this the card of the owner the caller expected? Each check can refuse it."""
    digest = card_digest(card)
    urls = (fetched_from, *(i.url for i in card.supported_interfaces))
    if stray := [u for u in urls if origin(u) not in trust.origins]:  # 1. origin
        return TrustDecision(trust.owner, digest,
                             f"{origin(stray[0])} is not an origin pinned for "
                             f"{trust.owner}")
    if trust.keys:  # 2. signed by a key the caller holds, never one the card names
        if why := _unsigned(card, trust):
            return TrustDecision(trust.owner, digest, why)
    elif digest not in trust.digests:  # 2. or exactly a card the caller approved
        return TrustDecision(trust.owner, digest,
                             f"{digest[:19]} is not a card pinned for {trust.owner}")
    offered = {skill.id for skill in card.skills}
    if missing := sorted(set(uses) - (offered & trust.skills)):  # 3. what we call
        return TrustDecision(trust.owner, digest,
                             f"{', '.join(missing)} is not approved on this card")
    return TrustDecision(trust.owner, digest, "",
                         "key" if trust.keys else "digest",
                         frozenset(offered & trust.skills),
                         tuple(sorted(offered - trust.skills)))  # 4. inflation


def _unsigned(card: AgentCard, trust: PartnerTrust) -> str:
    """Why no signature on the card verifies with a key pinned for its owner, or ""."""

    def pinned_key(kid: str | None, jku: str | None) -> PyJWK:
        # jku, the key URL the card's own header names, is never fetched.
        if kid is None or kid not in trust.keys:
            raise InvalidKeyError(f"key {kid} is not pinned for {trust.owner}")
        return trust.keys[kid]

    try:
        create_signature_verifier(pinned_key, ["EdDSA"])(card)
    except NoSignatureError:
        return "the card is not signed"
    except InvalidSignaturesError:
        return f"no signature on the card verifies with a key pinned for {trust.owner}"
    return ""


def log_decision(decision: TrustDecision) -> None:
    """One record per decision, with the digest, on logger `seamkit.trust`."""
    if not decision.trusted:
        trust_logger.warning("card refused: %s: %s (%s)", decision.owner,
                             decision.refused, decision.digest)
        return
    trust_logger.info("card trusted: %s by %s (%s)", decision.owner,
                      decision.pinned_by, decision.digest)
    if decision.inflated:  # capability inflation: recorded, and never callable
        trust_logger.warning("unapproved skills on %s's card: %s", decision.owner,
                             ", ".join(decision.inflated))
