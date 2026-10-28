"""Revocation: taking back authority that is already in other owners' hands.

A delegation chain verifies offline, which is why a partner across a company line can
check one without calling the issuer. The same property means a chain cannot be
recalled: whoever holds it can use it until it expires. Revocation puts a bound on
that instead of an instant.

The root issuer signs a revocation list with the key partners already pinned for its
chains, so a partner trusts nothing new to read it. The list kills chains by person
(every root for her that expires before a cut-off) and, rarely, by chain id. A partner
keeps a copy, pulls a fresh one on a cadence, and refuses to use a copy older than the
list's next-update time plus an agreed grace: a stale list is never read as "nothing
revoked". An owner re-checks a task's chain at every act point, the moments a task acts
again, not only when the work arrived.

The encoding is this book's own (canonical JSON, one Ed25519 signature), like the chain.
"""

import base64
import binascii
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from seamkit.delegation import Chain, RevocationCheck, Verified, chain_id
from seamkit.errors import DelegationRefused, DelegationRevoked

logger = logging.getLogger("seamkit.revocation")


@dataclass(frozen=True)
class RevocationList:
    """The root issuer's signed word on which of its chains are dead."""

    issuer: str  # the root issuer whose chains it covers, and whose key signs it
    serial: int  # goes up with every list; a partner never goes back to a lower one
    issued_at: float
    next_update: float  # the next list is due by then; a copy past it plus grace is stale
    subjects: Mapping[str, float]  # person -> her roots that expire before this are dead
    chains: frozenset[str] = frozenset()  # single chains, by chain id
    sig: str = ""


def revoked(chain: Chain, rl: RevocationList) -> str:
    """Why the list kills this chain, or "": by its person first, then by its id."""
    root = chain[0]
    if root.issuer != rl.issuer:
        return ""  # another issuer's list says nothing about this chain
    cut = rl.subjects.get(root.subject)
    if cut is not None and root.expires < cut:
        return f"{root.subject} revoked (list {rl.serial})"
    if chain_id(chain) in rl.chains:
        return f"chain revoked (list {rl.serial})"
    return ""


# Signing and verifying the list -------------------------------------------------------

def _payload(rl: RevocationList) -> bytes:
    data = {k: v for k, v in asdict(rl).items() if k != "sig"}
    data["subjects"] = dict(sorted(rl.subjects.items()))
    data["chains"] = sorted(rl.chains)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def sign_list(rl: RevocationList, key: Ed25519PrivateKey) -> RevocationList:
    signature = key.sign(_payload(rl))
    return replace(rl, sig=base64.urlsafe_b64encode(signature).decode())


def list_signed_by(rl: RevocationList, key: Ed25519PublicKey) -> bool:
    try:
        key.verify(base64.urlsafe_b64decode(rl.sig), _payload(rl))
    except (InvalidSignature, ValueError, binascii.Error):
        return False
    return True


# A partner's copy ---------------------------------------------------------------------

class ListUnavailable(Exception):
    """The issuer's list endpoint could not be reached."""


@dataclass
class RevocationSource:
    """A partner's copy of one issuer's list: pulled on a cadence, verified, fail-closed.

    `fetch` reads the list from the issuer's fixed URL; the tests pass a function so a
    drill can cut it off. The key is the one the partner pinned for the issuer's chains.
    """

    issuer: str
    key: Ed25519PublicKey
    fetch: Callable[[], RevocationList]
    every: float = 60.0  # pull a fresh copy once the one held is this old
    grace: float = 30.0  # how long past next_update a copy may still be used
    latest: RevocationList | None = None
    pulled_at: float = float("-inf")
    pulls: list[float] = field(default_factory=list)  # when each good pull happened
    gap: float = 5.0  # Chapter 11: the least time between two doorbell pulls

    def ring(self, now: float) -> None:
        """A doorbell (Chapter 11): pull now, whatever the cadence, with every check.

        Nothing the ring carries is read. A forged ring costs one fetch of the real
        list, and a flood of them costs one fetch per `gap`.
        """
        if now - self.pulled_at >= self.gap:
            self._pull(now)

    def current(self, now: float) -> RevocationList:
        """The list to check against now, or a refusal: never a stale copy."""
        if now - self.pulled_at >= self.every:
            self._pull(now)
        if self.latest is None or now > self.latest.next_update + self.grace:
            raise DelegationRefused("", f"revocation list from {self.issuer} is stale")
        return self.latest

    def _pull(self, now: float) -> None:
        try:
            fresh = self.fetch()
        except ListUnavailable as exc:  # keep the old copy; current() decides
            logger.warning("revocation list pull failed: %s", exc)
            return
        if fresh.issuer != self.issuer or not list_signed_by(fresh, self.key):
            logger.warning("revocation list from %s is not signed", self.issuer)
            return
        if self.latest is not None and fresh.serial < self.latest.serial:
            logger.warning("revocation list went back to serial %s", fresh.serial)
            return  # an old list replayed would bring revoked chains back
        self.latest, self.pulled_at = fresh, now
        self.pulls.append(now)


def checker(sources: Mapping[str, RevocationSource]) -> RevocationCheck:
    """The guard's hook: why a verified chain may not be used now, or ""."""
    def check(chain: Chain, now: float) -> str:
        source = sources.get(chain[0].issuer)
        if source is None:
            return ""  # this tenant publishes no list; its chains end at expiry
        try:
            return revoked(chain, source.current(now))
        except DelegationRefused as exc:
            return exc.detail
    return check


# Act points ---------------------------------------------------------------------------

class ActPoint(StrEnum):
    """The moments a task acts again on its delegation, each a re-check."""

    ACCEPT = "accept"  # a call arrives: the guard
    RESUME = "resume"  # a paused task is answered or renewed
    ONWARD = "onward"  # the task calls another owner: the minting hook
    COMMIT = "commit"  # the task's result leaves the owner


Holds = Literal["ok", "expired", "revoked", "unknown"]


def still_holds(verified: Verified, sources: Mapping[str, RevocationSource],
                now: float) -> Holds:
    """At an act point: may this task act again on the chain it holds?"""
    source = sources.get(verified.chain[0].issuer)
    if source is not None:
        try:
            if revoked(verified.chain, source.current(now)):
                return "revoked"  # final: the task ends, and nothing renews it
        except DelegationRefused:
            return "unknown"  # a stale list: no act until it can be checked
    if verified.scope.expires <= now:
        return "expired"  # renewable by the same person, for the same claim
    return "ok"


def checked(parent: Callable[[], Chain], check: RevocationCheck,
            now: Callable[[], float]) -> Callable[[], Chain]:
    """The onward act point: a run's chain is re-checked before a link is minted."""
    def chain() -> Chain:
        held = parent()
        if why := check(held, now()):
            raise DelegationRevoked("", why)
        return held
    return chain


def as_dict(rl: RevocationList) -> dict[str, Any]:
    """The list as the issuer serves it, for logs and the drill's report."""
    return {**asdict(rl), "subjects": dict(rl.subjects), "chains": sorted(rl.chains)}
