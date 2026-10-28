"""Meridian's identity: signing keys per owner, and the issuer that starts every chain.

The issuer stands in for Meridian's identity provider doing an OAuth 2.0 token
exchange (RFC 8693): an adjuster's session goes in, and a root link comes out, naming
the adjuster as the subject and her open claims as the purpose. In production that is
one call to the identity provider per run; here it is a function with a cache.

Each owner signs its own links with its own key. A partner verifies them against public
keys it pinned out of band, so verifying a chain needs no call back to Meridian.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from seamkit.delegation import Chain, Grant, Link, Trust, chain_id, run_value, sign
from seamkit.errors import DelegationRefused
from seamkit.revocation import (
    ListUnavailable,
    RevocationList,
    RevocationSource,
    revoked,
    sign_list,
)

ISSUER = "meridian-idp"

# One key per signing owner. Generated per process here; in production each owner's
# key lives in that owner's own key store and only its public half is ever shared.
KEYS: dict[str, Ed25519PrivateKey] = {
    owner: Ed25519PrivateKey.generate()
    for owner in (ISSUER, "claims", "risk", "finance")}

# What a partner pins to verify Meridian's chains: the root issuer and every key.
MERIDIAN = Trust("meridian", ISSUER, {o: k.public_key() for o, k in KEYS.items()})
MERIDIAN_TRUSTS: Mapping[str, Trust] = {ISSUER: MERIDIAN}

ADJUSTER_SESSION = "session-adj-117"
ADJUSTER = "adjuster:adj-117"

# The issuer's own records: whose session it is, and which claims she is working.
SESSIONS: dict[str, tuple[str, frozenset[str]]] = {
    ADJUSTER_SESSION: (ADJUSTER, frozenset({"CLM-1001", "CLM-1002"}))}

# Everything a Claims run may ever do on an adjuster's behalf, and two hops below it.
ROOT = Grant(
    skills=frozenset({"screen_fraud", "estimate_history", "estimate_repair",
                      "pay_claim"}),
    fields=frozenset({"claim_id", "amount", "days_to_report", "prior_claims",
                      "damaged_parts", "repair_estimate", "police_report",
                      "payment_memo"}),
    hops=2)

ROOT_LIFETIME = 15 * 60.0  # seconds; every link below expires no later

# Chapter 7: Meridian's settings for its revocation list, not measurements. A partner
# that cannot pull a list for LIST_EVERY + LIST_GRACE seconds stops using it.
LIST_EVERY = 60.0  # a new list is due at least this often
LIST_GRACE = 30.0  # the skew budget the seam contracts already agree
CANCEL_LIFETIME = 60.0  # a cancel root's life, if the list does not end it sooner
REVOCATION_URL = "http://idp.meridian.test/revocations"  # where partners pull it


@dataclass
class MeridianIssuer:
    """The token exchange at the root of every Meridian chain."""

    sessions: dict[str, tuple[str, frozenset[str]]] = field(
        default_factory=lambda: dict(SESSIONS))
    key: Ed25519PrivateKey = field(default_factory=lambda: KEYS[ISSUER])
    now: Callable[[], float] = time.time
    exchanges: int = 0
    _roots: dict[str, Chain] = field(default_factory=dict)
    # Chapter 7: who is revoked (person -> her roots expiring before this are dead),
    # the list that says so, and every root issued, by chain id, for the resume gate.
    revoked: dict[str, float] = field(default_factory=dict)
    serving: bool = True  # False: the list endpoint is down (the drill cuts it)
    issued: dict[str, Link] = field(default_factory=dict)
    _list: RevocationList | None = None

    def exchange(self, session: str) -> Chain:
        """An adjuster's session in, a root link addressed to Claims out."""
        if session not in self.sessions:
            raise DelegationRefused("claims", "unknown or ended session")
        subject, claims = self.sessions[session]
        self.exchanges += 1
        return self._root(subject, claims, self.now() + ROOT_LIFETIME)

    def _root(self, subject: str, claims: frozenset[str], expires: float) -> Chain:
        root = sign(Link(ISSUER, "claims", subject, claims, ROOT.skills, ROOT.fields,
                         ROOT.hops, expires), self.key)
        self.issued[chain_id((root,))] = root
        return (root,)

    def revoke(self, subject: str, at: float) -> RevocationList:
        """Suspend a person: end her sessions, drop her cached roots, sign a list."""
        for session, (who, _) in list(self.sessions.items()):
            if who == subject:
                del self.sessions[session]  # no exchange will answer for her again
                self._roots.pop(session, None)  # the cache was the first zombie
        # Links carry no issue time, so the cut is in expiry terms: every root
        # issued before `at` expires before at + ROOT_LIFETIME.
        self.revoked[subject] = at + ROOT_LIFETIME
        return self.publish(at)

    def publish(self, now: float) -> RevocationList:
        serial = self._list.serial + 1 if self._list else 1
        self._list = sign_list(RevocationList(
            ISSUER, serial, now, now + LIST_EVERY, dict(self.revoked)), self.key)
        return self._list

    def served_list(self) -> RevocationList:
        """What REVOCATION_URL returns: the latest list, re-signed when one is due."""
        if not self.serving:
            raise ListUnavailable(REVOCATION_URL)
        if self._list is None or self.now() >= self._list.next_update:
            return self.publish(self.now())
        return self._list

    def revoked_id(self, chain: str) -> str:
        """Why the chain with this id is dead, or "": Claims' check on a recorded id."""
        root = self.issued.get(chain)
        if root is None:
            return f"{chain} was not issued here"  # unknown authority is no authority
        # The issuer's own list, read in place: at home there is no copy to go stale.
        return revoked((root,), self._list or self.publish(self.now()))

    def subject_of(self, chain: str) -> str:
        root = self.issued.get(chain)
        return root.subject if root else ""

    def cancel_root(self, subject: str) -> Chain:
        """A root for a revoked person, dead on the list from the moment it is minted.

        A guard accepts it for one call only: cancelling a task already bound to her,
        from the workload that sent it. It can start, read, or renew nothing.
        """
        cut = self.revoked.get(subject)
        if cut is None:
            raise DelegationRefused("claims", f"{subject} is not revoked")
        expires = min(self.now() + CANCEL_LIFETIME, cut - 1.0)  # inside the cut
        if expires <= self.now():
            raise DelegationRefused("claims", f"every chain for {subject} has expired")
        claims = frozenset().union(*(r.claims for r in self.issued.values()
                                     if r.subject == subject))
        return self._root(subject, claims, expires)

    def run_chain(self) -> Chain:
        """The root for the current Claims run, exchanged once per session lifetime."""
        if subject := run_value("cancel_for"):  # Chapter 7's resume gate, cancelling
            return self.cancel_root(str(subject))
        session = run_value("session")
        if not session:
            raise DelegationRefused("claims", "this run carries no adjuster session")
        root = self._roots.get(session)
        if root is None or root[0].expires - 60 <= self.now():
            root = self._roots[session] = self.exchange(session)
        return root


# One issuer per Claims process, shared by every partner connection it opens.
CLAIMS_ISSUER = MeridianIssuer()


def revocation_sources(issuer: MeridianIssuer = CLAIMS_ISSUER,
                       ) -> dict[str, RevocationSource]:
    """What a partner holds to check Meridian's list: one source, the pinned key.

    The key is the issuer key the partner already pinned for Meridian's chains.
    """
    return {ISSUER: RevocationSource(ISSUER, MERIDIAN.keys[ISSUER], issuer.served_list,
                                     every=LIST_EVERY, grace=LIST_GRACE)}
