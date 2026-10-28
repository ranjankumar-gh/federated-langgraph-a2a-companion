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

from seamkit.delegation import Chain, Grant, Link, Trust, run_value, sign
from seamkit.errors import DelegationRefused

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
                      "damaged_parts", "repair_estimate", "police_report"}),
    hops=2)

ROOT_LIFETIME = 15 * 60.0  # seconds; every link below expires no later


@dataclass
class MeridianIssuer:
    """The token exchange at the root of every Meridian chain."""

    sessions: Mapping[str, tuple[str, frozenset[str]]] = field(
        default_factory=lambda: SESSIONS)
    key: Ed25519PrivateKey = field(default_factory=lambda: KEYS[ISSUER])
    now: Callable[[], float] = time.time
    exchanges: int = 0
    _roots: dict[str, Chain] = field(default_factory=dict)

    def exchange(self, session: str) -> Chain:
        """An adjuster's session in, a root link addressed to Claims out."""
        if session not in self.sessions:
            raise DelegationRefused("claims", "unknown or ended session")
        subject, claims = self.sessions[session]
        self.exchanges += 1
        root = Link(ISSUER, "claims", subject, claims, ROOT.skills, ROOT.fields,
                    ROOT.hops, self.now() + ROOT_LIFETIME)
        return (sign(root, self.key),)

    def run_chain(self) -> Chain:
        """The root for the current Claims run, exchanged once per session lifetime."""
        session = run_value("session")
        if not session:
            raise DelegationRefused("claims", "this run carries no adjuster session")
        root = self._roots.get(session)
        if root is None or root[0].expires - 60 <= self.now():
            root = self._roots[session] = self.exchange(session)
        return root


# One issuer per Claims process, shared by every partner connection it opens.
CLAIMS_ISSUER = MeridianIssuer()
