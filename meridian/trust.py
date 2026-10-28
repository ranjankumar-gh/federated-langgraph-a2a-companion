"""Meridian's card trust policy: what Claims pinned, out of band, about each partner.

The Federation Test's trust-domain answer picks each partner's pin mode. Northgate is
across a company line, signs its card, and ships on its own schedule, so it is pinned
by key: any card Northgate signs is Northgate's, and a release needs no approval here.
Risk and Finance are inside Meridian's trust domain and release through Meridian's own
pipeline, which publishes each card's digest with the release, so they are pinned by
digest: a release is trusted once a pull request adds its digest to this file.

Every value here is a Meridian decision. None of it is read from a card.
"""

from northgate.server import NORTHGATE_KID, NORTHGATE_PUBLIC_JWK, NORTHGATE_URL
from meridian import finance_server, risk_server
from seamkit.trust import Gradient, PartnerTrust, origin

NORTHGATE = PartnerTrust(
    "Northgate", origins=frozenset({origin(NORTHGATE_URL)}),
    skills=frozenset({"estimate_repair", "estimate_history"}),
    keys={NORTHGATE_KID: NORTHGATE_PUBLIC_JWK},  # handed over at onboarding
    refetch_seconds=3600,
    # Northgate's estimate decides what Finance pays, so it is capped until proven.
    gradients={"estimate_repair": Gradient("estimate", unproven_ceiling=5_000.0)})

RISK = PartnerTrust(
    "Risk", origins=frozenset({origin(risk_server.RISK_URL)}),
    skills=frozenset({"screen_fraud"}), digests=risk_server.PUBLISHED_DIGESTS,
    refetch_seconds=300)

FINANCE = PartnerTrust(
    "Finance", origins=frozenset({origin(finance_server.FINANCE_URL)}),
    skills=frozenset({"pay_claim"}), digests=finance_server.PUBLISHED_DIGESTS,
    refetch_seconds=300)

# Keyed by the owner Claims' expectations name, never by the provider a card names.
MERIDIAN_TRUST: dict[str, PartnerTrust] = {
    p.owner: p for p in (NORTHGATE, RISK, FINANCE)}
