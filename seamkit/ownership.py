"""The ownership map: one owner's seams, and who answers for each signal they raise.

Chapter 1's seam inventory has a `paged` column: one name per seam. A seam raises more
than one kind of signal, and they do not all belong to the same person. A breaker
transition and a partner stop (Chapter 14), a drift finding (Chapter 15), a
reconciliation finding (Chapter 12), a residue or a refused compensation (Chapter 13),
a push verdict (Chapter 11), a receipt mismatch and an SLO burn (Chapter 17): each kind
goes to the caller's on-call, the partner's contact, a named queue, or more than one.

Each owner keeps a map of its own seams, the ones it calls and the ones it serves. No
owner's map names a partner's partner: Claims' map sends the screening seam's signals
to Risk, and Risk's own map sends its estimate-history seam's signals to Northgate. A
seam missing from the map, or a kind of signal with nowhere to go, fails at startup,
because a signal with nobody to route to is dropped without anyone deciding to drop it.
"""

import logging
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from seamkit.breaker import SeamBreaker
from seamkit.reconcile import Finding
from seamkit.slo import Budget

KINDS = ("breaker", "stop", "drift", "reconcile", "residue", "push", "receipt", "slo")
SIDES = ("caller", "callee")  # any other route is a named queue

# A finding's verdict, as Chapters 12, 13 and 15 write it, and the kind it signals.
FINDING_KINDS = {"residue": "residue", "refused": "residue",
                 "uncompensated": "residue", "lost": "residue",
                 "canary drift": "drift"}

page_logger = logging.getLogger("seamkit.page")


class UnmappedSeam(Exception):
    """A seam, or a kind of signal on it, that the owner's map sends nowhere."""


@dataclass(frozen=True)
class Signal:
    """Something a seam raised that a person has to hear about."""

    seam: str
    kind: str  # one of KINDS
    detail: str
    at: float = 0.0


@dataclass(frozen=True)
class Page:
    """One signal, addressed: who hears it, and on which side of the seam."""

    to: str
    side: str  # "caller", "callee", or "queue"
    signal: Signal


@dataclass(frozen=True)
class SeamOwners:
    """One entry on the map: the seam, its two owners, and where each signal goes."""

    seam: str
    caller: str
    callee: str
    caller_oncall: str
    callee_contact: str
    routes: Mapping[str, tuple[str, ...]]  # kind -> "caller", "callee", or a queue


@dataclass(frozen=True)
class OwnershipMap:
    owner: str
    seams: Mapping[str, SeamOwners]

    def route(self, signal: Signal) -> list[Page]:
        """Every page one signal becomes, on this owner's map."""
        entry = self.seams.get(signal.seam)
        if entry is None or signal.kind not in entry.routes:
            raise UnmappedSeam(f"the {self.owner} map sends {signal.kind} on "
                               f"{signal.seam} nowhere")
        names = {"caller": entry.caller_oncall, "callee": entry.callee_contact}
        return [Page(names.get(to, to), to if to in SIDES else "queue", signal)
                for to in entry.routes[signal.kind]]

    def require(self, seams: Iterable[str]) -> None:
        """At startup: every seam this owner runs is on the map, every kind routed."""
        missing = [s for s in seams if s not in self.seams]
        unrouted = [f"{s.seam} {kind}" for s in self.seams.values()
                    for kind in KINDS if not s.routes.get(kind)]
        if missing or unrouted:
            raise UnmappedSeam("; ".join(
                [f"{s} is not on the {self.owner} map" for s in missing]
                + [f"{u} goes nowhere" for u in unrouted]))


def load_map(path: Path) -> OwnershipMap:
    """One owner's map, from its TOML file. The owner keeps it beside its code."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    defaults = {k: tuple(v) for k, v in raw.get("routes", {}).items()}
    seams = {}
    for name, entry in raw["seams"].items():
        routes = {**defaults, **{k: tuple(v) for k, v in entry.get("routes", {}).items()}}
        caller = str(entry.get("caller", raw["owner"]))  # a served seam names it
        seams[name] = SeamOwners(name, caller, str(entry["callee"]),
                                 str(entry.get("caller_oncall", raw["oncall"])),
                                 str(entry["callee_contact"]), routes)
    return OwnershipMap(str(raw["owner"]), seams)


def log_page(page: Page) -> None:
    page_logger.warning("page", extra={"page": {"to": page.to, "side": page.side,
                                                "seam": page.signal.seam,
                                                "kind": page.signal.kind,
                                                "detail": page.signal.detail}})


# The signals the earlier chapters already raise, as the map reads them.

def from_breaker(breaker: SeamBreaker, since: float = float("-inf")) -> list[Signal]:
    """Each move of a breaker since `since`: opened, half-open, closed."""
    return [Signal(breaker.seam, "breaker", f"{breaker.owner} {breaker.skill}: {state}",
                   when) for when, state in breaker.moves if when >= since]


def from_finding(finding: Finding, at: float = 0.0) -> Signal:
    """A finding is a reconciliation signal unless its verdict says otherwise."""
    kind = FINDING_KINDS.get(finding.verdict, "reconcile")
    return Signal(finding.seam, kind, f"{finding.verdict}: {finding.detail}", at)


def from_budget(budget: Budget, alert: float, at: float = 0.0) -> list[Signal]:
    """An SLO burn past the alert share is a signal; anything under it is not."""
    return [Signal(budget.seam, "slo", str(budget), at)] if budget.spent >= alert else []


__all__ = ["FINDING_KINDS", "KINDS", "OwnershipMap", "Page", "SeamOwners", "Signal",
           "UnmappedSeam", "from_breaker", "from_budget", "from_finding", "load_map",
           "log_page"]
