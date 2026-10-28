"""The federation gateway: one owner's seams, built once, from one configuration.

Every control the book placed at a seam was switched on in its own chapter, by its own
builder: a trust policy at connect (Chapter 5), a chain minted per call (Chapter 6), a
resume gate (Chapters 7 and 10), the ingress filter (Chapter 8), the ledger (Chapter 9),
a retry policy (Chapter 12), a breaker (Chapter 14), receipts (Chapter 17), a budget
(Chapter 18) and evidence terms (Chapter 19). The gateway reads one owner's directory
and builds every seam with every control its expectations declare, through the same
constructors, so each control runs in the order its chapter's tests pin.

It adds no rule a partner sees. It adds one rule of its own: `check` refuses to start a
seam that is missing a control the owner's policy requires, unless the seam's entry
waives it with a reason a reviewer can read.

A gateway belongs to one owner and runs in that owner's process. It is never shared:
an owner's ledger, resume checks and compensations need the owner's checkpoint, which a
proxy in front of several owners cannot read, and a gateway several owners route
through is an owner of its own on every seam.

Jobs that run on a schedule (canaries, reconciliation, the backstop poll, the daily
witness) read the same expectations files; the gateway lists them and does not run
them.
"""

import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from a2a.types import AgentCard
from a2a.server.agent_execution import AgentExecutor
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from langchain_core.tools import StructuredTool
from starlette.routing import BaseRoute
from starlette.types import ASGIApp

from seamkit.breaker import SeamBreaker
from seamkit.contract import (
    NO_CROSSING,
    CrossingPolicy,
    Expectations,
    load_expectations,
    seam_spec,
)
from seamkit.costs import CostLedger, Costs, Meter, RateCard
from seamkit.delegation import (
    Chain,
    DelegationAuth,
    Grant,
    Guard,
    RevocationCheck,
    Trust,
)
from seamkit.inventory import PROPERTIES, SeamEntry, SeamInventory
from seamkit.ownership import OwnershipMap, load_map
from seamkit.receipt import ReceiptLog, Receipts
from seamkit.remote import (
    CrossingRecord,
    FilteredNode,
    PartnerConnection,
    RemoteAgentNode,
    SeamSpec,
    connect,
)
from seamkit.server import Countersigner, a2a_app
from seamkit.tools import remote_skill_tool
from seamkit.trust import PartnerTrust

# Every control the gateway can put on an outbound seam: its property and its chapter.
CONTROLS: dict[str, tuple[str, str]] = {
    "contract": ("contract", "Ch 4, 15"), "trust": ("authority", "Ch 5"),
    "chain": ("authority", "Ch 6"), "revocation": ("authority", "Ch 7"),
    "ingress": ("authority", "Ch 8"), "ledger": ("state", "Ch 9"),
    "resume": ("state", "Ch 10"), "retry": ("state", "Ch 12"),
    "breaker": ("failure", "Ch 14"), "receipts": ("accountability", "Ch 17"),
    "budget": ("accountability", "Ch 18"), "evidence": ("accountability", "Ch 19"),
}


@dataclass(frozen=True)
class Point:
    """One control the gateway runs on a seam: an enforcement point."""

    seam: str
    control: str
    property: str
    chapter: str
    detail: str


class UnguardedSeam(Exception):
    """A seam that would start without a control the owner's policy requires."""

    def __init__(self, seam: str, missing: Sequence[str]) -> None:
        super().__init__(f"{seam} would start without {', '.join(missing)}")
        self.seam, self.missing = seam, tuple(missing)


@dataclass(frozen=True)
class SeamConfig:
    """One seam the owner sends on: its expectations, its partner, what its link
    grants, and the controls it is let off, each with its reason."""

    expects: Expectations
    url: str
    grant: Grant
    shape: str = "node"  # "node": the graph decides the call; "tool": a model does
    waivers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Serves:
    """The owner's inbound half: what its server runs before its executor sees a call."""

    audience: str  # the name a chain's last link must be minted for
    countersigns: bool = False
    meters: bool = False


@dataclass(frozen=True)
class GatewayConfig:
    owner: str
    workload: str  # the identity its client certificate names
    requires: tuple[str, ...]  # the owner's policy: what every outbound seam carries
    seams: Mapping[str, SeamConfig]  # in the order the owner's graph sends on them
    ownership: OwnershipMap | None = None
    serves: Serves | None = None
    jobs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path, name: str = "gateway.toml") -> "GatewayConfig":
        """Read one owner's gateway once: its file, and the expectations it names."""
        with (directory / name).open("rb") as fh:
            raw = tomllib.load(fh)
        seams = {}
        for seam, entry in raw.get("seams", {}).items():
            expects = load_expectations(directory / entry["expects"])
            grant = entry["grant"]
            seams[seam] = SeamConfig(
                expects, str(entry["url"]),
                Grant(frozenset(grant["skills"]),
                      frozenset(expects.sends) | frozenset(grant.get("also", [])),
                      int(grant["hops"])),
                str(entry.get("shape", "node")),
                {k: str(v) for k, v in entry.get("waive", {}).items()})
        serves = raw.get("serves")
        mapped = raw.get("ownership")
        return cls(str(raw["owner"]), str(raw["workload"]), tuple(raw["requires"]),
                   seams, load_map(directory / mapped) if mapped else None,
                   Serves(str(serves["audience"]), bool(serves.get("countersigns")),
                          bool(serves.get("meters"))) if serves else None,
                   {seam: _jobs(c.expects) for seam, c in seams.items()})


def _jobs(e: Expectations) -> tuple[str, ...]:
    """The scheduled jobs a seam's expectations ask for; the gateway does not run them."""
    found = ["reconcile"] if e.retry.attempts > 1 or e.lookup else []
    if e.canary:
        found.append(f"canary every {e.canary.every_seconds:g} s")
    if e.backstop_seconds:
        found.append(f"backstop poll every {e.backstop_seconds} s")
    if e.evidence and e.evidence.witness == "daily":
        found.append("witness daily")
    return tuple(found)


@dataclass
class Runtime:
    """What the owner's process hands its gateway at start: keys, stores and hooks."""

    key: Ed25519PrivateKey  # signs the owner's links and its half of every receipt
    kid: str
    # Outbound: an owner that serves and sends on nothing leaves these as they are
    root: Callable[[], Chain] = field(default=lambda: ())  # links are minted under it
    authority: Callable[[], str] = field(default=lambda: "")  # a ledger entry's chain id
    trust: Mapping[str, PartnerTrust] = field(default_factory=dict)  # by owner
    policy: CrossingPolicy = NO_CROSSING  # the data-class policy
    pinned: Mapping[str, Ed25519PublicKey] = field(default_factory=dict)  # partner keys
    secret: bytes = b""  # what receipt nonces derive from; only this owner holds it
    rates: RateCard | None = None
    receipts: ReceiptLog = field(default_factory=ReceiptLog)
    costs: CostLedger = field(default_factory=CostLedger)
    log: Callable[[CrossingRecord], None] = field(default=lambda record: None)
    # The owner's resume gate (Chapters 7 and 10), a FilteredNode subclass; None sends
    # with FilteredNode alone, which re-checks nothing before reusing a recorded task.
    gate: Callable[..., FilteredNode] | None = None
    # The onward act point (Chapter 7): re-checks the run's chain before a link is
    # minted. A caller whose root it mints itself needs none.
    rechecks: bool = False
    # Inbound: the tenants' issuers this owner's guard verifies, and its list check
    trusts: Mapping[str, Trust] = field(default_factory=dict)
    revoked: RevocationCheck | None = None
    now: Callable[[], float] = field(default=lambda: 0.0)


@dataclass
class Gateway:
    """One owner's federation gateway. `connect` first, then build seams from it."""

    config: GatewayConfig
    runtime: Runtime
    connections: dict[str, PartnerConnection] = field(default_factory=dict)
    built: dict[str, RemoteAgentNode] = field(default_factory=dict)

    async def connect(self, http: Mapping[str, httpx.AsyncClient]) -> None:
        """Fetch each card, decide its trust, check its terms, and set its chain."""
        rt = self.runtime
        for seam, sc in self.config.seams.items():
            owner = sc.expects.owner
            conn = await connect(sc.url, http[seam], sc.expects, rt.policy,
                                 trust=rt.trust[owner])
            assert conn.trust is not None
            http[seam].auth = DelegationAuth(self.config.workload, rt.key, owner.lower(),
                                             sc.grant, rt.root, conn.card,
                                             conn.trust.covers)
            self.connections[seam] = conn

    def outbound(self, seam: str) -> RemoteAgentNode:
        """The seam's node, every control its expectations declare, in the node's order.

        No subclass and no new step: the node every chapter extended, with each
        chapter's attribute set from the seam's file, so each earlier test still holds.
        """
        sc, rt, conn = self.config.seams[seam], self.runtime, self.connections[seam]
        spec, e = seam_spec(sc.expects), sc.expects
        build = rt.gate or FilteredNode
        node = build(spec, conn, rt.log, rt.authority, policy=rt.policy)
        if e.breaker:
            node.breaker = SeamBreaker(seam, e.owner, e.skill, e.breaker)
        node.receipts = self._receipts(e)
        if e.budget:
            assert rt.rates is not None, "a budgeted seam needs the owner's rate card"
            held = conn.terms[0].skills.get(e.skill) if conn.terms else None
            node.costs = Costs(e.budget, rt.costs, rt.rates, rt.pinned,
                               reported=bool(held and held.usage == "reported"))
        self.built[seam] = node
        return node

    def tool(self, seam: str) -> StructuredTool:
        """A seam a model decides to call: Chapter 3's remote skill tool, with the
        controls a tool call can carry. It has no thread to link, so no ledger entry."""
        sc, rt, conn = self.config.seams[seam], self.runtime, self.connections[seam]
        e = sc.expects
        breaker = SeamBreaker(seam, e.owner, e.skill, e.breaker) if e.breaker else None
        return remote_skill_tool(conn, e.skill, e.owner, rt.log,
                                 rules=seam_spec(e).rules, breaker=breaker,
                                 receipts=self._receipts(e), authority=rt.authority)

    def _receipts(self, e: Expectations) -> Receipts:
        rt = self.runtime
        return Receipts(rt.key, rt.kid, rt.secret, rt.pinned, rt.receipts,
                        e.receipts == "countersigned")

    def inbound(self, card: AgentCard, executor: AgentExecutor,
                meter: Meter | None = None, extra: Sequence[BaseRoute] = (),
                **kept: Any) -> ASGIApp:
        """The owner's server: the chain and the list checked by the guard, the key
        index, the countersigner and the meter around the owner's own executor.

        `kept` passes what outlives one release's app (a task store, a key index, the
        guard's task bindings) to the constructor that keeps it.
        """
        serves, rt = self.config.serves, self.runtime
        assert serves is not None, f"{self.config.owner} serves no skill"
        guarded = {k: kept.pop(k) for k in ("tasks", "holders") if k in kept}
        signs = (Countersigner(rt.key, rt.kid, rt.now) if serves.countersigns
                 else None)
        inner = a2a_app(card, executor, extra, now=rt.now, countersign=signs,
                        meter=meter if serves.meters else None, **kept)
        return Guard(inner, serves.audience, rt.trusts, now=rt.now,
                     revoked=rt.revoked, **guarded)

    def points(self, seam: str) -> list[Point]:
        """What is on: each control on the seam, its property, and its chapter."""
        sc, e = self.config.seams[seam], self.config.seams[seam].expects
        conn = self.connections.get(seam)
        on: dict[str, str] = {}
        if conn and conn.terms:
            held = conn.terms[0].revision
            on["contract"] = (f"terms {held} breached ({len(conn.breaches)})"
                              if conn.breaches else f"terms {held} checked")
            if e.protocols:
                on["contract"] += f"; protocol {conn.protocol or 'none spoken'}"
        if conn and conn.trust:
            on["trust"] = (f"card trusted by {conn.trust.pinned_by}" if conn.trust.trusted
                           else f"card refused: {conn.trust.refused}")
        onward = sorted(sc.grant.skills - {e.skill})
        on["chain"] = (f"link for {e.skill}, {sc.grant.hops} hop(s) below"
                       + (f", may pass on {', '.join(onward)}" if onward else ""))
        if self.runtime.gate is not None or self.runtime.rechecks:
            seconds = conn.terms[0].revocation_seconds if conn and conn.terms else None
            on["revocation"] = ("re-checked at every act point"
                                + (f"; revoked within {seconds} s" if seconds else ""))
        rules = seam_spec(e).rules or {}
        text = sum(1 for rule in rules.values() if rule.kind == "text")
        on["ingress"] = f"{len(rules)} field(s) filtered, {text} quarantined"
        if sc.shape == "node":
            on["ledger"] = "intent and key recorded before the send"
            if e.roles:
                on["resume"] = (f"asked only of {', '.join(e.roles)}, "
                                f"waits {e.wait_seconds} s")
            elif e.answers:
                on["resume"] = f"answers {', '.join(e.answers)}"
        if e.retry.attempts > 1:
            on["retry"] = (f"{e.retry.attempts} attempts under the honoured key, "
                           f"{e.retry.deadline_seconds:g} s deadline")
        elif e.lookup:
            on["retry"] = "looked up before a send: the partner honours no key"
        if e.breaker:
            on["breaker"] = (f"{e.breaker.failures} in {e.breaker.window_seconds:g} s "
                             f"opens it; {e.breaker.mode}")
        on["receipts"] = ("countersigned" if e.receipts == "countersigned"
                          else "ours, with the partner's half when it sends one")
        if e.budget and sc.shape == "node":
            on["budget"] = f"ceiling {e.budget.per_task:g} per task"
        if e.evidence:
            on["evidence"] = (f"kept {e.evidence.retain_days} days, produced within "
                              f"{e.evidence.produce_within_days}, witnessed "
                              f"{e.evidence.witness}")
        return [Point(seam, name, *CONTROLS[name], detail)
                for name, detail in on.items()]

    def check(self) -> None:
        """Refuse to start any seam that lacks a required control it has not waived."""
        for seam, sc in self.config.seams.items():
            on = {p.control for p in self.points(seam)}
            missing = [c for c in self.config.requires
                       if c not in on and c not in sc.waivers]
            if missing:
                raise UnguardedSeam(seam, missing)

    def waived(self, seam: str) -> dict[str, str]:
        return dict(self.config.seams[seam].waivers)

    def inventory(self, caller: str) -> SeamInventory:
        """Chapter 1's inventory for this owner's seams, every column from its points."""
        entries = []
        for seam, sc in self.config.seams.items():
            entry = SeamEntry(caller, seam, self.config.owner, sc.expects.owner)
            points = self.points(seam)
            for prop in PROPERTIES:
                cells = [f"{p.control}: {p.detail}" for p in points
                         if p.property == prop]
                cells += [f"{c} waived: {why}" for c, why in sc.waivers.items()
                          if CONTROLS[c][0] == prop]
                if cells:
                    entry.answers[prop] = "; ".join(cells)
            held = self.config.ownership and self.config.ownership.seams.get(seam)
            if held:
                entry.paged = f"{held.caller_oncall}, {held.callee_contact}"
            entries.append(entry)
        return SeamInventory(entries)


__all__ = ["CONTROLS", "Gateway", "GatewayConfig", "Point", "Runtime", "SeamConfig",
           "Serves", "UnguardedSeam"]
