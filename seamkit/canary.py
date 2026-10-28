"""Behavioural canaries: golden tasks sent to a live partner skill on a schedule.

Every check before Chapter 15 reads what a partner declares: its card, its terms, the
shape of an answer. None of them notices a partner whose skill starts answering
differently under an unchanged card, because nothing it declares has changed. A canary
reads what the partner does to a known input. Each golden task has an answer a person
accepted (the baseline) and a band per output field; a run whose answers leave their
bands is drift, and the seam's drift action runs.

The caller assigns the behaviour version, because the partner publishes none: it is a
short hash of the accepted baseline. A person accepts each new one. A canary's answers
are judged and kept with a provenance tag that names the golden, and they never reach
claim state, never count in a track record, and never touch a breaker.

A canary is a real task at a real partner: it is billed, it needs authority, and on a
skill that commits it would be a real transaction. Chapter 4's `check` refuses a
`[canary]` table on such a skill.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from seamkit.delegation import run_values
from seamkit.errors import PartnerRejected, SeamError

if TYPE_CHECKING:  # seamkit.contract reads the [canary] table, and both import it
    from seamkit.reconcile import Finding
    from seamkit.remote import RemoteAgentNode

DRIFT_ACTIONS = frozenset({"unproven", "hold"})

canary_logger = logging.getLogger("seamkit.canary")


def log_finding(finding: "Finding") -> None:
    """Chapter 12's finding log, which a drift finding joins."""
    from seamkit.reconcile import log_finding as log

    log(finding)


@dataclass(frozen=True)
class Band:
    """How far an answer may move from the accepted one and be the same behaviour."""

    absolute: float = 0.0  # in the field's own unit, such as points on 0-100
    relative: float = 0.0  # a fraction of the accepted value

    def holds(self, accepted: float, got: float) -> bool:
        return abs(got - accepted) <= max(self.absolute, self.relative * abs(accepted))


@dataclass(frozen=True)
class CanaryPolicy:
    """A seam's `[canary]` table: how often, how much movement, and what drift does."""

    every_seconds: float
    band: Band
    confirm: int = 1  # consecutive runs out of band before the drift action runs
    on_drift: str = "unproven"  # "unproven": back under the ceiling; "hold": no sends


@dataclass(frozen=True)
class Golden:
    """One golden task: a synthetic claim under a reserved id, and the fields judged."""

    id: str  # a reserved claim id (CLM-C...), never a customer's
    request: Mapping[str, Any]  # what the seam sends, and nothing else
    judged: tuple[str, ...]  # the state fields the band applies to


@dataclass(frozen=True)
class Baseline:
    """The answers a person accepted as the skill's behaviour, and who accepted them."""

    answers: Mapping[str, Mapping[str, float]]  # golden id -> field -> accepted value
    accepted_by: str
    accepted_at: float

    @property
    def version(self) -> str:
        """The behaviour version: the caller's name for this behaviour."""
        return behaviour_version(self.answers, self.accepted_by, self.accepted_at)


def behaviour_version(answers: Mapping[str, Mapping[str, float]], by: str = "",
                      at: float = 0.0) -> str:
    raw = json.dumps({"answers": answers, "by": by, "at": at}, sort_keys=True)
    return "bv:" + hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Verdict:
    """One run of a suite: in band, drift, refused, or no answer."""

    kind: str
    goldens: tuple[str, ...] = ()  # the goldens behind any verdict but "in band"
    answers: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    detail: str = ""


Answer = Mapping[str, Any] | SeamError


def judge(goldens: Sequence[Golden], answers: Mapping[str, Answer],
          baseline: Baseline, band: Band) -> Verdict:
    """One run's answers against the accepted ones, field by field, inside the band."""
    failed = {gid: a for gid, a in answers.items() if isinstance(a, SeamError)}
    if failed:  # a refusal is an answer about the contract; silence is no answer
        refused = all(isinstance(e, PartnerRejected) for e in failed.values())
        detail = "; ".join(f"{gid}: {e.detail}" for gid, e in failed.items())
        return Verdict("refused" if refused else "no answer", tuple(failed),
                       detail=detail)
    got = {gid: a for gid, a in answers.items() if not isinstance(a, SeamError)}
    seen = {g.id: {f: float(got[g.id][f]) for f in g.judged} for g in goldens}
    out = tuple(g.id for g in goldens if not all(
        band.holds(baseline.answers[g.id][f], seen[g.id][f]) for f in g.judged))
    return Verdict("drift" if out else "in band", out, seen)


@dataclass
class CanarySuite:
    """One seam's canaries, run on the clock and judged against the accepted baseline.

    `node` is the seam's own node: a canary crosses the same seam, under the same
    ingress filter, as a real task. `run` names the session the canary chain is
    exchanged from, the seam's on-call, whose caseload is the reserved golden ids.
    """

    seam: str
    policy: CanaryPolicy
    goldens: tuple[Golden, ...]
    baseline: Baseline
    node: "RemoteAgentNode"
    run: Mapping[str, Any]  # the run config values the canary chain is minted under
    file: Callable[["Finding"], None] = log_finding
    last: float | None = None  # when the suite last ran, on the shared clock
    history: list[tuple[float, Verdict]] = field(default_factory=list)
    accepted: list[Baseline] = field(default_factory=list)  # every one, oldest first
    seen: dict[str, Any] = field(default_factory=dict)  # golden id -> answer, tagged
    streak: int = 0  # consecutive runs out of band

    def __post_init__(self) -> None:
        self.accepted.append(self.baseline)
        # The first baseline describes the behaviour the skill's record was earned
        # under, so the record is kept; every later version starts a new one.
        self.node.partner.rekey(self.skill, self.baseline.version, keep=True)

    @property
    def skill(self) -> str:
        return self.node.seam.skill

    def due(self, now: float) -> bool:
        return self.last is None or now - self.last >= self.policy.every_seconds

    async def ask(self, golden: Golden, now: float) -> Answer:
        """One golden task across the seam, under the canary chain. Never in state."""
        try:
            with run_values(**self.run):
                answer, tags = await self.node.probe(
                    golden.request, f"canary-{golden.id}-{now:.0f}")
        except SeamError as exc:
            return exc
        self.seen[golden.id] = {"answer": answer, "provenance": {
            name: {**tag, "canary": golden.id} for name, tag in tags.items()}}
        return answer

    async def go(self, now: float) -> Verdict:
        """Send every golden, judge the run, and act on drift."""
        self.last = now
        answers = {g.id: await self.ask(g, now) for g in self.goldens}
        verdict = judge(self.goldens, answers, self.baseline, self.policy.band)
        self.history.append((now, verdict))
        canary_logger.info("canary run: %s %s %s", self.seam, verdict.kind,
                           ", ".join(verdict.goldens))
        if verdict.kind == "in band":
            self.streak = 0
            return verdict
        if verdict.kind == "drift":
            self.streak += 1
            if self.streak < self.policy.confirm:
                return verdict
            self.drifted(verdict)
        self.file(self.finding(verdict))
        return verdict

    def drifted(self, verdict: Verdict) -> None:
        """The drift action: the record re-keyed to the unaccepted behaviour, and for
        a hold, no send on the skill until a person accepts or releases it."""
        partner = self.node.partner
        partner.rekey(self.skill, "unaccepted:" + behaviour_version(verdict.answers))
        if self.policy.on_drift == "hold":
            partner.held[self.skill] = (
                f"{self.skill} is held: canary drift on {', '.join(verdict.goldens)} "
                f"against {self.baseline.version}")

    def finding(self, verdict: Verdict) -> "Finding":
        from seamkit.reconcile import Finding

        return Finding(self.seam, f"canary {verdict.kind}", "", "",
                       partner_release=self.node.partner.release,
                       detail=f"{self.skill}: {', '.join(verdict.goldens)} against "
                              f"{self.baseline.version}"
                              + (f"; {verdict.detail}" if verdict.detail else ""))

    def accept(self, by: str, now: float) -> Baseline:
        """A person accepts the last run's answers as the skill's behaviour.

        The old baseline is kept. The skill restarts unproven under the new version,
        and a hold is lifted: accepting a change costs the skill its track record.
        """
        _, last = self.history[-1]
        if last.kind not in ("drift", "in band"):
            raise ValueError(f"the last run was {last.kind}: nothing to accept")
        self.baseline = Baseline(dict(last.answers), by, now)
        self.accepted.append(self.baseline)
        self.node.partner.rekey(self.skill, self.baseline.version)
        self.node.partner.held.pop(self.skill, None)
        self.streak = 0
        return self.baseline


async def tick(suites: Sequence[CanarySuite], now: float) -> dict[str, Verdict]:
    """Run every suite that is due: what a scheduler calls on the shared clock."""
    return {s.seam: await s.go(now) for s in suites if s.due(now)}


__all__ = ["DRIFT_ACTIONS", "Band", "Baseline", "CanaryPolicy", "CanarySuite",
           "Golden", "Verdict", "behaviour_version", "judge", "tick"]
