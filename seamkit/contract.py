"""The seam contract: the partner's published terms and the caller's expectations.

The two halves have two owners. The partner publishes its terms on its agent card, as a
data-only extension, one dated revision at a time; the caller keeps its expectations in
its own repository. `check` holds one against the other before a task is sent, and
reads declarations only: a partner whose behaviour changes under unchanged terms passes.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dc_field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from a2a.types import AgentCard, AgentExtension
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, create_model

from seamkit.breaker import MODES, BreakerPolicy
from seamkit.canary import DRIFT_ACTIONS, Band, CanaryPolicy
from seamkit.costs import CostBudget, budget_of
from seamkit.retry import NO_RETRY, RetryPolicy
from seamkit.slo import SeamSLO, slo_of

if TYPE_CHECKING:
    from seamkit.remote import SeamSpec

EXTENSION_URI = "urn:seamkit:seam-contract"

# The field types a contract may use, and the Python type a tool argument gets for each.
TYPES: dict[str, Any] = {"string": str, "number": float, "integer": int,
                         "boolean": bool, "string[]": list[str],
                         "object": dict[str, Any], "object[]": list[dict[str, Any]]}


@dataclass(frozen=True)
class Field:
    """One input or output of a skill: a row two owners can diff."""

    name: str
    type: str  # one of TYPES
    required: bool = True
    bounds: tuple[float, float] | None = None  # the scale a number is on
    unit: str = ""  # what a number counts, such as "claim currency"

    def satisfied_by(self, promised: "Field | None") -> bool:
        """True if the partner's output gives what this read depends on."""
        return (promised is not None and promised.required
                and promised.type == self.type
                and (self.bounds is None or promised.bounds == self.bounds)
                and (not self.unit or promised.unit == self.unit))

    def __str__(self) -> str:
        scale = f" on {self.bounds[0]:g}-{self.bounds[1]:g}" if self.bounds else ""
        unit = f" in {self.unit}" if self.unit else ""
        return f"{self.name} as a required {self.type}{scale}{unit}"


@dataclass(frozen=True)
class SkillTerms:
    """What one skill takes, returns, may pause to ask, and promises."""

    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]
    asks: tuple[str, ...] = ()  # what the skill may stop in input-required for
    p95_seconds: float | None = None  # 95 of 100 tasks answer inside this
    cost_per_task: float | None = None  # declared in Chapter 4, read from Chapter 18
    compensated_by: str = ""  # the skill that undoes this one, from Chapter 13
    # Chapter 10: the longest a task may wait on the partner's own people before the
    # partner reassigns or fails it; None states no wait horizon
    wait_seconds: int | None = None
    # Chapter 12: how long the partner answers a repeat of a send's key with the task
    # the first send opened; None honours no key, and a resend is a second task
    key_seconds: int | None = None
    # Chapter 13: how long after a task of this skill commits the partner promises to
    # honour `compensated_by`; None promises nothing, and an undo may be refused
    compensate_seconds: int | None = None
    # Chapter 18: "reported" if the partner signs a usage report into every task, and
    # whether it reads the ceiling a caller sends before it spends
    usage: str = ""
    honours_budget: bool = False
    # Chapter 19: the field names the skill reads (() states none), and where it
    # processes and stores what it receives (() states none)
    reads: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceTerms:
    """What one owner keeps for an auditor: how long, how fast it hands it over, and
    whether it witnesses the other owner's log head ("daily" or "none")."""

    retain_days: int
    produce_within_days: int
    witness: str


@dataclass(frozen=True)
class SeamTerms:
    """One dated revision of a partner's terms, as its card publishes them."""

    revision: str
    effective: date
    skills: Mapping[str, SkillTerms]
    auth: tuple[str, ...]  # what every request must carry
    skew_seconds: int  # clock skew the partner's verifier tolerates
    accepts: frozenset[str]  # the caller's data classes the partner agrees to hold
    # Chapter 7: the longest the partner may act under a revoked chain; None states none
    revocation_seconds: int | None = None
    # Chapter 17: "countersigned" if the partner signs its half of every seam receipt
    receipts: str = ""
    # Chapter 19: what the partner keeps for an auditor; None states nothing
    evidence: EvidenceTerms | None = None


@dataclass(frozen=True)
class Expectations:
    """The caller's half: what one seam sends, reads, and depends on."""

    seam: str
    owner: str
    skill: str
    sends: tuple[str, ...]
    reads: Mapping[str, Field]  # artifact key -> the field we depend on
    into: Mapping[str, str]  # artifact key -> our state field
    answers: tuple[str, ...] = ()  # partner questions our own people answer
    refuses: tuple[str, ...] = ()  # partner questions we stop the claim on
    auth: tuple[str, ...] = ()
    skew_seconds: int = 0
    slo_seconds: float = 10.0
    revocation_seconds: int | None = None  # the most we accept; None asks for nothing
    # Chapter 10: the Claims roles a question on this seam may be put to, and the
    # longest Claims waits on a partner's people; () keeps Chapter 3's rule
    roles: tuple[str, ...] = ()
    wait_seconds: int | None = None
    # Chapter 11: how Claims hears that a task moved (a row of seamkit.delivery.MODES),
    # and how often a backstop poll reads a task a push may have been lost for
    delivery: str = "stream"
    backstop_seconds: int | None = None
    # Chapter 12: how calls on this seam are retried, and whether a first send looks
    # its key up first, on a partner that honours none (a crash between send and record)
    retry: RetryPolicy = NO_RETRY
    lookup: bool = False
    # Chapter 14: the seam's breaker and its degraded mode; None keeps no breaker
    breaker: BreakerPolicy | None = None
    # Chapter 15: the protocol versions this seam speaks, newest first (() takes the
    # card as the SDK finds it); the days of notice a breaking revision must give;
    # and the seam's behavioural canaries, None for a seam nobody probes
    protocols: tuple[str, ...] = ()
    notice_days: int | None = None
    canary: CanaryPolicy | None = None
    # Chapter 8: artifact key -> how a read is closed (pattern, choices, items) or
    # capped (max_chars); seamkit.ingress turns these into the seam's ingress rules.
    ingress: Mapping[str, Mapping[str, Any]] = dc_field(default_factory=dict)
    # Chapter 17: "countersigned" needs the partner's half on every receipt, and "ours"
    # keeps one-sided receipts; the seam's SLO, None for a seam nobody measures
    receipts: str = "ours"
    slo: SeamSLO | None = None
    # Chapter 18: the seam's cost budget, the ceiling sent with every task and a line
    # per purpose; None for a seam nobody budgets
    budget: CostBudget | None = None
    # Chapter 19: the evidence an auditor will ask this seam's partner for; None asks
    # for nothing
    evidence: EvidenceTerms | None = None


@dataclass(frozen=True)
class CrossingPolicy:
    """The caller's data classes, and which of them may cross to each owner."""

    classes: Mapping[str, str]  # field -> data class
    allowed: Mapping[str, frozenset[str]]  # owner -> classes that may cross to it
    # Chapter 19: class -> the only regions its data may be processed in; a class
    # not named may go to any region a partner declares
    stays_in: Mapping[str, frozenset[str]] = dc_field(default_factory=dict)


# A caller that names no policy lets nothing cross: the check fails closed.
NO_CROSSING = CrossingPolicy({}, {})


@dataclass(frozen=True)
class Breach:
    """One disagreement between a revision of the terms and the caller's needs."""

    seam: str
    revision: str
    effective: date
    detail: str

    def __str__(self) -> str:
        return f"{self.seam}: {self.revision} (from {self.effective}): {self.detail}"


def check(terms: SeamTerms, expects: Expectations,
          policy: CrossingPolicy) -> list[Breach]:
    """Every way this revision of the partner's terms breaks what the seam needs."""
    skill = terms.skills.get(expects.skill)
    if skill is None:  # 1. the skill exists
        return [_breach(terms, expects, "is not offered")]
    takes = {f.name for f in skill.inputs}
    gives = {f.name: f for f in skill.outputs}
    crosses = policy.allowed.get(expects.owner, frozenset()) & terms.accepts
    found = [f"requires {f.name}, which this seam does not send"  # 2.
             for f in skill.inputs if f.required and f.name not in expects.sends]
    for name in expects.sends:  # 3. in a class that may cross, and accepted
        kind = policy.classes.get(name, "unclassified")
        if kind not in crosses:
            found.append(f"would receive {name}, which is {kind} data")
        elif name not in takes:
            found.append(f"does not accept {name}")
    found += [f"does not promise {want}"  # 4. what we read, on our scale
              for key, want in expects.reads.items()
              if not want.satisfied_by(gives.get(key))]
    found += [f"may ask for {q}, which this seam neither answers nor refuses"  # 5.
              for q in skill.asks if q not in (*expects.answers, *expects.refuses)]
    found += _auth(terms, expects) + _slo(skill, expects)  # 6. and 7.
    found += _wait(skill, expects)  # 8. Chapter 10
    found += _retry(skill, terms, expects)  # 9. Chapter 12
    found += _late(skill, expects)  # Chapter 14
    found += _canary(skill, expects)  # Chapter 15
    found += _receipts(terms, expects)  # Chapter 17
    found += _costs(skill, expects)  # Chapter 18
    found += _evidence(skill, terms, expects, policy)  # Chapter 19
    return [_breach(terms, expects, d) for d in found]


def held_to(revisions: Sequence[SeamTerms], expects: Expectations,
            policy: CrossingPolicy, seen: Mapping[str, date] | None = None,
            ) -> tuple[list[Breach], list[Breach]]:
    """Breaches in the revision in force, and in every revision announced after it.

    `seen` is when Claims first saw each revision on the card (Chapter 15): an
    announced breaking revision seen too late also breaks the seam's notice term.
    """
    if not revisions:
        return [Breach(expects.seam, "none", date.min,
                       f"{expects.owner} publishes no terms")], []
    current, *announced = revisions
    later: list[Breach] = []
    for terms in announced:
        found = check(terms, expects, policy)
        later += found + _notice(terms, found, expects, seen or {})
    return check(current, expects, policy), later


def _notice(terms: SeamTerms, found: list[Breach], expects: Expectations,
            seen: Mapping[str, date]) -> list[Breach]:
    """A breaking revision first seen too close to its date breaks the notice term."""
    first = seen.get(terms.revision)
    if not found or expects.notice_days is None or first is None:
        return []
    if (days := (terms.effective - first).days) >= expects.notice_days:
        return []
    return [_breach(terms, expects, f"breaks this seam {days} days after Claims first "
                    f"saw it, inside the {expects.notice_days} days of notice agreed")]


def _breach(terms: SeamTerms, expects: Expectations, detail: str) -> Breach:
    return Breach(expects.seam, terms.revision, terms.effective,
                  f"{expects.skill} {detail}")


def _auth(terms: SeamTerms, expects: Expectations) -> list[str]:
    found = [f"requires {scheme}, which this seam does not present"
             for scheme in terms.auth if scheme not in expects.auth]
    if terms.skew_seconds != expects.skew_seconds:
        found.append(f"tolerates {terms.skew_seconds} s of clock skew, "
                     f"not the {expects.skew_seconds} s agreed")
    if expects.revocation_seconds is not None:  # Chapter 7's term, compared like skew
        if terms.revocation_seconds is None:
            found.append("states no revocation horizon")
        elif terms.revocation_seconds > expects.revocation_seconds:
            found.append(f"may act {terms.revocation_seconds} s on a revoked chain, "
                         f"longer than the {expects.revocation_seconds} s agreed")
    return found


def _slo(skill: SkillTerms, expects: Expectations) -> list[str]:
    if skill.p95_seconds is None:
        return ["promises no response time"]
    if skill.p95_seconds > expects.slo_seconds:
        return [f"promises {skill.p95_seconds:g} s at p95, "
                f"longer than the {expects.slo_seconds:g} s this seam waits"]
    return []


def _wait(skill: SkillTerms, expects: Expectations) -> list[str]:
    """A skill that may pause for a person states how long; never longer than ours."""
    if not skill.asks or expects.wait_seconds is None:
        return []
    if skill.wait_seconds is None:
        return ["states no wait horizon"]
    if skill.wait_seconds > expects.wait_seconds:
        return [f"may wait {skill.wait_seconds} s on its own people, longer than "
                f"the {expects.wait_seconds} s this seam waits"]
    return []


def _retry(skill: SkillTerms, terms: SeamTerms, expects: Expectations) -> list[str]:
    """A seam that resends must finish resending while the partner keeps its key."""
    policy = expects.retry
    if policy.attempts <= 1 or skill.key_seconds is None:  # nothing is resent
        return []
    if policy.deadline_seconds > skill.key_seconds - terms.skew_seconds:
        return [f"keeps a key {skill.key_seconds} s, and this seam resends for "
                f"{policy.deadline_seconds:g} s plus {terms.skew_seconds} s of skew"]
    return []


def _late(skill: SkillTerms, expects: Expectations) -> list[str]:
    """A breaker that counts what the partner promised opens on a healthy partner."""
    late = expects.breaker.late_seconds if expects.breaker else None
    if late is None or skill.p95_seconds is None or late >= skill.p95_seconds:
        return []
    if skill.p95_seconds > expects.slo_seconds:  # Chapter 4's breach already holds it
        return []
    return [f"promises {skill.p95_seconds:g} s at p95, and this seam's breaker "
            f"counts an answer after {late:g} s against it"]


def _canary(skill: SkillTerms, expects: Expectations) -> list[str]:
    """A canary is a real task: on a skill that commits, a real transaction."""
    if expects.canary is None or not skill.compensated_by:
        return []
    return [f"commits at the partner (undone by {skill.compensated_by}), "
            "so it is never sent a canary"]


def _receipts(terms: SeamTerms, expects: Expectations) -> list[str]:
    """A receipt the terms never promised is a promise nobody made."""
    if expects.receipts != "countersigned" or terms.receipts == "countersigned":
        return []
    return ["countersigns no receipts, and this seam expects every answer "
            "countersigned"]


def _costs(skill: SkillTerms, expects: Expectations) -> list[str]:
    """A seam with a budget needs a price it can afford, or a report and a promise."""
    budget = expects.budget
    if budget is None:
        return []
    if skill.cost_per_task is not None:  # a fixed price: the terms are the price
        if skill.cost_per_task <= budget.per_task:
            return []
        return [f"costs {skill.cost_per_task:g} per task, more than the "
                f"{budget.per_task:g} this seam budgets"]
    if skill.usage != "reported":
        return ["states no price and reports no usage, and this seam budgets "
                f"{budget.per_task:g} per task"]
    if not skill.honours_budget:
        return ["reports usage but does not promise to honour the ceiling"]
    return []


def _evidence(skill: SkillTerms, terms: SeamTerms, expects: Expectations,
              policy: CrossingPolicy) -> list[str]:
    """What is sent must be declared, stay where it may, and be provable later."""
    found = [f"does not declare reading {name}" for name in expects.sends
             if skill.reads and name not in skill.reads]
    for name in expects.sends:  # no regions declared is a gap for the auditor
        kind = policy.classes.get(name, "unclassified")
        home = policy.stays_in.get(kind, frozenset(skill.regions))
        if not set(skill.regions) <= home:
            found.append(f"would process {name} ({kind}) in {sorted(skill.regions)}, "
                         f"not in {sorted(home)}")
    return found + _kept(terms.evidence, expects.evidence)


def _kept(have: EvidenceTerms | None, want: EvidenceTerms | None) -> list[str]:
    if want is None:
        return []
    if have is None:
        return ["states no evidence terms, and this seam expects them"]
    found = []
    if have.retain_days < want.retain_days:
        found.append(f"keeps evidence {have.retain_days} days, fewer than the "
                     f"{want.retain_days} expected")
    if have.produce_within_days > want.produce_within_days:
        found.append(f"produces it in {have.produce_within_days} days, slower than "
                     f"the {want.produce_within_days} expected")
    if want.witness == "daily" and have.witness != "daily":
        found.append("does not witness the log head, and this seam expects it daily")
    return found


def violations(answer: Mapping[str, Any], shapes: Mapping[str, Field]) -> list[str]:
    """What in a completed task's artifact is not the shape the seam depends on."""
    return [f"{key} is {answer[key]!r}, not {want}"
            for key, want in shapes.items()
            if key in answer and not _conforms(answer[key], want)]


def _conforms(value: Any, want: Field) -> bool:
    if want.type in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if want.type == "integer" and not float(value).is_integer():
            return False
        low, high = want.bounds or (float("-inf"), float("inf"))
        return low <= value <= high
    if want.type == "string[]":
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if want.type == "object[]":
        return isinstance(value, list) and all(isinstance(v, dict) for v in value)
    if want.type == "object":
        return isinstance(value, dict)
    return isinstance(value, TYPES[want.type])


# On the card: the terms as JSON, inside a data-only extension's params.

def publish(*revisions: SeamTerms) -> AgentExtension:
    """The extension a partner puts on its card: the revision in force first."""
    extension = AgentExtension(
        uri=EXTENSION_URI, required=False,
        description="Seam contract terms, current revision first")
    extension.params.update({"revisions": [_plain(r) for r in revisions]})
    return extension


def terms_of(card: AgentCard) -> tuple[SeamTerms, ...]:
    """The revisions a card publishes, current first; () if it publishes none."""
    for extension in card.capabilities.extensions:
        if extension.uri == EXTENSION_URI:
            params = MessageToDict(extension.params)
            return tuple(_terms(r) for r in params.get("revisions", []))
    return ()


def _plain(terms: SeamTerms) -> dict[str, Any]:
    def skill(s: SkillTerms) -> dict[str, Any]:
        out = {k: v for k, v in asdict(s).items()
               if v not in (None, "", ()) and v is not False}  # older cards unchanged
        for key in ("inputs", "outputs"):
            out[key] = [{k: v for k, v in f.items() if v not in (None, "")}
                        for f in out.get(key, [])]
        return out

    plain = {"revision": terms.revision, "effective": terms.effective.isoformat(),
             "skills": {sid: skill(s) for sid, s in terms.skills.items()},
             "auth": list(terms.auth), "skew_seconds": terms.skew_seconds,
             "accepts": sorted(terms.accepts)}
    if terms.revocation_seconds is not None:  # Chapter 7; absent from older revisions
        plain["revocation_seconds"] = terms.revocation_seconds
    if terms.receipts:  # Chapter 17; absent from older revisions
        plain["receipts"] = terms.receipts
    if terms.evidence:  # Chapter 19; absent from older revisions
        plain["evidence"] = asdict(terms.evidence)
    return plain


def _field(raw: Mapping[str, Any]) -> Field:
    bounds = raw.get("bounds")
    return Field(str(raw["name"]), str(raw["type"]), bool(raw.get("required", True)),
                 (float(bounds[0]), float(bounds[1])) if bounds else None,
                 str(raw.get("unit", "")))


def _skill(raw: Mapping[str, Any]) -> SkillTerms:
    p95, cost = raw.get("p95_seconds"), raw.get("cost_per_task")
    return SkillTerms(
        tuple(_field(f) for f in raw.get("inputs", [])),
        tuple(_field(f) for f in raw.get("outputs", [])),
        tuple(str(q) for q in raw.get("asks", [])),
        float(p95) if p95 is not None else None,
        float(cost) if cost is not None else None,
        str(raw.get("compensated_by", "")),
        int(raw["wait_seconds"]) if raw.get("wait_seconds") is not None else None,
        int(raw["key_seconds"]) if raw.get("key_seconds") is not None else None,
        int(raw["compensate_seconds"]) if raw.get("compensate_seconds") is not None
        else None,
        str(raw.get("usage", "")), bool(raw.get("honours_budget", False)),
        tuple(str(n) for n in raw.get("reads", [])),
        tuple(str(r) for r in raw.get("regions", [])))


def _terms(raw: Mapping[str, Any]) -> SeamTerms:
    revocation = raw.get("revocation_seconds")
    return SeamTerms(
        str(raw["revision"]), date.fromisoformat(str(raw["effective"])),
        {sid: _skill(s) for sid, s in raw.get("skills", {}).items()},
        tuple(str(a) for a in raw.get("auth", [])),
        int(raw.get("skew_seconds", 0)),  # protobuf Struct numbers arrive as floats
        frozenset(str(c) for c in raw.get("accepts", [])),
        int(revocation) if revocation is not None else None,
        str(raw.get("receipts", "")),
        _evidence_terms(raw["evidence"]) if "evidence" in raw else None)


# In the caller's repository: one TOML file per seam.

def load_expectations(path: Path) -> Expectations:
    """Read one seam's expectations file. The keys name the seam, not the filename."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    reads = raw.get("reads", {})
    return Expectations(
        seam=str(raw["seam"]), owner=str(raw["owner"]), skill=str(raw["skill"]),
        sends=tuple(raw.get("sends", [])),
        reads={key: _field({"name": key, **spec}) for key, spec in reads.items()},
        into={key: str(spec.get("into", key)) for key, spec in reads.items()},
        answers=tuple(raw.get("answers", [])), refuses=tuple(raw.get("refuses", [])),
        auth=tuple(raw.get("auth", [])), skew_seconds=int(raw.get("skew_seconds", 0)),
        slo_seconds=float(raw.get("slo_seconds", 10.0)),
        revocation_seconds=(int(raw["revocation_seconds"])
                            if "revocation_seconds" in raw else None),
        ingress={key: {k: v for k, v in spec.items() if k in INGRESS_KEYS}
                 for key, spec in reads.items()},
        roles=tuple(raw.get("roles", [])),
        wait_seconds=int(raw["wait_seconds"]) if "wait_seconds" in raw else None,
        delivery=str(raw.get("delivery", "stream")),
        backstop_seconds=(int(raw["backstop_seconds"])
                          if "backstop_seconds" in raw else None),
        retry=_policy(raw.get("retry", {})), lookup=bool(raw.get("lookup", False)),
        breaker=_breaker(raw["breaker"]) if "breaker" in raw else None,
        protocols=tuple(str(p) for p in raw.get("protocols", [])),
        notice_days=int(raw["notice_days"]) if "notice_days" in raw else None,
        canary=_canary_policy(raw["canary"]) if "canary" in raw else None,
        receipts=str(raw.get("receipts", "ours")),
        slo=slo_of(raw["slo"]) if "slo" in raw else None,
        budget=budget_of(raw["budget"]) if "budget" in raw else None,
        evidence=_evidence_terms(raw["evidence"]) if "evidence" in raw else None)


def _evidence_terms(raw: Mapping[str, Any]) -> EvidenceTerms:
    witness = str(raw.get("witness", "none"))
    if witness not in ("daily", "none"):
        raise ValueError(f"witness {witness!r} is not daily or none")
    return EvidenceTerms(int(raw["retain_days"]), int(raw["produce_within_days"]),
                         witness)


def revised(base: SeamTerms, path: Path) -> SeamTerms:
    """The partner's next revision: its file names what changes from `base`.

    The partner owns the file, one per revision. A skill table may change the fields
    the skill reads and the regions it processes them in.
    """
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    changes = raw.get("skills", {})
    skills = {sid: replace(
        skill, reads=tuple(changes.get(sid, {}).get("reads", skill.reads)),
        regions=tuple(changes.get(sid, {}).get("regions", skill.regions)))
        for sid, skill in base.skills.items()}
    return replace(base, revision=str(raw["revision"]), effective=raw["effective"],
                   skills=skills, evidence=_evidence_terms(raw["evidence"]))


def _policy(raw: Mapping[str, Any]) -> RetryPolicy:
    return RetryPolicy(int(raw.get("attempts", 1)), float(raw.get("base", 1.0)),
                       float(raw.get("cap", 60.0)),
                       float(raw.get("deadline_seconds", 0.0)))


def _breaker(raw: Mapping[str, Any]) -> BreakerPolicy:
    mode = str(raw["mode"])
    if mode not in MODES:
        raise ValueError(f"breaker mode {mode!r} is not one of {sorted(MODES)}")
    return BreakerPolicy(float(raw["late_seconds"]), int(raw["failures"]),
                         float(raw["window_seconds"]), float(raw["open_seconds"]),
                         str(raw.get("trial", "")), int(raw["max_in_flight"]), mode)


def _canary_policy(raw: Mapping[str, Any]) -> CanaryPolicy:
    action = str(raw.get("on_drift", "unproven"))
    if action not in DRIFT_ACTIONS:
        raise ValueError(f"drift action {action!r} is not in {sorted(DRIFT_ACTIONS)}")
    return CanaryPolicy(float(raw["every_seconds"]),
                        Band(float(raw.get("absolute", 0.0)),
                             float(raw.get("relative", 0.0))),
                        int(raw.get("confirm", 1)), action)


# The keys of a [reads.*] table the ingress filter reads (Chapter 8); the shape check
# never looks at them.
INGRESS_KEYS = frozenset({"max_chars", "pattern", "choices", "items"})


def seam_spec(expects: Expectations) -> "SeamSpec":
    """Chapter 3's SeamSpec, derived from the expectations instead of typed by hand."""
    # Imported here, not at the top: seamkit.remote imports Field from this module.
    from seamkit.ingress import rules_of
    from seamkit.remote import SeamSpec

    return SeamSpec(expects.seam, expects.owner, expects.skill, expects.sends,
                    receives=dict(expects.into), asks_caller=bool(expects.answers),
                    timeout=expects.slo_seconds, shapes=dict(expects.reads),
                    rules=rules_of(expects), roles=expects.roles,
                    wait_seconds=expects.wait_seconds, retry=expects.retry,
                    lookup=expects.lookup)


def args_model(skill_id: str, skill: SkillTerms) -> type[BaseModel]:
    """A tool's argument schema, generated from the partner's input terms."""
    fields: dict[str, Any] = {
        f.name: (TYPES[f.type], ...) if f.required else (TYPES[f.type] | None, None)
        for f in skill.inputs}
    model: type[BaseModel] = create_model(f"{skill_id}_args", **fields)
    return model


__all__ = ["EXTENSION_URI", "NO_CROSSING", "Breach", "CrossingPolicy", "EvidenceTerms",
           "Expectations", "Field", "SeamTerms", "SkillTerms", "args_model", "check",
           "held_to", "load_expectations", "publish", "revised", "seam_spec",
           "terms_of", "violations"]
