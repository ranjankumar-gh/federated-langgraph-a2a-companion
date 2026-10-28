"""The seam contract: the partner's published terms and the caller's expectations.

The two halves have two owners. The partner publishes its terms on its agent card, as a
data-only extension, one dated revision at a time; the caller keeps its expectations in
its own repository. `check` holds one against the other before a task is sent, and
reads declarations only: a partner whose behaviour changes under unchanged terms passes.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from a2a.types import AgentCard, AgentExtension
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, create_model

from seamkit.breaker import MODES, BreakerPolicy
from seamkit.retry import NO_RETRY, RetryPolicy

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
    cost_per_task: float | None = None  # declared here, enforced from Chapter 18
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
    # Chapter 8: artifact key -> how a read is closed (pattern, choices, items) or
    # capped (max_chars); seamkit.ingress turns these into the seam's ingress rules.
    ingress: Mapping[str, Mapping[str, Any]] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class CrossingPolicy:
    """The caller's data classes, and which of them may cross to each owner."""

    classes: Mapping[str, str]  # field -> data class
    allowed: Mapping[str, frozenset[str]]  # owner -> classes that may cross to it


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
    return [_breach(terms, expects, d) for d in found]


def held_to(revisions: Sequence[SeamTerms], expects: Expectations,
            policy: CrossingPolicy) -> tuple[list[Breach], list[Breach]]:
    """Breaches in the revision in force, and in every revision announced after it."""
    if not revisions:
        return [Breach(expects.seam, "none", date.min,
                       f"{expects.owner} publishes no terms")], []
    current, *announced = revisions
    return (check(current, expects, policy),
            [b for terms in announced for b in check(terms, expects, policy)])


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
        out = {k: v for k, v in asdict(s).items() if v not in (None, "", ())}
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
        else None)


def _terms(raw: Mapping[str, Any]) -> SeamTerms:
    revocation = raw.get("revocation_seconds")
    return SeamTerms(
        str(raw["revision"]), date.fromisoformat(str(raw["effective"])),
        {sid: _skill(s) for sid, s in raw.get("skills", {}).items()},
        tuple(str(a) for a in raw.get("auth", [])),
        int(raw.get("skew_seconds", 0)),  # protobuf Struct numbers arrive as floats
        frozenset(str(c) for c in raw.get("accepts", [])),
        int(revocation) if revocation is not None else None)


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
        breaker=_breaker(raw["breaker"]) if "breaker" in raw else None)


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


__all__ = ["EXTENSION_URI", "NO_CROSSING", "Breach", "CrossingPolicy", "Expectations",
           "Field", "SeamTerms", "SkillTerms", "args_model", "check", "held_to",
           "load_expectations", "publish", "seam_spec", "terms_of", "violations"]
