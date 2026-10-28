"""The seam ingress filter: what a partner returns is untrusted input.

Every answer passes it after the contract's shape check and before any node reads it.
A field is a closed value or it is text. A number, an integer or a boolean is a value;
a string is a value only if the caller's expectations close it with a pattern or a set
of choices. Every other string is text: normalised, capped, and tagged as text, so that
only a model with no tools and no other state may read it (quarantine), and nothing
written from it leaves through another seam (egress).

Detection is not the gate. `signals` flags text that looks addressed to a model, for
the log and an alert; the structure decides what the text can reach.

Nothing in this module knows about Meridian: the rules come from the caller's
expectations, the classes from the caller's crossing policy.
"""

import json
import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from seamkit.errors import MalformedAnswer, UntrustedEgress

if TYPE_CHECKING:
    from seamkit.contract import CrossingPolicy, Expectations, Field

Kind = Literal["value", "text"]

DEFAULT_MAX_CHARS = 2_000  # a text field the expectations do not cap
QUESTION_MAX_CHARS = 500  # a partner's input-required question, before a person sees it
DETAIL_MAX_CHARS = 200  # a refusal reason, before it reaches a log or a model
CLOSED_TYPES = frozenset({"number", "integer", "boolean"})


@dataclass(frozen=True)
class IngressRule:
    """How one field a partner returns may enter state: closed, or quarantined."""

    field: str  # our state field
    kind: Kind  # "value" is closed and checkable; "text" is quarantined
    max_chars: int = DEFAULT_MAX_CHARS  # text only: a longer answer is refused
    pattern: str = ""  # a string that matches this whole is a value
    choices: tuple[str, ...] = ()  # or one of these
    items: Mapping[str, "IngressRule"] = dc_field(default_factory=dict)  # rows' keys


class Provenance(TypedDict):
    """Where a state field came from: kept beside it, never inside it."""

    seam: str  # the seam that wrote it; "" for a field Claims computed
    task_id: str  # the partner's task whose answer wrote it
    release: str  # the partner release that answered
    kind: Kind  # "text" until a closed value stands between it and the partner
    derived_from: list[str]  # the fields it was computed from, Claims' own included


def merge_provenance(current: Mapping[str, Provenance],
                     update: Mapping[str, Provenance]) -> dict[str, Provenance]:
    """The ClaimState.provenance reducer: a field's latest write brings its tag."""
    return {**current, **update}


@dataclass(frozen=True)
class Admitted:
    """What the filter let into state, the tags for it, and what it noticed."""

    update: dict[str, Any]
    tags: dict[str, Provenance]
    findings: list[str]  # for the log and an alert; never the text itself


def rule_for(name: str, shape: "Field", closes: Mapping[str, Any] | None = None,
             ) -> IngressRule:
    """A field's rule: closed by its type, by a pattern or choices, or text."""
    closes = closes or {}
    pattern = str(closes.get("pattern", ""))
    choices = tuple(str(c) for c in closes.get("choices", ()))
    closed = shape.type in CLOSED_TYPES or (shape.type == "string"
                                            and bool(pattern or choices))
    items = {key: rule_for(key, _shape(spec), spec)
             for key, spec in closes.get("items", {}).items()}
    if shape.type == "object[]" and items:
        closed = all(rule.kind == "value" for rule in items.values())
    return IngressRule(name, "value" if closed else "text",
                       int(closes.get("max_chars", DEFAULT_MAX_CHARS)), pattern,
                       choices, items)


def rules_of(expects: "Expectations") -> dict[str, IngressRule]:
    """The seam's rules, read from the same expectations file as its field map."""
    return {expects.into[key]: rule_for(expects.into[key], shape,
                                        expects.ingress.get(key))
            for key, shape in expects.reads.items()}


def admit(answer: Mapping[str, Any], rules: Mapping[str, IngressRule], seam: str,
          task_id: str, release: str) -> Admitted:
    """Close or quarantine every field of a mapped answer, and tag what gets in."""
    update: dict[str, Any] = {}
    tags: dict[str, Provenance] = {}
    findings: list[str] = []
    for name, value in answer.items():
        rule = rules.get(name)
        if rule is None:  # no rule, no entry: an unread field never reaches state
            findings.append(f"{name}: dropped, no ingress rule")
            continue
        if rule.kind == "value":
            update[name] = closed_value(rule, value, seam, task_id, findings)
        else:
            text, removed = clean(value)
            if (size := chars(text)) > rule.max_chars:  # refused, never truncated
                raise MalformedAnswer(seam, f"{name} is {size} characters, over "
                                            f"its {rule.max_chars}", task_id)
            if removed:
                findings.append(f"{name}: removed {removed} invisible character(s)")
            findings += [f"{name}: {s}" for s in signals(text)]
            update[name] = text
        tags[name] = tag(rule, seam, task_id, release)
    return Admitted(update, tags, findings)


def tag(rule: IngressRule, seam: str, task_id: str, release: str) -> Provenance:
    """The tag of a field a partner's answer wrote: its seam, task, release and kind."""
    return Provenance(seam=seam, task_id=task_id, release=release, kind=rule.kind,
                      derived_from=[])


def closed_value(rule: IngressRule, value: Any, seam: str, task_id: str,
                 findings: list[str]) -> Any:
    """A value the rule closes, or MalformedAnswer. Rows keep only the keys it names."""
    if rule.items:
        if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
            raise MalformedAnswer(seam, f"{rule.field} is not a list of rows", task_id)
        extra = sorted({k for row in value for k in row} - set(rule.items))
        if extra:
            findings.append(f"{rule.field}: dropped {', '.join(extra)} from its rows")
        return [{k: closed_value(r, row[k], seam, task_id, findings)
                 for k, r in rule.items.items() if k in row} for row in value]
    if isinstance(value, str):
        fits = (re.fullmatch(rule.pattern, value) if rule.pattern
                else value in rule.choices)
        if not fits:
            raise MalformedAnswer(seam, f"{rule.field} is not a closed value", task_id)
    elif not isinstance(value, int | float):  # booleans are ints; nothing else is
        raise MalformedAnswer(seam, f"{rule.field} is not a closed value", task_id)
    return value


# Normalising text ----------------------------------------------------------------------

def clean(value: Any) -> tuple[Any, int]:
    """Every string with control, format and tag characters removed, and the count.

    Unicode tag characters (U+E0000-U+E007F) and zero-width characters render as
    nothing and still reach a model's tokenizer. Newlines and tabs are kept.
    """
    if isinstance(value, str):
        kept = "".join(c for c in value if c in "\n\t"
                       or unicodedata.category(c) not in ("Cc", "Cf", "Co", "Cn"))
        return kept, len(value) - len(kept)
    if isinstance(value, list):
        pairs = [clean(v) for v in value]
        return [v for v, _ in pairs], sum(n for _, n in pairs)
    if isinstance(value, dict):
        items = {k: clean(v) for k, v in value.items()}
        return {k: v for k, (v, _) in items.items()}, sum(n for _, n in items.values())
    return value, 0


def chars(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(json.dumps(value))


def quote(value: Any, limit: int) -> Any:
    """Partner text for a person or a log: cleaned, and cut to `limit` characters."""
    text, _ = clean(value)
    if isinstance(text, str):
        return text if len(text) <= limit else text[:limit] + " [cut]"
    if isinstance(text, dict):
        return {k: quote(v, limit) for k, v in text.items()}
    if isinstance(text, list):
        return [quote(v, limit) for v in text]
    return text


# The alarm, never the gate ---------------------------------------------------------------

SIGNALS: dict[str, re.Pattern[str]] = {
    "addresses a model": re.compile(
        r"\b(?:note|message|instructions?)\s+(?:for|to)\s+(?:\w+\s+){0,2}"
        r"(?:reviewers?|assistants?|models?|agents?|bots?|ai)\b"
        r"|\b(?:ai|automated|assistant|llm|bot|agent)s?\b.{0,60}"
        r"\b(?:should|must|please|are asked)\b", re.I | re.S),
    "overrides instructions": re.compile(
        r"\b(?:ignore|disregard|override|forget)\b.{0,40}"
        r"\b(?:instructions?|rules?|previous|above)\b", re.I | re.S),
    "role marker": re.compile(r"^\s*(?:system|assistant|user)\s*:", re.I | re.M),
    "asks to copy data": re.compile(
        r"\b(?:copy|include|paste|forward|add)\b.{0,80}"
        r"\b(?:memo|record|reply|output|response)\b", re.I | re.S),
}


def signals(text: Any) -> list[str]:
    """Which instruction-like patterns the text trips. False negatives are expected."""
    flat = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return [name for name, pattern in SIGNALS.items() if pattern.search(flat)]


ingress_logger = logging.getLogger("seamkit.ingress")


def log_findings(seam: str, task_id: str, findings: list[str]) -> None:
    for finding in findings:  # the field and the finding, never the text
        ingress_logger.warning("ingress finding", extra={"ingress": {
            "seam": seam, "task_id": task_id, "finding": finding}})


# Derivation and egress -----------------------------------------------------------------

def derive(tags: Mapping[str, Provenance], *inputs: str,
           closed: bool = False) -> Provenance:
    """The tag of a field computed from `inputs`: text if any input was, unless a
    closed schema stands between them; derived from every input and its sources."""
    sources: set[str] = set()
    text = False
    for name in inputs:
        tag = tags.get(name)
        sources.add(name)
        if tag is not None:
            sources.update(tag["derived_from"])
            text = text or tag["kind"] == "text"
    return Provenance(seam="", task_id="", release="",
                      kind="value" if closed or not text else "text",
                      derived_from=sorted(sources))


def egress(data: Mapping[str, Any], provenance: Mapping[str, Provenance], owner: str,
           policy: "CrossingPolicy", seam: str) -> None:
    """Refuse a send carrying partner text, or a class `owner` may not hold.

    A field's classes are its own and those of everything it was derived from.
    """
    allowed = policy.allowed.get(owner, frozenset())
    found: list[str] = []
    for name in data:
        tag = provenance.get(name)
        if tag is None:
            continue  # Claims' own field: Chapter 4's check already classed it
        if tag["kind"] == "text":
            found.append(f"{name} carries partner text"
                         + (f" from {tag['seam']}" if tag["seam"] else ""))
        classes = {policy.classes.get(f, "unclassified")
                   for f in (name, *tag["derived_from"])}
        found += [f"{name} is derived from {c} data, which may not cross to {owner}"
                  for c in sorted(classes - allowed)]
    if found:
        raise UntrustedEgress(seam, "; ".join(found))


def withhold(update: Mapping[str, Any],
             rules: Mapping[str, IngressRule]) -> dict[str, Any]:
    """A tool result for a model that can act: values, and no text."""
    return {name: (value if rules[name].kind == "value"
                   else {"withheld": "partner text", "chars": chars(value)})
            for name, value in update.items()}


def _shape(spec: Mapping[str, Any]) -> "Field":
    from seamkit.contract import Field  # contract imports this module's users

    return Field(str(spec.get("name", "")), str(spec.get("type", "string")))


__all__ = ["Admitted", "IngressRule", "Kind", "Provenance", "UntrustedEgress", "admit",
           "clean", "derive", "egress", "log_findings", "merge_provenance", "quote",
           "rule_for", "rules_of", "signals", "tag", "withhold"]
