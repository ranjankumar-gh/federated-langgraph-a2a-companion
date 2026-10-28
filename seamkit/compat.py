"""The compatibility matrix: every revision of a partner's terms Claims has seen, held
to Claims' current expectations, with the behaviour the canaries found.

`diff` labels each change between two revisions of one seam's skill as breaking or safe
for one expectations file: the function Chapter 4's third exercise asked for. `matrix`
builds one row per revision, in the order Claims first saw them, and `render` prints
the rows as the Markdown table committed beside the seam's expectations. A CI test
regenerates the table; a changed cell fails the build until a person commits it.

The release string is recorded in a row and never decides anything. The partner may
move it, or not, with any revision of its terms.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any
from pathlib import Path

from seamkit.contract import (
    CrossingPolicy,
    Expectations,
    Field,
    SeamTerms,
    SkillTerms,
    check,
)

# The scalar terms of a skill whose change `diff` reports; a slower p95 is the only
# one a seam's expectations can make breaking, and `check` already holds it to that.
TERMS = ("p95_seconds", "cost_per_task", "wait_seconds", "key_seconds",
         "compensated_by", "compensate_seconds")


@dataclass(frozen=True)
class Change:
    """One difference between two revisions of the seam's skill."""

    kind: str  # "input added", "output removed", "range changed", ...
    field: str
    breaking: bool  # for this seam's expectations; safe here may break another caller
    detail: str = ""

    def __str__(self) -> str:
        mark = "breaking" if self.breaking else "safe"
        detail = f" ({self.detail})" if self.detail else ""
        return f"{self.kind} {self.field}{detail}: {mark}"


def diff(old: SeamTerms, new: SeamTerms, expects: Expectations) -> list[Change]:
    """Every change to the seam's skill from `old` to `new`, breaking or safe."""
    was, now = old.skills.get(expects.skill), new.skills.get(expects.skill)
    if was is None or now is None:
        gone = was is not None and now is None
        return [Change("skill removed" if gone else "skill added", expects.skill,
                       gone)] if was or now else []
    return _inputs(was, now, expects) + _outputs(was, now, expects) + [
        Change("ask added", q, q not in (*expects.answers, *expects.refuses))
        for q in now.asks if q not in was.asks] + [
        Change("ask removed", q, False) for q in was.asks if q not in now.asks] + [
        Change("term changed", t, _slower(t, now, expects),
               f"{getattr(was, t)} to {getattr(now, t)}")
        for t in TERMS if getattr(was, t) != getattr(now, t)] + _seam_terms(
        old, new, expects)


def _seam_terms(old: SeamTerms, new: SeamTerms, expects: Expectations) -> list[Change]:
    """The terms every skill shares: auth, skew, revocation, and data classes."""
    changes = []
    for term in ("auth", "skew_seconds", "revocation_seconds", "accepts"):
        a, b = getattr(old, term), getattr(new, term)
        if a == b:
            continue
        shown = (sorted(a), sorted(b)) if isinstance(a, frozenset | tuple) else (a, b)
        changes.append(Change("term changed", term, _breaks(term, a, b, expects),
                              f"{shown[0]} to {shown[1]}"))
    return changes


def _breaks(term: str, a: Any, b: Any, expects: Expectations) -> bool:
    if term == "auth":  # a scheme the seam does not present
        return bool(set(b) - set(a) - set(expects.auth))
    if term == "skew_seconds":
        return bool(b != expects.skew_seconds)
    if term == "revocation_seconds":
        return expects.revocation_seconds is not None and (
            b is None or b > expects.revocation_seconds)
    return bool(set(a) - set(b))  # accepts: a class dropped may be one this seam sends


def _inputs(was: SkillTerms, now: SkillTerms, expects: Expectations) -> list[Change]:
    old, new = {f.name: f for f in was.inputs}, {f.name: f for f in now.inputs}
    changes = [Change("input added", n, new[n].required and n not in expects.sends,
                      "required" if new[n].required else "optional")
               for n in new if n not in old]
    changes += [Change("input removed", n, n in expects.sends) for n in old
                if n not in new]
    return changes + [Change("input now required", n, n not in expects.sends)
                      for n in new if n in old and new[n].required
                      and not old[n].required]


def _outputs(was: SkillTerms, now: SkillTerms, expects: Expectations) -> list[Change]:
    """An output change breaks the seam if a field it reads no longer satisfies it."""
    old, new = {f.name: f for f in was.outputs}, {f.name: f for f in now.outputs}
    changes = []
    for name in [*old, *(n for n in new if n not in old)]:
        a, b = old.get(name), new.get(name)
        if a == b:
            continue
        want = expects.reads.get(name)
        breaks = want is not None and not want.satisfied_by(b)
        changes.append(Change(_kind(a, b), name, breaks, _shift(a, b)))
    return changes


def _kind(a: Field | None, b: Field | None) -> str:
    if a is None or b is None:
        return "output added" if a is None else "output removed"
    if a.type != b.type:
        return "type changed"
    if a.bounds != b.bounds:
        return "range changed"
    return "unit changed" if a.unit != b.unit else "requirement changed"


def _shift(a: Field | None, b: Field | None) -> str:
    if a is None or b is None or a.bounds == b.bounds:
        return ""
    return f"{a.bounds} to {b.bounds}"


def _slower(term: str, now: SkillTerms, expects: Expectations) -> bool:
    return (term == "p95_seconds" and now.p95_seconds is not None
            and now.p95_seconds > expects.slo_seconds)


@dataclass(frozen=True)
class Seen:
    """When Claims first saw a revision on the card, and the release that carried it."""

    first_seen: date
    release: str


def load_seen(path: Path) -> dict[str, Seen]:
    """Read one seam's record of first sightings, kept in Claims' repository."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return {rev: Seen(row["first_seen"], str(row["release"]))
            for rev, row in raw.get("revisions", {}).items()}


def record_seen(path: Path, card_revisions: Sequence[SeamTerms], release: str,
                today: date) -> dict[str, Seen]:
    """Add every revision on the card that Claims has not seen before, dated today.

    A revision already recorded keeps its first date: that date is the notice the
    partner gave, and nothing later may move it.
    """
    seen = load_seen(path) if path.exists() else {}
    for terms in card_revisions:
        seen.setdefault(terms.revision, Seen(today, release))
    lines = ["# When Claims first saw each revision of the partner's terms on its",
             "# card. Written by record_seen, committed, and never edited by hand.", ""]
    for rev, row in seen.items():
        lines += [f'[revisions."{rev}"]', f"first_seen = {row.first_seen.isoformat()}",
                  f'release = "{row.release}"', ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return seen


@dataclass(frozen=True)
class Row:
    """One revision of the partner's terms, held to Claims' expectations today."""

    revision: str
    release: str
    first_seen: date
    effective: date
    changes: tuple[Change, ...]  # from the revision before it, in the order seen
    contract: str  # "compatible", or "breaking: " and each breach
    notice: str  # the days of notice given, and whether a breaking revision gave enough
    behaviour: str  # for the revision in force: the canaries' verdict and version


def matrix(revisions: Sequence[SeamTerms], expects: Expectations,
           policy: CrossingPolicy, seen: Mapping[str, Seen], in_force: str,
           behaviour: str = "") -> list[Row]:
    """One row per revision Claims has seen, in the order they take effect, each held
    to today's expectations and diffed against the revision before it."""
    ordered = sorted(revisions, key=lambda t: (t.effective, t.revision))
    current = next(t for t in ordered if t.revision == in_force)
    rows: list[Row] = []
    before: SeamTerms | None = None
    for terms in ordered:
        breaches = check(terms, expects, policy)
        days = (terms.effective - seen[terms.revision].first_seen).days
        short = (terms.effective > current.effective and bool(breaches)
                 and expects.notice_days is not None
                 and days < expects.notice_days)  # the notice rule, on what is to come
        notice = (f"{days} days" + (", short" if short else "")) if before else (
            "onboarding")
        rows.append(Row(
            terms.revision, seen[terms.revision].release,
            seen[terms.revision].first_seen, terms.effective,
            tuple(diff(before, terms, expects)) if before else (),
            "breaking: " + "; ".join(b.detail.removeprefix(f"{expects.skill} ")
                                     for b in breaches) if breaches else "compatible",
            notice, behaviour if terms.revision == in_force else ""))
        before = terms
    return rows


def render(seam: str, rows: Sequence[Row]) -> str:
    """The Markdown table committed at compat/<seam>.md."""
    head = ["revision", "release", "first seen", "effective", "changes", "contract",
            "notice", "behaviour"]
    lines = [f"# Compatibility matrix: {seam}", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        changes = "; ".join(str(c) for c in r.changes) or "none"
        cells = [r.revision, r.release, r.first_seen.isoformat(),
                 r.effective.isoformat(), changes, r.contract, r.notice,
                 r.behaviour or "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["Change", "Row", "Seen", "diff", "load_seen", "matrix", "record_seen",
           "render"]
