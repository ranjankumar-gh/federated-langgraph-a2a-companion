"""The fault script: failures a stub partner produces on demand.

A fault is a failure to answer, and only a failure. Anything the partner's terms let it
answer (a refusal, a question, a compensation that leaves a residue) is an exchange,
recorded or scripted, and held to the terms when the stub is built. So a fault script
can make the stub slow, down or broken, and it can never make it say something the
partner never promised.

Each kind acts in one layer:

- on the wire: "slow" (the send takes `seconds`, on the drill's clock), "down" (no
  answer at all), "lose_answer" (the partner does the work, and the answer never
  reaches the caller);
- in the executor: "fail" (the task ends failed), "hang" (the task stays working until
  the test finishes it), and "push_lost" (a hang whose finish rings nobody).
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

WIRE = frozenset({"slow", "down", "lose_answer"})
EXECUTOR = frozenset({"fail", "hang", "push_lost"})
KINDS = WIRE | EXECUTOR


@dataclass(frozen=True)
class Fault:
    """One failure: what kind, and for how long or with what detail."""

    kind: str  # one of KINDS
    seconds: float = 0.0  # "slow": how long the send takes
    detail: str = ""  # "fail": the reason the failed task carries

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"fault {self.kind!r} is not one of {sorted(KINDS)}")


@dataclass(frozen=True)
class Rule:
    """Which sends a fault applies to: a skill, a claim, and which call of them."""

    fault: Fault
    skill: str = ""  # "" is any skill
    claim_id: str = ""  # "" is any claim
    nth: int = 0  # 0 is every matching call; n is only the nth

    def covers(self, skill: str, request: Mapping[str, Any]) -> bool:
        return ((not self.skill or skill == self.skill)
                and (not self.claim_id or request.get("claim_id") == self.claim_id))


@dataclass
class FaultScript:
    """The rules, in order: the first rule that covers a send decides its fault."""

    rules: tuple[Rule, ...] = ()
    calls: Counter[int] = field(default_factory=Counter)  # rule index -> sends covered

    def match(self, skill: str, request: Mapping[str, Any],
              layer: frozenset[str]) -> Fault | None:
        """The fault for this send in one layer, or None. Each rule counts only the
        sends of its own layer, so a send is counted once, by the layer that acts."""
        for index, rule in enumerate(self.rules):
            if rule.fault.kind not in layer or not rule.covers(skill, request):
                continue
            self.calls[index] += 1
            if rule.nth in (0, self.calls[index]):
                return rule.fault
        return None

    def add(self, fault: Fault, skill: str = "", claim_id: str = "",
            nth: int = 0) -> "FaultScript":
        self.rules = (*self.rules, Rule(fault, skill, claim_id, nth))
        return self


__all__ = ["EXECUTOR", "KINDS", "WIRE", "Fault", "FaultScript", "Rule"]
