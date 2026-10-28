"""The Federation Test and the federation decision record.

Four questions asked of every proposed agent-to-agent call. The owner question decides
whether the call is a seam at all; cadence, trust domain and data boundary decide which
seam properties it hardens first. The answers are derived from a table of owner facts,
never typed per call, so changing one fact changes every verdict it touches.

misplaced() runs the test the other way round, on a compiled graph and the set of nodes
deployed behind a seam: it finds a remote call with one owner on both sides, and an
owner crossing still running in-process.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph

Option = Literal["subgraph", "mcp_tool", "a2a_agent"]
OPTIONS: tuple[Option, ...] = ("subgraph", "mcp_tool", "a2a_agent")
QUESTIONS = ("owner", "cadence", "trust_domain", "data_boundary")
# The seam properties a "yes" puts first, and the chapters that harden them.
PUTS_FIRST = {"cadence": ("contract",), "trust_domain": ("authority",),
              "data_boundary": ("contract", "authority")}
HARDENED_BY = {"contract": "Ch 4, 15", "authority": "Ch 5-8"}


@dataclass(frozen=True)
class OwnerFacts:
    owner: str
    cadence: str  # the release train the owner ships on
    trust_domain: str  # the identity authority its workloads authenticate against
    data_boundary: str  # the boundary its data sits behind


@dataclass(frozen=True)
class Candidate:
    caller: str
    callee: str
    caller_facts: OwnerFacts
    callee_facts: OwnerFacts
    stateful: bool  # the callee has steps worth checkpointing
    model_decides: bool  # a model, not the graph, decides whether the call happens


@dataclass(frozen=True)
class Verdict:
    answers: dict[str, tuple[bool, str]]  # question -> (yes?, the facts that decided)
    option: Option
    first: tuple[str, ...]  # seam properties to harden first
    refused: dict[str, str]  # the other two options, each with its reason
    warnings: tuple[str, ...] = ()  # a "yes" with no owner line: not a reason to seam


def federation_test(c: Candidate) -> Verdict:
    a, b = c.caller_facts, c.callee_facts
    answers = {q: (getattr(a, q) != getattr(b, q), f"{getattr(a, q)} / {getattr(b, q)}")
               for q in QUESTIONS}
    rest = [q for q in QUESTIONS[1:] if answers[q][0]]
    if not answers["owner"][0]:  # one owner: no seam, whatever else differs
        tool = c.model_decides and not c.stateful
        option: Option = "mcp_tool" if tool else "subgraph"
        return Verdict(answers, option, (), _refusals(c, option), tuple(rest))
    first = tuple(dict.fromkeys(p for q in rest for p in PUTS_FIRST[q]))
    return Verdict(answers, "a2a_agent", first, _refusals(c, "a2a_agent"))


def _refusals(c: Candidate, chosen: Option) -> dict[str, str]:
    owner = c.callee_facts.owner
    reasons: dict[str, str] = {
        "subgraph": (
            f"{owner} deploys and releases its own runtime; a subgraph compiles it "
            "into ours" if chosen == "a2a_agent"
            else "a model, not the graph, decides whether the call happens"),
        "mcp_tool": (
            f"{owner} runs its own state, people and task lifecycle behind the call"
            if chosen == "a2a_agent"
            else "it has steps worth checkpointing, and they belong in our checkpoint"
            if c.stateful
            else "the graph, not a model, decides whether the call happens"),
        "a2a_agent": f"one owner ({owner}) on both sides: a seam's costs, none of its "
                     "reasons",
    }
    return {o: reasons[o] for o in OPTIONS if o != chosen}


@dataclass(frozen=True)
class Finding:
    caller: str
    callee: str
    problem: Literal["remote with one owner", "owner line in-process"]


def misplaced(graph: CompiledStateGraph[Any, Any, Any, Any],
              owners: Mapping[str, str], remote: Collection[str]) -> list[Finding]:
    found = []
    for edge in graph.get_graph().edges:
        a, b = edge.source, edge.target
        if not edge.conditional or a not in owners or b not in owners:
            continue  # the dispatch edge is the call; the edge back is its return
        crosses, deployed = owners[a] != owners[b], b in remote
        if deployed and not crosses:
            found.append(Finding(a, b, "remote with one owner"))
        elif crosses and not deployed:
            found.append(Finding(a, b, "owner line in-process"))
    return found


TEMPLATE = """\
# Federation decision: {caller} -> {callee}

Status: {status}

## Context
{caller} ({caller_owner}) calls {callee} ({callee_owner}).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
{answer_rows}

## Decision
{option}. {decision_note}

## Options refused
{refused_rows}

## Predicted costs
{cost_rows}

## Shape
{shape}

## Paged
{paged}

## Revisit when
{revisit}
"""


@dataclass(frozen=True)
class DecisionRecord:
    candidate: Candidate
    verdict: Verdict
    costs: Mapping[str, str] = field(default_factory=dict)
    shape: str = ""
    paged: str = ""
    revisit: str = ""

    def render(self) -> str:
        c, v = self.candidate, self.verdict
        seam = v.option == "a2a_agent"
        if seam:
            first = ", ".join(f"{p} ({HARDENED_BY[p]})" for p in v.first)
            note = "A seam. Harden first: " + (first or "nothing extra")
        else:
            note = "Not a seam: no owner line."
            if v.warnings:
                differs = " and ".join(v.warnings)
                note += (f"\nWarning: {differs} differs. Under one owner that is a "
                         "deployment or access\nproblem, not a reason for a seam.")
        return TEMPLATE.format(
            caller=c.caller, callee=c.callee,
            caller_owner=c.caller_facts.owner, callee_owner=c.callee_facts.owner,
            status="accepted" if seam else "refused as a seam",
            answer_rows="\n".join(
                f"| {q} | {'yes' if yes else 'no'} | {why} |"
                for q, (yes, why) in v.answers.items()),
            option=v.option, decision_note=note,
            refused_rows="\n".join(f"- {o}: {why}" for o, why in v.refused.items()),
            cost_rows="\n".join(f"- {k}: {val}" for k, val in self.costs.items())
            or "- none recorded",
            shape=self.shape or "-", paged=self.paged or "nobody",
            revisit=self.revisit or "-")
