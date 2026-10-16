"""The seam inventory: one row per call that crosses an ownership boundary.

The rows are derived from a compiled graph's edges and an owner map, never typed by
hand, so a call added after the architecture review still gets a row.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph

PROPERTIES = ("contract", "authority", "state", "failure", "accountability")
HARDENED_IN = {"contract": "Ch 4, 15", "authority": "Ch 5-8", "state": "Ch 9-13",
               "failure": "Ch 14, 16", "accountability": "Ch 17-19"}


@dataclass
class SeamEntry:
    caller: str
    callee: str
    caller_owner: str
    callee_owner: str
    answers: dict[str, str] = field(default_factory=dict)  # property -> how it is held
    paged: str = ""  # who is woken when this seam fails

    def open_properties(self) -> list[str]:
        return [p for p in PROPERTIES if not self.answers.get(p)]


@dataclass
class SeamInventory:
    entries: list[SeamEntry]

    def render(self) -> str:
        head = ["seam", "owners", *PROPERTIES, "paged"]
        rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for e in self.entries:
            cells = [e.answers.get(p) or f"open ({HARDENED_IN[p]})" for p in PROPERTIES]
            rows.append(f"| {e.caller} -> {e.callee} | {e.caller_owner} -> "
                        f"{e.callee_owner} | " + " | ".join(cells)
                        + f" | {e.paged or 'nobody'} |")
        return "\n".join(rows)


def inventory_from_graph(graph: CompiledStateGraph[Any, Any, Any, Any],
                         owners: Mapping[str, str]) -> SeamInventory:
    seams: dict[frozenset[str], SeamEntry] = {}
    edges = sorted(graph.get_graph().edges, key=lambda e: not e.conditional)
    for edge in edges:  # dispatch edges first, so the caller is the side that routes
        a, b = edge.source, edge.target
        if a not in owners or b not in owners or owners[a] == owners[b]:
            continue  # same owner, or START/END: not a seam
        seams.setdefault(frozenset((a, b)), SeamEntry(a, b, owners[a], owners[b]))
    return SeamInventory(list(seams.values()))
