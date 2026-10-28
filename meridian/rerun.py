"""The Federation Test, rerun on Meridian (the closer).

Chapter 2's five decision records, each asked again with what its seam measured when
the federation ran end to end behind its owners' gateways: the option from the same
owner facts, the shape from what crossed, and every predicted cost beside its
measurement. Lines the records never had (the chain, the receipts, the price) are
added, and are not mistakes: the book had not built them when the records were written.

Wall-clock latency is printed by `python -m meridian.assembled` and never committed,
because it is a property of the machine. Everything committed here is deterministic.

Run `python -m meridian.rerun` to rewrite docs/decisions/rerun/; a test fails if the
committed files drift from what a fresh run renders.
"""

import asyncio
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from meridian.assembled import AUGUST, Assembled, Measured, assemble, august, measure
from meridian.decisions import DECISIONS_DIR, OWNERS, records
from seamkit.drill import Clock
from seamkit.federation import Crossed, DecisionRecord, Line, Rerun, rerun

# What each owner keeps behind its boundary and does not publish (Chapter 2's facts):
# Risk's fraud signals stay in Risk's checkpoint, and its reply carries only the score.
KEPT = {"Risk": ("fraud_signals",)}
RERUN_DIR = DECISIONS_DIR / "rerun"
FLOOR = "an in-process floor, printed by the run and never committed"


def lines(record: DecisionRecord, m: Measured) -> list[Line]:
    """Each predicted line beside its measurement, then the lines the record lacked."""
    owner = OWNERS[record.candidate.callee]
    predicted = record.costs
    out = [
        Line("re-sent per task", predicted["re-sent per task"],
             f"{len(m.fields)} field{'s' if len(m.fields) > 1 else ''} "
             f"({', '.join(m.fields)}): {m.request_bytes} bytes a request, "
             f"about {m.tokens} tokens"),
        Line("latency", predicted["latency"], FLOOR),
        Line("checkpoint visibility", predicted["checkpoint visibility"],
             f"{_s(owner)} steps in {_s(owner)} checkpoint; the task in "
             + (f"{_s(m.ledger)} ledger" if m.ledger != "none"
                else f"no ledger: {_s(m.caller)} crossing records only")),
        Line("authority", "", f"a {m.links}-link chain, {m.authority_bytes} bytes a "
             "request"),
        Line("receipts", "", f"{m.receipts} for {m.tasks} task{'s' if m.tasks > 1 else ''}"
             f", in {_s(m.caller)} log"),
        Line("cost per task", "",
             f"{m.cost_per_task:.2f} on {_s(m.caller)} cost ledger"
             if m.cost_per_task is not None else "nothing priced on any cost ledger"),
    ]
    return out


def _s(owner: str) -> str:
    """The possessive the book prints: Claims', Risk's."""
    return owner + ("'" if owner.endswith("s") else "'s")


def crossed(a: Assembled, seam: str) -> Crossed:
    """What crossed the peer seam, from the receipts Risk signed and Risk's rules."""
    log = a.risk.runtime.receipts
    sent = sorted({f for r in log.receipts([seam]) for f in r.ours.fields})
    back = list(a.risk.config.seams[seam].expects.into.values())
    return Crossed(tuple(sent), tuple(back),
                   chosen=a.risk.config.seams[seam].shape == "tool")


def opened(a: Assembled, seam: str, caller: str) -> list[str]:
    gw = a.gateways()[caller]
    return [f"{control} waived: {why}" for control, why in gw.waived(seam).items()]


def finding(peer: Mapping[str, Measured], hub: Mapping[str, Measured]) -> str:
    """The superseding decision for the peer seam, with what the move costs."""
    screen_peer, screen_hub = peer["fraud_screening"], hub["fraud_screening"]
    history, asked = hub["estimate_history"], peer["estimate_history"]
    return (
        "Superseded: hub. Claims asks Northgate for the history of every claim it "
        "screens, before screening, and sends it to Risk as a fifth field under Risk's "
        "terms revision 2027.08. Meridian's data-class policy lets estimate data "
        "cross to Risk, and the adjuster's root names the field.\n\n"
        f"Measured on the same claims: {history.tasks} history tasks for "
        f"{asked.tasks} (one for every screened claim, at the 0.5 a task Northgate's "
        f"terms state), a {history.links}-link chain for {asked.links}, the task in "
        f"Claims' ledger, and {screen_hub.request_bytes - screen_peer.request_bytes} "
        f"more bytes and {screen_hub.tokens - screen_peer.tokens} more tokens on every "
        "screening request. Risk's assistant no longer chooses when to look.\n\n"
        "Retired at Risk: its outbound gateway, its breaker on Northgate, the "
        "estimate_history entry on its ownership map, and both waivers.")


@dataclass
class Shapes:
    """Both shapes, run on the same claims, each on a clock of its own, and measured."""

    peer: Assembled
    hub: Assembled

    @property
    def measured_peer(self) -> dict[str, Measured]:
        return measure(self.peer)

    @property
    def measured_hub(self) -> dict[str, Measured]:
        return measure(self.hub)


async def measured(rounds: int = 1) -> Shapes:
    """Assemble each shape and run the three claims through it `rounds` times."""
    built: list[Assembled] = []
    for hub in (False, True):
        with august(Clock(AUGUST)) as clock:
            a = await assemble(clock, hub)
            for _ in range(rounds):
                await a.run()
        built.append(a)
    return Shapes(*built)


def reruns(shapes: Shapes) -> list[Rerun]:
    """Chapter 2's five records, rerun."""
    a, peer, hub = shapes.peer, shapes.measured_peer, shapes.measured_hub
    found = []
    for record in records():
        seam = record.candidate.callee
        if seam == "policy_lookup":  # a subgraph: nothing crosses, nothing to measure
            found.append(rerun(record, [
                Line("re-sent per task", record.costs["re-sent per task"],
                     "none: the subgraph reads the claim in place"),
                Line("request body (CLM-1001)", record.costs["request body (CLM-1001)"],
                     "306 bytes as the incident deployed it (Chapter 2's test, rerun "
                     "in the same suite); none as a subgraph")], None))
            continue
        m = peer[seam]
        if record.shape.startswith("peer"):
            found.append(rerun(record, lines(record, m), crossed(a, seam),
                               KEPT[OWNERS[record.candidate.caller]],
                               opened(a, seam, m.caller), finding(peer, hub)))
        else:
            found.append(rerun(record, lines(record, m), None,
                               open_=opened(a, seam, m.caller)))
    return found


def write_reruns(found: list[Rerun], directory: Path = RERUN_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for done in found:
        path = directory / f"{done.seam}.md"
        path.write_text(done.render(), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


async def main() -> None:  # pragma: no cover - rewrites the committed files
    for written in write_reruns(reruns(await measured())):
        print(written)


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["FLOOR", "KEPT", "RERUN_DIR", "Shapes", "crossed", "lines", "measured", "reruns",
           "write_reruns"]
