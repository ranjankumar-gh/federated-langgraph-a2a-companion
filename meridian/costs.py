"""Claims' cost budgets, its seam cost ledger, and its invoice check, as Meridian
deploys them (Chapter 18).

Two seams bill in two shapes. Northgate is another company and states a price per task
in its terms (`cost_per_task`, 3.5 on estimate_repair), billed to Meridian monthly.
Risk is a team inside Meridian whose model spend Finance charges back to the Claims
team's cost centre at the rate card's price; from release 2027.04 Risk signs a usage
report into every screening and honours the ceiling Claims sends.

Each seam's budget is in its expectations file's [budget] table, in the costed
variants (`*.costed.toml`), so the earlier chapters' federations are unchanged. A
compensation's charge, the residue a partner says is owed, lands on Claims' own
compensation line: Meridian decided to cancel.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from meridian.receipts import OWNERSHIP, pinned
from meridian.rates import RATES
from meridian.saga import EXPECTED
from meridian.trust import SAGA_TRUST
from northgate.server import TERMS as NORTHGATE_TERMS
from seamkit.contract import Expectations, load_expectations
from seamkit.costs import CostBudget, CostLedger, Costs, Spend, spend
from seamkit.inventory import SeamInventory
from seamkit.ownership import OwnershipMap, Page, Signal
from seamkit.remote import RemoteAgentNode
from seamkit.trust import PartnerTrust

COSTED: dict[str, Expectations] = {
    name: load_expectations(EXPECTED / f"{name}.costed.toml")
    for name in ("fraud_screening", "repair_estimate")}

WINDOW = 28 * 86_400.0
# Claims' own line for what a compensation leaves owed. Nothing is paid for a cancel
# itself, so its ceiling is 0; the residue is owed under the booking's terms.
COMPENSATION = CostBudget(0.0, WINDOW, {"compensation": 20_000.0})

# revision -> skill -> the price its terms state: what the invoice check prices by.
NORTHGATE_PRICES: dict[str, dict[str, float | None]] = {
    terms.revision: {sid: s.cost_per_task for sid, s in terms.skills.items()}
    for terms in NORTHGATE_TERMS.values()}


def budget_for(seam: str) -> CostBudget | None:
    if seam.endswith(".undo"):
        return COMPENSATION
    held = COSTED.get(seam)
    return held.budget if held else None


def paging(owners: OwnershipMap, pages: list[Page],
           now: Callable[[], float]) -> Callable[[str, str], None]:
    """A cost signal, routed through Claims' ownership map like every other."""
    def signal(seam: str, detail: str) -> None:
        pages.extend(owners.route(Signal(seam, "cost", detail, now())))
    return signal


def costed(nodes: Iterable[RemoteAgentNode], ledger: CostLedger,
           signal: Callable[[str, str], None] = lambda seam, detail: None,
           trust: Mapping[str, PartnerTrust] = SAGA_TRUST,
           budgets: Mapping[str, CostBudget] | None = None) -> None:
    """Give each node with a budget what it needs to price its tasks, into one ledger.

    `budgets` replaces a seam's budget, as a test shrinks a window's line.
    """
    keys = pinned(trust)
    for node in nodes:
        budget = (budgets or {}).get(node.seam.name) or budget_for(node.seam.name)
        if budget is None:
            continue
        held = node.partner.terms[0] if node.partner.terms else None
        skill = held.skills.get(node.seam.skill) if held else None
        node.costs = Costs(budget, ledger, RATES, keys,
                           reported=bool(skill and skill.usage == "reported"),
                           signal=signal)


def with_costs(inventory: SeamInventory, ledger: CostLedger, now: float,
               budgets: Mapping[str, CostBudget] | None = None) -> SeamInventory:
    """Add each budgeted seam's cost to the accountability column Chapter 17 filled."""
    for entry in inventory.entries:
        budget = (budgets or {}).get(entry.callee) or budget_for(entry.callee)
        if budget is None:
            continue
        held = COSTED[entry.callee].skill if entry.callee in COSTED else ""
        spent: dict[str, Spend] = spend(ledger, entry.callee, budget, now)
        total = sum(budget.lines.values())
        price = (f"cost: {budget.per_task:g} per task" if held == "estimate_repair"
                 else f"cost: usage reported, ceiling {budget.per_task:g}")
        entry.answers["accountability"] = (
            entry.answers.get("accountability", "") + f"; {price}, budget "
            f"{total:,.0f} per {budget.window_seconds / 86_400:g} days, spent "
            f"{sum(s.spent for s in spent.values()):,.2f}")
    return inventory


def shrunk(seam: str, purpose: str, line: float) -> dict[str, CostBudget]:
    """One seam's budget with one line replaced: how a test spends a window quickly."""
    budget = budget_for(seam)
    assert budget is not None
    return {seam: replace(budget, lines={**budget.lines, purpose: line})}


__all__ = ["COMPENSATION", "COSTED", "NORTHGATE_PRICES", "OWNERSHIP", "budget_for",
           "costed", "paging", "shrunk", "with_costs"]
