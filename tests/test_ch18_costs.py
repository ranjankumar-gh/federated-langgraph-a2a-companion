"""Who pays: the screening that cost three times as much, the ceiling that travels with
the task, the signed usage report, the cost ledger and its budget lines, and the tax in
tokens."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import seamkit.ledger
from meridian import federated_naive
from meridian.canaries import SCREENING_BASELINE, SCREENING_GOLDENS
from meridian.costs import COSTED, OWNERSHIP, costed, paging, shrunk, with_costs
from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import SEAMS, A2APartners, a2a_workers, connect_partners
from meridian.identity import CANARY_SESSION, KEYS
from meridian.inprocess import ClaimGraph, ClaimState, build_graph
from meridian.rates import RATES
from meridian.samples import clean_claim
from meridian.risk_server import (
    RISK_TERMS_2027_02,
    RISK_TERMS_2027_04,
    RiskPartner,
    ScreeningModel,
)
from northgate.server import NORTHGATE_V7_5
from seamkit.canary import CanaryPolicy, CanarySuite
from seamkit.contract import check
from seamkit.costs import (
    BudgetSpent,
    CostBudget,
    CostLedger,
    UsageReport,
    spend,
    tokens,
    usage_of,
)
from seamkit.errors import SeamError
from seamkit.inventory import SeamEntry, SeamInventory
from seamkit.ownership import Page
from seamkit.remote import CrossingRecord, RemoteAgentNode
from seamkit.slo import CLAIMS_REFUSED, SeamSLO, burn
from tests.wire import thread

pytestmark = pytest.mark.anyio

SCREENING = "fraud_screening"
EXPECTS = COSTED[SCREENING]
assert EXPECTS.slo is not None and EXPECTS.canary is not None
SLO: SeamSLO = EXPECTS.slo
CANARY: CanaryPolicy = EXPECTS.canary


def claims() -> list[dict[str, Any]]:
    """The two sample claims, as Claims sends them: the four fields the screening
    seam's terms name."""
    return [{"claim_id": "CLM-1001", "amount": 3_000.0, "days_to_report": 2,
             "prior_claims": 0},
            {"claim_id": "CLM-1002", "amount": 14_500.0, "days_to_report": 45,
             "prior_claims": 3}]


def release(name: str, priced: bool = True, honours: bool = True) -> RiskPartner:
    """Risk as deployed at one release. 2027.02 screens in one model pass; 2027.03 adds
    the self-check; 2027.04 reports signed usage and, if `honours`, reads the ceiling."""
    records = asyncio.Event()
    records.set()  # Risk 2026.12's records lookup, answered at once
    model = ScreeningModel(self_check=name != "2027.02",
                           honours=name == "2027.04" and honours)
    terms = RISK_TERMS_2027_04 if name == "2027.04" else RISK_TERMS_2027_02
    return RiskPartner(release=name, lookups=records, terms=terms, model=model,
                       meters=name == "2027.04" and priced)


@dataclass
class Screening:
    """Claims' screening seam alone, against one Risk release, on its own graph."""

    risk: RiskPartner
    priced: bool = False  # Claims sends a ceiling and keeps a cost ledger
    budgets: Mapping[str, CostBudget] | None = None
    ledger: CostLedger = field(default_factory=CostLedger)
    pages: list[Page] = field(default_factory=list)
    records: list[CrossingRecord] = field(default_factory=list)
    answers: dict[str, float] = field(default_factory=dict)

    async def graph(self) -> ClaimGraph:
        apps = {SCREENING: A2APartners(risk=self.risk).apps()[SCREENING]}
        connections = await connect_partners(apps)
        nodes = a2a_workers(connections, self.records.append, {SCREENING:
                                                                SEAMS[SCREENING]})
        self.node: RemoteAgentNode = nodes[SCREENING]
        if self.priced:
            costed([self.node], self.ledger,
                   paging(OWNERSHIP, self.pages, seamkit.ledger.CLOCK),
                   budgets=self.budgets)
        return build_graph(nodes, InMemorySaver())

    async def canaries(self) -> str:
        """Chapter 15's screening canaries, run once: three goldens, judged."""
        await self.graph()
        suite = CanarySuite(SCREENING, CANARY, SCREENING_GOLDENS, SCREENING_BASELINE,
                            self.node, {"session": CANARY_SESSION}, lambda f: None)
        return (await suite.go(seamkit.ledger.CLOCK())).kind

    async def screen(self, sent: Sequence[dict[str, Any]], tag: str = "t") -> None:
        graph = await self.graph()
        for n, claim in enumerate(sent):
            out = await graph.ainvoke(cast(ClaimState, dict(claim)),
                                      thread(f"{tag}-{n}"))
            self.answers[str(claim["claim_id"])] = float(out["fraud_score"])

    def spent(self) -> int:
        model = self.risk.model
        assert model is not None
        return sum(model.spent.values())

    def calls(self) -> int:
        model = self.risk.model
        assert model is not None
        return sum(model.calls.values())


# The incident --------------------------------------------------------------------------

async def test_the_incident_the_same_answers_at_three_times_the_tokens() -> None:
    before, after = Screening(release("2027.02")), Screening(release("2027.03"))
    for screening in (before, after):
        await screening.screen(claims())
        assert await screening.canaries() == "in band"  # Chapter 15's canaries pass
    assert after.answers == before.answers == {"CLM-1001": 40.0, "CLM-1002": 82.0}
    assert after.calls() == 2 * before.calls()  # model calls per claim double
    assert round(after.spent() / before.spent(), 1) == 3.0  # tokens per claim triple
    budget = burn(SCREENING, SLO, after.records, seamkit.ledger.CLOCK())
    assert (budget.sends, budget.bad) == (5, 0)  # inside 5 s: the SLO spends nothing
    # Nothing on Claims' side carries a number: no ceiling sent, no report back.
    assert after.ledger.entries() == [] and after.risk.meter is None
    model = after.risk.model
    assert model is not None
    chargeback = sum(model.statement().values())  # Risk's own record, at month end
    assert chargeback == pytest.approx(3 * RATES.price(before.spent()), rel=0.05)


# A ceiling travels with the task -------------------------------------------------------

async def test_fixed_the_ceiling_travels_and_risk_signs_what_it_spent() -> None:
    screening = Screening(release("2027.04"), priced=True)
    assert await screening.canaries() == "in band"
    await screening.screen(claims())
    assert screening.answers == {"CLM-1001": 40.0, "CLM-1002": 82.0}
    assert screening.calls() == 5  # the self-check skipped on every one at 1.0
    meter = screening.risk.meter
    assert meter is not None
    assert {c.per_task for c in meter.ceilings.values()} == {1.0}  # it arrived
    assert set(meter.notes.values()) == {"self-check skipped: ceiling"}
    entries = screening.ledger.entries([SCREENING])
    assert [e.purpose for e in entries] == ["canary"] * 3 + ["claim"] * 2
    assert all(e.usage and e.price is not None and e.price <= 1.0 for e in entries)
    model = screening.risk.model
    assert model is not None
    assert sum(e.price or 0.0 for e in entries) == pytest.approx(
        sum(model.statement().values()))  # the ledger agrees with Risk's chargeback
    assert screening.pages == []  # nothing over its ceiling, no line near its alert


async def test_a_higher_ceiling_buys_the_self_check() -> None:
    screening = Screening(release("2027.04"), priced=True,
                          budgets={SCREENING: CostBudget(3.0, 28 * 86_400,
                                                         {"claim": 1_400.0,
                                                          "canary": 1_500.0})})
    await screening.screen(claims())  # the two sample claims
    assert screening.calls() == 4  # two passes each: the caller chose to pay for it
    assert all(1.0 < (e.price or 0.0) <= 3.0
               for e in screening.ledger.entries([SCREENING]))


async def test_a_usage_report_is_signed_with_the_partners_key() -> None:
    screening = Screening(release("2027.04"), priced=True)
    await screening.screen(claims()[1:])
    entry = screening.ledger.entries([SCREENING])[0]
    with seamkit_run():
        task = await screening.node.read(entry.task_id)
    report = usage_of(task)
    assert report is not None and report.verifies(KEYS["risk"].public_key())
    assert report.digest == entry.usage  # the ledger keeps what Risk signed
    forged = UsageReport(**{**report.__dict__, "cost": 0.1})
    assert not forged.verifies(KEYS["risk"].public_key())


def seamkit_run() -> Any:
    from tests.wire import as_run

    return as_run(thread("t-0"))


# When the partner ignores it -----------------------------------------------------------

async def test_a_partner_that_ignores_the_ceiling_is_signalled_then_refused() -> None:
    screening = Screening(release("2027.04", honours=False), priced=True,
                          budgets=shrunk(SCREENING, "claim", 6.0))
    sent = [claims()[1]] * 4  # CLM-1002, four times, on four threads
    with pytest.raises(BudgetSpent) as spent:
        await screening.screen(sent)
    assert screening.calls() == 6  # three screenings, both passes: nothing was read
    over = [p for p in screening.pages if "over the 1 ceiling" in p.signal.detail]
    assert {p.to for p in over} == {"claims-oncall", "risk-oncall"}  # both owners
    assert len(over) == 2 * 3  # every report was over
    assert "claim spent" in spent.value.detail
    assert len(screening.risk.executor.received) == 3  # the fourth never left Claims
    assert "BudgetSpent" in CLAIMS_REFUSED  # Claims' own refusal
    budget = burn(SCREENING, SLO, screening.records, seamkit.ledger.CLOCK())
    assert (budget.sends, budget.bad) == (3, 0)  # it spends no error budget


async def test_check_holds_each_seams_terms_to_its_budget() -> None:
    screening, estimate = COSTED[SCREENING], COSTED["repair_estimate"]
    [no_price] = check(RISK_TERMS_2027_02, screening, MERIDIAN_POLICY)
    assert no_price.detail == ("screen_fraud states no price and reports no usage, and "
                               "this seam budgets 1 per task")
    assert check(RISK_TERMS_2027_04, screening, MERIDIAN_POLICY) == []
    assert check(NORTHGATE_V7_5, estimate, MERIDIAN_POLICY) == []
    dearer = _priced(NORTHGATE_V7_5, 4.0)
    [breach] = check(dearer, estimate, MERIDIAN_POLICY)
    assert breach.detail.endswith("costs 4 per task, more than the 3.5 this seam "
                                  "budgets")


def _priced(terms: Any, price: float) -> Any:
    from dataclasses import replace

    skill = replace(terms.skills["estimate_repair"], cost_per_task=price)
    return replace(terms, revision="v7.6", skills={**terms.skills,
                                                   "estimate_repair": skill})


# Retries and loops, priced -------------------------------------------------------------

async def test_a_retry_to_risk_is_a_second_screening_read_again() -> None:
    risk = release("2027.04")
    screening = Screening(risk, priced=True)
    await screening.graph()
    sent = claims()[1]
    with seamkit_run():
        await screening.node.call_once(sent, key="k-retry")
        await screening.node.call_once(sent, key="k-retry")  # the same key again
    assert screening.calls() == 2 and len(risk.executor.received) == 2
    model = risk.model
    assert model is not None
    assert len(model.statement()) == 2  # Risk honours no key: two tasks, two charges


# The tax in tokens ---------------------------------------------------------------------

def test_the_tax_in_tokens_naive_send_against_the_shaped_send(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, dict[str, Any]] = {}
    handle = federated_naive.PartnerRuntime.handle

    def kept(self: federated_naive.PartnerRuntime, body: str,
             credential: str | None) -> str:
        sent.setdefault(self.owner, json.loads(body))  # what Chapter 1 sent each
        return handle(self, body, credential)

    monkeypatch.setattr(federated_naive.PartnerRuntime, "handle", kept)
    federated_naive.build_federated_graph().invoke(clean_claim(), thread("naive"))
    shaped = {k: sent["Risk"][k] for k in SEAMS[SCREENING].sends}  # Chapter 4's send
    assert (tokens(sent["Risk"]), tokens(shaped)) == (38, 19)  # per pass Risk reads
    classes = MERIDIAN_POLICY.classes

    def exposed(data: Mapping[str, Any]) -> set[str]:
        return {classes.get(k, "unclassified") for k in data}

    assert exposed(sent["Risk"]) - exposed(shaped) == {"vehicle", "unclassified"}
    northgate = {k: sent["Northgate"][k] for k in SEAMS["repair_estimate"].sends}
    assert exposed(sent["Northgate"]) - exposed(northgate) >= {
        "fraud_assessment", "claim_history"}  # across a company line


# The ledger and the inventory ----------------------------------------------------------

async def test_the_ledger_entry_carries_its_cost() -> None:
    screening = Screening(release("2027.04"), priced=True)
    graph = await screening.graph()
    await graph.ainvoke(cast(ClaimState, dict(claims()[1])), thread("t-led"))
    state = await graph.aget_state(thread("t-led"))
    entry = state.values["ledger"][SCREENING]
    [cost] = screening.ledger.entries([SCREENING])
    assert entry["cost"] == cost.digest and cost.idempotency_key == entry[
        "idempotency_key"]


async def test_the_inventory_names_each_seams_cost() -> None:
    screening = Screening(release("2027.04"), priced=True)
    await screening.screen(claims())
    inventory = SeamInventory([SeamEntry("claims_intake", SCREENING, "Claims", "Risk"),
                               SeamEntry("claims_intake", "repair_estimate", "Claims",
                                         "Northgate")])
    with_costs(inventory, screening.ledger, seamkit.ledger.CLOCK())
    risk, northgate = (e.answers["accountability"] for e in inventory.entries)
    assert risk.startswith("; cost: usage reported, ceiling 1, budget 2,950 per 28 "
                           "days, spent ")
    assert northgate == ("; cost: 3.5 per task, budget 2,688 per 28 days, spent 0.00")


def test_spend_counts_each_purpose_against_its_own_line() -> None:
    ledger = CostLedger()
    budget = COSTED["repair_estimate"].budget
    assert budget is not None
    from seamkit.costs import CostEntry

    for n, purpose in enumerate(["claim", "canary", "canary", "trial", "recording"]):
        ledger.append(CostEntry("repair_estimate", "Northgate", "t", f"task-{n}",
                                f"k-{n}", "estimate_repair", purpose, "v7.5", 3.5, "",
                                "", 0.0))
    lines = spend(ledger, "repair_estimate", budget, 0.0)
    assert {p: s.spent for p, s in lines.items()} == {
        "claim": 3.5, "trial": 3.5, "canary": 7.0, "recording": 3.5}
    assert str(lines["canary"]) == ("repair_estimate: canary spent 7.00 of 1,568.00, "
                                    "0% of its line")


def test_a_seam_error_is_what_budget_spent_is() -> None:
    assert issubclass(BudgetSpent, SeamError)
