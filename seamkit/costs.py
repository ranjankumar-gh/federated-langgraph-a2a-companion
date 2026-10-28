"""Who pays: the budget ceiling, the usage report, and the seam cost ledger.

A partner's spend is its own. Its model, its prompts and its onward calls are on its
side of the seam, and a release can change what a task costs without changing one
answer. So cost crosses the seam as terms. The caller sends a ceiling with every task,
in the message's metadata under this book's budget extension URI: the most it agrees
to pay for that task. A partner whose terms promise to honour the ceiling reads it
before it spends, and a partner whose terms promise usage reports signs one into the
task's metadata, under the same URI, with the key behind its card. A partner whose
skill has a fixed price (`cost_per_task`) needs neither: the terms are the price.

The caller keeps the seam cost ledger: one entry per charge a seam causes, with the
task, its key, its purpose, what priced it and the receipt it is checked against. It
is append-only and kept beside the receipt log, apart from any checkpoint. A cost
budget per seam and per purpose is summed over a window on the shared clock; past its
alert share it raises a `cost` signal, and a spent line refuses sends with
`BudgetSpent`, Claims' own refusal, which spends no error budget (Chapter 17).

The invoice check reads a partner's statement against the ledger and the receipt log,
joined by task id and idempotency key. The A2A specification says nothing about cost,
budgets or billing; every format here is the book's own.
"""

import base64
import hashlib
import json
import math
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import rfc8785
from a2a.types import Message, Task
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from google.protobuf.json_format import MessageToDict

from seamkit.errors import SeamError

BUDGET_URI = "urn:seamkit:seam-budget"  # the ceiling out, the usage report back
PURPOSES = ("claim", "trial", "canary", "recording", "compensation", "retry")


def tokens(payload: Any) -> int:
    """A stand-in token counter: one token per four characters of canonical JSON.

    The companion's models are scripted, so nothing reports real counts. A deployed
    partner reads the counts its model provider returns with every call.
    """
    return math.ceil(len(rfc8785.dumps(payload)) / 4)


@dataclass(frozen=True)
class Ceiling:
    """The most the caller agrees to pay for one task, and what the task is for."""

    per_task: float
    purpose: str = "claim"

    def plain(self) -> dict[str, Any]:
        return {"per_task": self.per_task, "purpose": self.purpose}


def ceiling_of(message: Message) -> Ceiling | None:
    """The ceiling a caller's message carries; None if it sent none."""
    raw = MessageToDict(message.metadata).get(BUDGET_URI) if message.metadata else None
    if not isinstance(raw, dict) or "per_task" not in raw:
        return None
    return Ceiling(float(raw["per_task"]), str(raw.get("purpose", "claim")))


@dataclass(frozen=True)
class UsageReport:
    """The partner's signed word about what one task spent. Not a measurement."""

    skill: str
    task_id: str
    tokens: int  # input and output tokens of every model call the task made
    cost: float  # the tokens priced by the rate card both sides use
    ceiling: float | None  # the ceiling the partner received; None if none came
    note: str  # what the partner did about the ceiling, such as a skipped pass
    signer: str  # the key id the signature verifies under
    at: str  # when the partner cut it, ISO 8601 in UTC
    sig: str = ""

    @property
    def over(self) -> bool:
        return self.ceiling is not None and self.cost > self.ceiling

    def payload(self) -> bytes:
        return rfc8785.dumps({k: v for k, v in asdict(self).items() if k != "sig"})

    def signed(self, key: Ed25519PrivateKey) -> "UsageReport":
        sig = base64.urlsafe_b64encode(key.sign(self.payload())).decode()
        return UsageReport(**{**asdict(self), "sig": sig})

    def verifies(self, key: Ed25519PublicKey) -> bool:
        try:
            key.verify(base64.urlsafe_b64decode(self.sig), self.payload())
        except (InvalidSignature, ValueError):
            return False
        return True

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(rfc8785.dumps(asdict(self))).hexdigest()


def usage_of(task: Task) -> UsageReport | None:
    """The partner's usage report, as it came back in the task's metadata."""
    raw = MessageToDict(task.metadata).get(BUDGET_URI) if task.metadata else None
    if not isinstance(raw, dict):
        return None
    try:
        return UsageReport(**{**raw, "tokens": int(raw["tokens"])})
    except (TypeError, KeyError):
        return None


@dataclass(frozen=True)
class RateCard:
    """Meridian's internal price for model tokens, owned by Finance (Chapter 18)."""

    owner: str
    per_thousand: float  # claim currency per thousand tokens; a design value

    def price(self, count: int) -> float:
        return round(count * self.per_thousand / 1000, 4)


def load_rates(path: Path) -> RateCard:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return RateCard(str(raw["owner"]), float(raw["per_thousand"]))


@dataclass
class Meter:
    """The partner's half: what each task spent, signed into the task as it leaves.

    Its model calls `spend` with each call's tokens. `stamp` returns a copy of the task
    with the signed report in its metadata, as Chapter 17's countersigner returns its
    half; the stored task is never changed.
    """

    key: Ed25519PrivateKey
    kid: str
    rates: RateCard
    now: Callable[[], float] = time.time
    spent: dict[str, int] = field(default_factory=dict)  # task -> tokens so far
    notes: dict[str, str] = field(default_factory=dict)
    ceilings: dict[str, Ceiling] = field(default_factory=dict)  # task -> as received
    skills: dict[str, str] = field(default_factory=dict)

    def received(self, task_id: str, message: Message) -> None:
        """What the caller's message asked of this task, kept for its report."""
        if (came := ceiling_of(message)) is not None:
            self.ceilings[task_id] = came
        self.skills[task_id] = (str(message.metadata["skill"])
                                if "skill" in message.metadata else "")

    def spend(self, task_id: str, count: int) -> None:
        self.spent[task_id] = self.spent.get(task_id, 0) + count

    def stamp(self, task: Task) -> Task:
        if task.id not in self.spent:
            return task
        count, came = self.spent[task.id], self.ceilings.get(task.id)
        at = datetime.fromtimestamp(self.now(), UTC).isoformat(timespec="seconds")
        report = UsageReport(self.skills.get(task.id, ""), task.id, count,
                             self.rates.price(count),
                             came.per_task if came else None,
                             self.notes.get(task.id, ""), self.kid, at)
        out = Task()
        out.CopyFrom(task)
        out.metadata.update({BUDGET_URI: asdict(report.signed(self.key))})
        return out


@dataclass(frozen=True)
class CostEntry:
    """One charge a seam caused, as the caller recorded it when the task answered."""

    seam: str
    partner: str
    thread: str
    task_id: str
    idempotency_key: str
    skill: str
    purpose: str  # one of PURPOSES
    terms: str  # the terms revision in force when the task was sent
    price: float | None  # None: nothing priced it, and the seam's cost is unknown
    usage: str  # the digest of the partner's signed usage report, or ""
    receipt: str  # the digest of the receipt the charge is checked against
    at: float  # on the shared clock

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(rfc8785.dumps(asdict(self))).hexdigest()


@dataclass
class CostLedger:
    """Append-only, one list per seam, kept beside the receipt log."""

    directory: Path | None = None
    kept: dict[str, list[CostEntry]] = field(default_factory=dict)

    def append(self, entry: CostEntry) -> CostEntry:
        self.kept.setdefault(entry.seam, []).append(entry)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{entry.seam}.costs.jsonl").open("a") as fh:
                fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def entries(self, seams: Sequence[str] = ()) -> list[CostEntry]:
        return [e for s in (seams or sorted(self.kept)) for e in self.kept.get(s, [])]


@dataclass(frozen=True)
class CostBudget:
    """One seam's cost budget, from its expectations file's [budget] table."""

    per_task: float  # the ceiling sent with every task
    window_seconds: float  # summed over this window, on the shared clock
    lines: Mapping[str, float]  # purpose -> what the window may spend on it
    alert: float = 0.8  # the share of a line spent that raises a cost signal


def budget_of(raw: Mapping[str, Any]) -> CostBudget:
    lines = {str(k): float(v) for k, v in raw.get("lines", {}).items()}
    if unknown := sorted(set(lines) - set(PURPOSES)):
        raise ValueError(f"budget lines {unknown} are not purposes {list(PURPOSES)}")
    return CostBudget(float(raw["per_task"]), float(raw["window_days"]) * 86_400,
                      lines, float(raw.get("alert", 0.8)))


@dataclass(frozen=True)
class Spend:
    """What one purpose has spent on one seam in the window, against its line."""

    seam: str
    purpose: str
    spent: float
    line: float

    @property
    def share(self) -> float:
        return self.spent / self.line if self.line else float(self.spent > 0)

    def __str__(self) -> str:
        return (f"{self.seam}: {self.purpose} spent {self.spent:,.2f} of "
                f"{self.line:,.2f}, {self.share:.0%} of its line")


def spend(ledger: CostLedger, seam: str, budget: CostBudget,
          now: float) -> dict[str, Spend]:
    """Each purpose's spend on this seam over the window, priced entries only."""
    held = [e for e in ledger.entries([seam])
            if now - e.at <= budget.window_seconds and e.price is not None]
    return {p: Spend(seam, p, round(sum(e.price or 0.0 for e in held
                                        if e.purpose == p), 4),
                     budget.lines.get(p, 0.0))
            for p in PURPOSES if p in budget.lines or any(e.purpose == p for e in held)}


class BudgetSpent(SeamError):
    """The seam's cost budget for this purpose is spent: Claims stops paying.

    Raised before anything is sent. It is Claims' decision, not the partner's breach,
    and like a breaker's refusal it spends no error budget.
    """


@dataclass
class Costs:
    """What a node needs to price its tasks: the seam's budget, the ledger, the rate
    card, the partner keys Claims pinned, and who hears a cost signal."""

    budget: CostBudget
    ledger: CostLedger
    rates: RateCard
    pinned: Mapping[str, Ed25519PublicKey]
    reported: bool = False  # the terms promise a usage report on every task
    signal: Callable[[str, str], None] = field(default=lambda seam, detail: None)

    def ceiling(self, purpose: str) -> Ceiling:
        return Ceiling(self.budget.per_task, purpose)

    def refuses(self, seam: str, purpose: str, now: float) -> str:
        """Why this purpose may not open another task on the seam now, or ""."""
        line = spend(self.ledger, seam, self.budget, now).get(purpose)
        if line is None or line.line == 0.0 or line.spent < line.line:
            return ""
        return str(line)

    def priced(self, seam: str, task: Task, price: float | None) -> tuple[
            float | None, str]:
        """The task's price and the digest of the report behind it, if any.

        A fixed price from the terms wins. Without one, the partner's report prices
        the task, by the rate card both sides use, and only if Claims can verify it.
        """
        report = usage_of(task)
        if report is not None:
            key = self.pinned.get(report.signer)
            if key is None or not report.verifies(key):
                self.signal(seam, f"task {task.id}: a usage report Claims cannot "
                                  f"verify ({report.signer})")
                report = None
        if report is None:
            if price is None and self.reported:
                self.signal(seam, f"task {task.id}: no usage report, and the terms "
                                  "promise one")
            return price, ""
        if report.over:
            self.signal(seam, f"task {task.id}: spent {report.cost:g}, over the "
                              f"{report.ceiling:g} ceiling")
        if price is None:
            price = self.rates.price(report.tokens)
        return price, report.digest

    def after(self, seam: str, now: float) -> None:
        """A line past its alert share is a cost signal."""
        for line in spend(self.ledger, seam, self.budget, now).values():
            if line.line and line.share >= self.budget.alert:
                self.signal(seam, str(line))


def purpose_of(data: Mapping[str, Any], resend: bool, trial: bool,
               compensates: bool) -> str:
    """What a task is for, from what the caller already knows about it.

    Canary and recording claims carry reserved ids (Chapters 15 and 16).
    """
    claim = str(data.get("claim_id", ""))
    if claim.startswith("CLM-C"):
        return "canary"
    if claim.startswith("CLM-R"):
        return "recording"
    if compensates:
        return "compensation"
    if trial:
        return "trial"
    return "retry" if resend else "claim"


# The invoice check ---------------------------------------------------------------------

@dataclass(frozen=True)
class Charge:
    """One line of a partner's statement: what it billed for one task."""

    task_id: str
    idempotency_key: str
    skill: str
    price: float


@dataclass(frozen=True)
class InvoiceFinding:
    """A charge the caller will not pay as billed, with what proves it."""

    kind: str  # "no receipt", "price", "twice", "canceled"
    charge: Charge
    detail: str


def check_invoice(lines: Iterable[Charge], ledger: Sequence[CostEntry],
                  receipts: Iterable[Any],
                  prices: Mapping[str, Mapping[str, float | None]],
                  ) -> list[InvoiceFinding]:
    """Every charge on the statement, matched to a receipt by task id and key.

    `prices` is revision -> skill -> the price its terms state: a task is priced by
    the revision on its receipt, never by today's card. A receipt with no charge is
    not a finding: the partner chose not to bill it.
    """
    by_task = {r.ours.task_id: r for r in receipts}
    costed = {e.task_id: e for e in ledger}
    found: list[InvoiceFinding] = []
    billed: dict[str, str] = {}  # key -> the first task billed under it
    for line in lines:
        receipt = by_task.get(line.task_id)
        if receipt is None or receipt.idempotency_key != line.idempotency_key:
            found.append(InvoiceFinding("no receipt", line,
                                        "no receipt names this task under this key"))
            continue
        if receipt.ours.state == "canceled":
            found.append(InvoiceFinding("canceled", line,
                                        "the task ended canceled, never answered"))
            continue
        if (first := billed.get(line.idempotency_key)) is not None:
            found.append(InvoiceFinding(
                "twice", line, f"key already billed for task {first}: one task, "
                               "one charge"))
            continue
        billed[line.idempotency_key] = line.task_id
        agreed = prices.get(receipt.terms, {}).get(line.skill)
        if agreed is not None and line.price != agreed:
            found.append(InvoiceFinding(
                "price", line, f"billed {line.price:g}; terms {receipt.terms} say "
                               f"{agreed:g}"))
        elif line.task_id not in costed:
            found.append(InvoiceFinding("no receipt", line,
                                        "the ledger holds no charge for this task"))
    return found


def credit(lines: Iterable[Charge], spent: float, share: float) -> float:
    """The service credit a breached SLO earns, as a clause of the agreement states it:
    `share` of the window's charges once the error budget is spent (spent >= 1)."""
    return round(share * sum(c.price for c in lines), 2) if spent >= 1.0 else 0.0


__all__ = ["BUDGET_URI", "PURPOSES", "BudgetSpent", "Ceiling", "Charge", "CostBudget",
           "CostEntry", "CostLedger", "Costs", "InvoiceFinding", "Meter", "RateCard",
           "Spend", "UsageReport", "budget_of", "ceiling_of", "check_invoice",
           "credit", "load_rates", "purpose_of", "spend", "tokens", "usage_of"]
