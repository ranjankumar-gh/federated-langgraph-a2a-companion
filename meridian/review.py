"""Claims' estimate review: the first model call in the Claims graph.

A person used to read every shop note before a claim was paid. The review is split in
two, so that the notes, which are partner text, reach only a model that cannot act:

- read_notes, the quarantined reader: no tools, sees only the notes and the claim's part
  list, and must answer in NoteFindings, a closed schema checked against PART_PRICES;
- review_estimate, the acting node: decides the route from typed values and the
  findings, never sees the notes, and has memo_of write the memo from a template.

A Reader is a scripted model, as Risk's Assistant is: a callable the tests write, so no
model key is needed and every run decides the same way.
"""

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from meridian.agents import PART_PRICES
from meridian.inprocess import ClaimState, Node
from seamkit.errors import MalformedAnswer
from seamkit.ingress import derive

REVIEW = "review_estimate"

# A model given the notes and the claim's part list, and nothing else.
Reader = Callable[[str, list[str]], Mapping[str, Any]]

MEMO_FIELDS = ("claim_id", "amount", "repair_estimate")  # all the memo is made of


class NoteFindings(BaseModel):
    """Everything the reader may say about a note: three closed values."""

    model_config = ConfigDict(extra="forbid", strict=True)

    supplement: bool  # the shop expects to bill more than this estimate
    parts: list[str]  # parts the note says were replaced, from the claim's own list
    needs_adjuster: bool


def read_notes(reader: Reader, notes: str, parts: list[str]) -> NoteFindings:
    """The quarantined reader: a note and a part list in, three closed values out."""
    raw = reader(notes, list(parts))  # no tools, no state, nothing it can send
    try:
        found = NoteFindings.model_validate(raw)
    except ValidationError as exc:
        raise MalformedAnswer(REVIEW, "the reader answered outside NoteFindings") from exc
    if not set(found.parts) <= set(parts) & set(PART_PRICES):
        raise MalformedAnswer(REVIEW, "the reader named a part the claim lacks")
    return found


NO_FINDINGS = NoteFindings(supplement=False, parts=[], needs_adjuster=False)


def memo_of(state: ClaimState) -> str:
    """The remittance memo, from typed values only: no model writes it."""
    return (f"Claim {state['claim_id']}: repair estimate "
            f"{state['repair_estimate']:,.0f} against a claim of {state['amount']:,.0f}.")


def reviewer(reader: Reader) -> Node:
    """The review node: the reader reads the note; this node decides, never reading it."""

    def review_estimate(state: ClaimState) -> ClaimState:
        notes = state.get("estimate_notes", "")
        parts = list(state["damaged_parts"])
        found = read_notes(reader, notes, parts) if notes else NO_FINDINGS
        tags = state.get("provenance", {})
        update: ClaimState = {
            "note_findings": found.model_dump(), "payment_memo": memo_of(state),
            "provenance": {
                "note_findings": derive(tags, "estimate_notes", "damaged_parts",
                                        closed=True),
                "payment_memo": derive(tags, *MEMO_FIELDS)},
            "completed": [REVIEW]}
        if found.needs_adjuster or found.supplement:
            update["decision"] = "refer"  # a person reads the note, as before
        return update

    return review_estimate


def keyword_reader(notes: str, parts: list[str]) -> Mapping[str, Any]:
    """A stand-in for a model where a test does not choose one: it finds part names and
    the word "supplement", and nothing in the note can make it say anything else."""
    text = notes.lower()
    return {"supplement": "supplement" in text,
            "parts": [p for p in parts if p in text], "needs_adjuster": False}
