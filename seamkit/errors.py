"""Typed failures at a seam.

A partner's missing, unfinished, or refused answer raises one of these. It never
reaches the caller's state as a missing key read through a default, the habit that
Chapter 1's naive seam showed paying a suspect claim.
"""


class SeamError(Exception):
    """A seam did not produce an answer the caller may write into its state."""

    def __init__(self, seam: str, detail: str, task_id: str = "") -> None:
        super().__init__(f"{seam}: {detail}")
        self.seam = seam
        self.detail = detail
        self.task_id = task_id


class PartnerUnavailable(SeamError):
    """The partner could not be reached, or did not answer inside the seam's timeout."""


class TaskUnfinished(SeamError):
    """The partner returned a task still submitted or working: no answer yet."""


class PartnerNeedsInput(SeamError):
    """The partner paused for input this seam has no way to supply."""


class PartnerNeedsAuth(SeamError):
    """The partner paused for a credential; nothing in a LangGraph run answers that."""


class PartnerFailed(SeamError):
    """The partner's task ended failed, or was canceled on the partner's side."""


class PartnerRejected(SeamError):
    """The partner refused the task."""


class MalformedAnswer(SeamError):
    """The task completed, but its artifact lacks a field the seam maps into state,
    or carries one in a shape the seam's contract does not allow."""


class ContractBreach(SeamError):
    """The partner's terms in force break what this seam depends on.

    Found at connect, raised by the seam's node before anything is sent. Only the
    breached seam stops; every other seam keeps running.
    """


class CardUntrusted(SeamError):
    """The card trust policy does not let this seam use the partner's card now.

    The card is not the expected owner's, the skill is not approved on it, the skill is
    suspended under it, or an answer is over what an unproven skill may decide. Raised
    by the seam's node before anything is sent, except for the ceiling, which is read
    off the answer. Only this seam stops.
    """


class StaleLedgerWrite(SeamError):
    """A ledger write was computed from an entry another writer has since replaced."""


class DelegationRefused(SeamError):
    """A delegation chain was missing, could not be minted, or failed verification.

    Raised on the caller's side when a link cannot be minted (the run has no chain, the
    partner's card fails the trust hook, or the link would be wider than its parent),
    and inside a partner's guard when an inbound chain fails a check. The guard answers
    the caller with an authorization error, which the caller sees as PartnerRejected.
    """


class DelegationRevoked(DelegationRefused):
    """The chain a task holds was revoked: the task ends, and nothing renews it.

    Raised by Claims' resume gate when the chain a task was sent under is on the
    issuer's list (after it cancels the task), by a partner's guard for any call but
    the task's own cancel, and before a link is minted under a revoked chain.
    """
