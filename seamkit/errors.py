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
    """The task completed, but its artifact lacks a field the seam maps into state."""
