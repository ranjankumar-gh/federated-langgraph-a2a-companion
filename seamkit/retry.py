"""The seam retry policy: which calls on a seam may be made again, and when.

A2A is at-least-once at the seam. A read may always be repeated. A send that opens a
task may be repeated only under a key the partner's terms promise to honour, and only
while that promise holds: the partner states a key window on its card
(SkillTerms.key_seconds), and the caller's retry deadline must end inside it, less the
clock skew both sides agree to budget for. Only a call that never reached an answer is
retried. A refusal, a revocation, a breach or a malformed answer is an answer, and
repeating it changes nothing.

The policy is per seam data, read from the seam's expectations file. `refuse_retry` is
the decision; `retrying` wraps one call and keeps the key it was first sent under.
"""

import asyncio
import hashlib
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

from seamkit.errors import PartnerUnavailable, SeamError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How often a call on one seam may be made, how far apart, and for how long."""

    attempts: int = 1  # every call, the first included; 1 retries nothing
    base: float = 1.0  # seconds: the backoff before the second attempt, at most
    cap: float = 60.0  # seconds: no backoff is longer than this
    deadline_seconds: float = 0.0  # from the first send: no resend after it


NO_RETRY = RetryPolicy()


def refuse_retry(policy: RetryPolicy, call: str, error: SeamError, attempt: int,
                 key_seconds: int | None, skew_seconds: int, first_sent: float,
                 now: float, held: str = "") -> str:
    """Why this call may not be made again now, or "" if it may.

    `held` is why the seam's breaker refuses new work now (Chapter 14), or "".
    """
    if not isinstance(error, PartnerUnavailable):  # 1. an answer: never repeat it
        return f"{type(error).__name__} is an answer"
    if held:  # Chapter 14: the seam's breaker is open, so nothing is made again
        return f"the seam's breaker holds: {held}"
    if attempt >= policy.attempts:  # 2. the attempts this seam allows, all spent
        return f"{policy.attempts} attempts made"
    if call != "send":  # 3. a read changes nothing: retry it
        return ""
    if key_seconds is None:  # 4. a key nobody honours is a label
        return "the partner's terms state no key window"
    window = min(policy.deadline_seconds, key_seconds - skew_seconds)
    if now - first_sent >= window:  # 5. past the deadline, or the key's window
        return f"{now - first_sent:.0f} s since the first send, past {window:.0f} s"
    return ""


def backoff(policy: RetryPolicy, attempt: int, error: SeamError,
            draw: Callable[[float, float], float] = random.uniform) -> float:
    """Full jitter, never sooner than a Retry-After the partner sent."""
    jittered = draw(0.0, min(policy.cap, policy.base * 2 ** (attempt - 1)))
    return max(jittered, retry_after(error) or 0.0)


def retry_after(error: BaseException) -> float | None:
    """The seconds a partner's Retry-After header asks for, if a response carried one."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, httpx.HTTPStatusError):
            raw = cause.response.headers.get("retry-after", "").strip()
            if raw.isdigit():
                return float(raw)
            try:  # the header may also be an HTTP date
                return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
            except (TypeError, ValueError):
                return None
        cause = cause.__cause__
    return None


async def retrying(call: Callable[[], Awaitable[T]], kind: str, policy: RetryPolicy,
                   key_seconds: int | None, skew_seconds: int, first_sent: float,
                   clock: Callable[[], float], made: int = 0,
                   sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                   held: Callable[[], str] = lambda: "") -> T:
    """Make `call` until it answers or the policy refuses; the last error is raised.

    `made` counts attempts already made and lost, such as the send an unknown entry
    records. `call` builds the same request every time, under the key it was first
    sent with: this wrapper never mints a key, so a retry never becomes a second task.
    """
    last: SeamError = PartnerUnavailable("", "an attempt made before this call")
    attempt = made
    while True:
        if attempt:  # an attempt was lost: wait before the next one
            await sleep(backoff(policy, attempt, last))
        try:
            return await call()
        except SeamError as exc:
            attempt, last = attempt + 1, exc
            if refuse_retry(policy, kind, exc, attempt, key_seconds, skew_seconds,
                            first_sent, clock(), held()):
                raise


def onward_key(inbound: str, seam: str) -> str:
    """A hop's key, derived from the key of the message it is answering."""
    return hashlib.sha256(f"{inbound}\n{seam}".encode()).hexdigest()[:32]
