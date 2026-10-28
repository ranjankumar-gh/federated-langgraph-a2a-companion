"""The delegation chain: authority that crosses seams as signed links, each narrower.

A chain starts at a root link, minted by the tenant's issuer when it exchanges a
person's session (the token-exchange step), and grows by one link per hop. Each link is
minted and signed by the owner the link above it was addressed to, names the one owner
that may accept it, and grants no more than its parent: the same person, a subset of
claims, skills and fields, fewer hops left, and an expiry no later. That rule is
`narrows`, and it is checked twice: by the minter, which refuses to widen, and by the
receiver, which checks every link of the chain, not only the last.

The outbound half is `DelegationAuth`, an httpx auth hook on each partner connection:
it reads the run's chain, mints the seam's link and sends the whole chain in one
header. The inbound half is `Guard`, an ASGI wrapper in front of a partner's A2A app: it
verifies the chain against the issuers it pinned, binds it to the workload that
presented it, checks the request against the last link's scope, and refuses with an
authorization error before the A2A handler sees anything.

Since Chapter 13 a link may also name tasks: a compensation's link names the one task it
undoes, and a partner's guard accepts it for that task only, and only from the workload
that sent the task. A link that names no task is not limited by task, as before.

The encoding is this book's own (canonical JSON, Ed25519 signatures per owner), chosen
so that the attenuation rule is visible in a few lines. It is not a standard format.
"""

import base64
import binascii
import hashlib
import json
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from typing import Any

import httpx
from a2a.types import AgentCard
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from langchain_core.runnables import ensure_config
from langchain_core.runnables.config import var_child_runnable_config
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seamkit.errors import DelegationRefused, DelegationRevoked

HEADER = "a2a-delegation"  # the whole chain, encoded; never logged, never checkpointed


@dataclass(frozen=True)
class Link:
    """One hop of authority: who granted it, to whom, for whom, for what, how far."""

    issuer: str  # the owner that minted and signed this link
    audience: str  # the one owner that may accept it, and extend it
    subject: str  # the person every link in the chain acts for
    claims: frozenset[str]  # the purpose: the claims a request under it may name
    skills: frozenset[str]  # skills the audience may run, or pass on
    fields: frozenset[str]  # request fields any hop under it may send
    hops: int  # links still allowed below this one
    expires: float  # epoch seconds, never later than the parent's
    parent: str = ""  # digest of the link above; empty at the root
    sig: str = ""  # the issuer's Ed25519 signature over every field above
    # Chapter 13: the only tasks a request under this link may name; empty is any task
    tasks: frozenset[str] = frozenset()


Chain = tuple[Link, ...]


@dataclass(frozen=True)
class Grant:
    """What one seam's link grants: skills, fields, and the hops left below it."""

    skills: frozenset[str]
    fields: frozenset[str]
    hops: int
    tasks: frozenset[str] = frozenset()  # Chapter 13: a compensation's one task


# Encoding, digests and signatures -----------------------------------------------------

def _body(link: Link) -> dict[str, Any]:
    data = asdict(link)
    for key in ("claims", "skills", "fields"):
        data[key] = sorted(data[key])
    tasks = data.pop("tasks")
    if tasks:  # absent when empty, so every link minted before Chapter 13 hashes the same
        data["tasks"] = sorted(tasks)
    return data


def _payload(link: Link) -> bytes:
    unsigned = {k: v for k, v in _body(link).items() if k != "sig"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def digest(link: Link) -> str:
    whole = json.dumps(_body(link), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(whole).hexdigest()


def chain_id(chain: Chain) -> str:
    """The chain's name in logs and ledgers: its root's digest, the same at each hop."""
    return "chain:" + digest(chain[0]).removeprefix("sha256:")[:16]


def sign(link: Link, key: Ed25519PrivateKey) -> Link:
    signature = key.sign(_payload(link))
    return replace(link, sig=base64.urlsafe_b64encode(signature).decode())


def signed_by(link: Link, key: Ed25519PublicKey) -> bool:
    try:
        key.verify(base64.urlsafe_b64decode(link.sig), _payload(link))
    except (InvalidSignature, ValueError, binascii.Error):
        return False
    return True


def encode(chain: Chain) -> str:
    raw = json.dumps([_body(link) for link in chain], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode(header: str) -> Chain:
    if not header:
        return ()
    try:
        items = json.loads(base64.urlsafe_b64decode(header))
        return tuple(Link(**{**item, "claims": frozenset(item["claims"]),
                             "skills": frozenset(item["skills"]),
                             "fields": frozenset(item["fields"]),
                             "tasks": frozenset(item.get("tasks", []))})
                     for item in items)
    except (ValueError, TypeError, KeyError, binascii.Error) as exc:
        raise DelegationRefused("", "the delegation header does not decode") from exc


# The attenuation rule -----------------------------------------------------------------

def narrows(parent: Link, child: Link) -> list[str]:
    """The attenuation rule: every way `child` is wider than `parent`, by name."""
    checks = {
        "minted by someone the parent was not addressed to":
            child.issuer == parent.audience,
        "acts for another person": child.subject == parent.subject,
        "widens the claims": child.claims <= parent.claims,
        "widens the skills": child.skills <= parent.skills,
        "widens the fields": child.fields <= parent.fields,
        "does not spend a hop": 0 <= child.hops < parent.hops,
        "outlives its parent": child.expires <= parent.expires,
        "does not name its parent": child.parent == digest(parent),
        "widens the tasks": not parent.tasks or (
            bool(child.tasks) and child.tasks <= parent.tasks),
    }
    return [why for why, holds in checks.items() if not holds]


def mint(chain: Chain, key: Ed25519PrivateKey, issuer: str, audience: str,
         grant: Grant, claims: frozenset[str]) -> Link:
    """The next link under `chain`, signed by `issuer`; refused if it would widen."""
    above = chain[-1]
    child = Link(issuer, audience, above.subject, claims, grant.skills, grant.fields,
                 grant.hops, above.expires, parent=digest(above), tasks=grant.tasks)
    if wider := narrows(above, child):
        raise DelegationRefused(audience, "cannot mint: " + "; ".join(wider))
    return sign(child, key)


# Outbound: the minting middleware -----------------------------------------------------

# Which cards may receive authority at all: the card trust policy decides (Chapter 5),
# usually TrustDecision.covers for the card the seam connected to.
TrustHook = Callable[[AgentCard], bool]


def run_value(key: str) -> Any:
    """A value from the current LangGraph run's config, or None outside a run."""
    return ensure_config().get("configurable", {}).get(key)


@contextmanager
def run_values(**values: Any) -> Iterator[None]:
    """Inside a run, make a nested call see extra configurable values (Chapter 7)."""
    config = ensure_config()
    token = var_child_runnable_config.set(
        {**config, "configurable": {**config.get("configurable", {}), **values}})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def inbound_chain() -> Chain:
    """The chain a partner verified on the request that started this run."""
    chain: Chain | None = run_value("delegation")
    if not chain:
        raise DelegationRefused("", "this run carries no delegation chain")
    return chain


def claim_named(content: bytes) -> str:
    """The claim a JSON-RPC send names in its data part, or "" for a read or a reply."""
    try:
        message = json.loads(content).get("params", {}).get("message", {})
        data = (message.get("parts") or [{}])[0].get("data", {})
    except (ValueError, AttributeError):
        return ""
    return str(data.get("claim_id", ""))


class DelegationAuth(httpx.Auth):
    """Outbound: mint this seam's link under the run's chain, and send the chain."""

    requires_request_body = True

    def __init__(self, owner: str, key: Ed25519PrivateKey, audience: str, grant: Grant,
                 parent: Callable[[], Chain], card: AgentCard,
                 trusted: TrustHook) -> None:
        self.owner, self.key, self.audience, self.grant = owner, key, audience, grant
        self.parent, self.card = parent, card
        self.trusted = trusted  # no default: no policy, no authority

    def grant_now(self) -> Grant:
        """The grant for the call being made; a compensation narrows it (Chapter 13)."""
        return self.grant

    def auth_flow(self, request: httpx.Request,
                  ) -> Generator[httpx.Request, httpx.Response, None]:
        if request.method != "POST":  # the card is public; only calls carry authority
            yield request
            return
        if not self.trusted(self.card):  # which cards may receive authority at all
            raise DelegationRefused(self.audience, "card fails the trust policy")
        chain = self.parent()  # the run's chain: none is a refusal, never a fallback
        claim = claim_named(request.content)  # a read names none: keep the parent's
        claims = frozenset([claim]) if claim else chain[-1].claims
        link = mint(chain, self.key, self.owner, self.audience, self.grant_now(), claims)
        request.headers[HEADER] = encode((*chain, link))
        yield request


# Inbound: verification and the guard --------------------------------------------------

@dataclass(frozen=True)
class Trust:
    """One tenant's issuers, pinned out of band by the partner that verifies them."""

    tenant: str
    root: str  # the only issuer that may start a chain: the tenant's token exchange
    keys: Mapping[str, Ed25519PublicKey]  # every issuer allowed to sign a link


@dataclass(frozen=True)
class Verified:
    """A chain that passed every check: who it acts for, and whose data it may touch."""

    chain: Chain
    chain_id: str
    subject: str
    tenant: str  # from the pinned root, never from the request

    @property
    def scope(self) -> Link:
        return self.chain[-1]

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(link.issuer for link in self.chain)


def verify(chain: Chain, peer: str, me: str, trusts: Mapping[str, Trust],
           now: float) -> Verified:
    """Inbound checks, cheapest and identity first; any failure refuses the call."""
    if not chain:
        raise DelegationRefused(me, "no delegation chain")  # never a default
    last = chain[-1]
    if peer != last.issuer:  # caller binding: only the minter of the last link holds it
        raise DelegationRefused(me, f"presented by {peer or 'nobody'}, "
                                    f"held by {last.issuer}")
    if last.audience != me:
        raise DelegationRefused(me, f"addressed to {last.audience}")
    if me in {link.issuer for link in chain}:  # Chapter 12: a chain may not come back
        raise DelegationRefused(me, f"a loop: {me} already passed this chain on")
    trust = trusts.get(chain[0].issuer)
    if trust is None or chain[0].parent:
        raise DelegationRefused(me, f"root {chain[0].issuer} is not pinned")
    for link in chain:  # every signature, against keys pinned out of band
        key = trust.keys.get(link.issuer)
        if key is None or not signed_by(link, key):
            raise DelegationRefused(me, f"link from {link.issuer} is not signed")
    for parent, child in pairwise(chain):  # the whole chain narrows, not just the end
        if wider := narrows(parent, child):
            raise DelegationRefused(me, f"{child.issuer}: " + "; ".join(wider))
    if last.expires <= now:
        raise DelegationRefused(me, "chain expired")
    return Verified(chain, chain_id(chain), last.subject, trust.tenant)


def outside(scope: Link, skill: str, data: Mapping[str, Any], claim: str) -> list[str]:
    """Every way one request goes beyond the last link's scope."""
    problems = []
    if skill not in scope.skills:
        problems.append(f"skill {skill or 'none'} not granted")
    if extra := sorted(set(data) - scope.fields):
        problems.append(f"fields {extra} not granted")
    if claim not in scope.claims:
        problems.append(f"claim {claim or 'none'} not granted")
    if "task" in data and str(data["task"]) not in scope.tasks:  # Chapter 13
        problems.append(f"task {data['task']} not granted")
    return problems


# Why a verified chain may not be used now ("" if it may): Chapter 7's revocation check,
# built from a partner's revocation lists by seamkit.revocation.checker.
RevocationCheck = Callable[[Chain, float], str]


class TaskHidden(Exception):
    """A task this chain may not see: answered exactly as a task that does not exist."""


def as_peer(app: ASGIApp, workload: str) -> ASGIApp:
    """Stand-in for an mTLS terminator: the workload named by the client certificate.

    A real deployment terminates mutual TLS in front of the app and passes the verified
    certificate's identity in the ASGI TLS extension. The tests have no sockets, so this
    wrapper sets the same key. The guard reads only this, never a header a caller sets.
    """
    async def with_peer(scope: Scope, receive: Receive, send: Send) -> None:
        extensions = {**scope.get("extensions", {}),
                      "tls": {"client_cert_name": workload}}
        await app({**scope, "extensions": extensions}, receive, send)

    return with_peer


def peer_of(scope: Scope) -> str:
    return str(scope.get("extensions", {}).get("tls", {}).get("client_cert_name", ""))


async def _read_body(receive: Receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        chunk: bytes = message.get("body", b"")
        body += chunk
        if not message.get("more_body"):
            return body


def _replay(body: bytes, receive: Receive) -> Receive:
    sent = False

    async def again() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return again


async def _answer(send: Send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


@dataclass
class Guard:
    """Inbound: verify every JSON-RPC call's chain before the A2A handler sees it."""

    app: ASGIApp
    me: str
    trusts: Mapping[str, Trust]
    now: Callable[[], float] = time.time
    tasks: dict[str, tuple[str, str]] = field(default_factory=dict)  # (subject, claim)
    log: list[dict[str, Any]] = field(default_factory=list)  # the partner's own record
    revoked: RevocationCheck | None = None  # Chapter 7: re-checked on every call
    holders: dict[str, str] = field(default_factory=dict)  # task -> workload that sent it

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)  # the card stays public
            return
        body = await _read_body(receive)
        rpc = json.loads(body or b"{}")
        try:
            verified, claim = self.check(scope, rpc)
        except DelegationRefused as exc:
            self.log.append(self._entry(scope, rpc, exc.detail))
            await _answer(send, 403, {"jsonrpc": "2.0", "id": rpc.get("id"), "error": {
                "code": 403, "message": f"delegation refused: {exc.detail}"}})
            return
        except TaskHidden:  # the same answer as a task that was never there
            await _answer(send, 200, {"jsonrpc": "2.0", "id": rpc.get("id"), "error": {
                "code": -32001, "message": "Task not found"}})
            return
        self.log.append(self._entry(scope, rpc, "", verified))
        await self.app({**scope, "auth": verified}, _replay(body, receive),
                       self._remember(send, verified, claim))

    def check(self, scope: Scope, rpc: Mapping[str, Any]) -> tuple[Verified, str]:
        headers = dict(scope["headers"])
        chain = decode(headers.get(HEADER.encode(), b"").decode())
        verified = verify(chain, peer_of(scope), self.me, self.trusts, self.now())
        params = rpc.get("params", {})
        peer = peer_of(scope)
        if self.revoked and (why := self.revoked(verified.chain, self.now())):
            if self._cancels_own_task(scope, rpc, verified):  # the one exception
                return verified, self._owned(str(params.get("id", "")), verified, peer)
            raise DelegationRevoked(self.me, why)
        if rpc.get("method") != "SendMessage":  # a read, a cancel: bound to the task
            return verified, self._owned(str(params.get("id", "")), verified, peer)
        message = params.get("message", {})
        data = (message.get("parts") or [{}])[0].get("data", {})
        skill = str(message.get("metadata", {}).get("skill", ""))
        task_id = str(message.get("taskId", ""))
        claim = self._owned(task_id, verified, peer) if task_id else str(
            data.get("claim_id", ""))
        if problems := outside(verified.scope, skill, data, claim):
            raise DelegationRefused(self.me, "; ".join(problems))
        if undoes := str(data.get("task", "")):  # Chapter 13: only its sender undoes it
            _, bound = self.tasks.get(undoes, ("", ""))
            if self.holders.get(undoes) != peer or bound != claim:
                raise DelegationRefused(self.me, f"task {undoes} was not sent by {peer} "
                                                 f"for {claim}")
        return verified, claim

    def _cancels_own_task(self, scope: Scope, rpc: Mapping[str, Any],
                          verified: Verified) -> bool:
        """A cancel only reduces authority: allowed under a dead chain, narrowly.

        The task must be one this guard recorded, for the chain's person, and the call
        must come from the workload that sent it.
        """
        task_id = str(rpc.get("params", {}).get("id", ""))
        subject, _ = self.tasks.get(task_id, ("", ""))
        return (rpc.get("method") == "CancelTask" and subject == verified.subject
                and self.holders.get(task_id) == peer_of(scope))

    def _owned(self, task_id: str, verified: Verified, peer: str = "") -> str:
        """A task is readable only under a chain for its person and its claim.

        Since Chapter 13, also under a link that names the task, from the workload that
        sent it: a compensation reads and cancels the one task it undoes.
        """
        subject, claim = self.tasks.get(task_id, ("", ""))
        if not subject:
            return ""  # not ours to judge: the handler answers for unknown tasks
        named = task_id in verified.scope.tasks and claim in verified.scope.claims
        if named and self.holders.get(task_id) == peer:
            return claim
        if subject != verified.subject or claim not in verified.scope.claims:
            raise TaskHidden(task_id)
        return claim

    def _remember(self, send: Send, verified: Verified, claim: str) -> Send:
        chunks: list[bytes] = []

        async def recording(message: Message) -> None:
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if not message.get("more_body"):
                    self._record_task(b"".join(chunks), verified, claim)
            await send(message)

        return recording

    def _record_task(self, body: bytes, verified: Verified, claim: str) -> None:
        try:
            result = json.loads(body).get("result", {})
        except ValueError:
            return
        task = result.get("task", result) if isinstance(result, dict) else {}
        if isinstance(task, dict) and task.get("id") and claim:
            self.tasks.setdefault(str(task["id"]), (verified.subject, claim))
            self.holders.setdefault(str(task["id"]), verified.path[-1])

    def _entry(self, scope: Scope, rpc: Mapping[str, Any], refused: str,
               verified: Verified | None = None) -> dict[str, Any]:
        message = rpc.get("params", {}).get("message", {})
        chain = verified.chain if verified else _quiet_decode(scope)
        return {"method": rpc.get("method", ""), "peer": peer_of(scope),
                "skill": message.get("metadata", {}).get("skill", ""),
                "chain": chain_id(chain) if chain else "",
                "subject": chain[-1].subject if chain else "",
                "path": tuple(link.issuer for link in chain), "refused": refused,
                "at": self.now()}


def _quiet_decode(scope: Scope) -> Chain:
    """For the refusal log only: what the caller claimed, unverified."""
    try:
        return decode(dict(scope["headers"]).get(HEADER.encode(), b"").decode())
    except DelegationRefused:
        return ()
