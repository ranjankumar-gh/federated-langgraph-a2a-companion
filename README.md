# Federated AI Agents - Companion Code

Companion repository for *Federated AI Agents: LangGraph and A2A for Agents You Don't Own*. Builds
one running project, **the Meridian claims federation**, incrementally across the book's
chapters. The reusable pieces live in the `seamkit` package.

## Tags

Each chapter and appendix has its own git tag, matching the book's own chapter slugs
(e.g. `ch01-multi-agent-is-not-federated`), cut in reading order. Check out a tag to see the
codebase exactly as it stood at the end of that chapter:

```
git checkout <slug>
```

Chapter 2's tag is the one exception to reading order. Its cost test measures `policy_lookup`
behind the A2A wire, so the Chapter 2 tag already contains the wire Chapter 3 builds
(`seamkit/remote.py`, `seamkit/server.py`, the partner servers) and the task ledger Chapter 9
introduces. Chapter 2's text says so where it borrows them.

## Running the tests

```
uv sync
uv run pytest
uv run mypy
```

Some tests are skip-guarded behind environment variables when a chapter integrates with a real
external service (see that chapter's section below). They skip cleanly without one - they are not
required for the rest of the suite to pass.

## Chapters

### Chapter 1: Multi-Agent Is Not Federated

- `meridian/inprocess.py`: the four-agent claims supervisor in one deployment.
- `meridian/federated_naive.py`: the same graph with three workers behind owner lines
  (no network; each partner is a separately constructed runtime that accepts JSON only).
- `seamkit/inventory.py`: the seam inventory, derived from a compiled graph and an owner map.
- `tests/test_ch01_breakage.py`: one test per seam property, each passing by showing the break.

### Chapter 2: The Federation Test

- `seamkit/federation.py`: the Federation Test (`federation_test`), `misplaced`, which finds
  calls in the wrong place from a graph and a deployment, and `DecisionRecord`.
- `meridian/decisions.py`: Meridian's owner facts, its five candidate calls, and their
  decision records. `python -m meridian.decisions` rewrites them into `docs/decisions/`.
- `meridian/policy.py`: the policy lookup folded back into the claims graph as a subgraph.
- `meridian/policy_server.py`: the chapter's opening incident, the policy lookup behind its own
  A2A server, kept deliberately wrong.
- `docs/decisions/`: one federation decision record per Meridian call.
- `tests/test_ch02_federation.py`, `tests/test_ch02_costs.py`: the test's verdicts, and the
  policy lookup's seam costs measured both ways.

### Chapter 3: A2A as a LangGraph Runtime Sees It

- `seamkit/remote.py`: `RemoteAgentNode`, the node call shape for a seam; `SeamSpec`, the
  thread link, the total task-state-to-outcome table `OUTCOMES`, and the crossing record.
- `seamkit/errors.py`: `SeamError` and its typed subclasses.
- `seamkit/server.py`: `GraphExecutor`, which serves a compiled LangGraph graph over A2A.
- `seamkit/tools.py`: the remote skill tool, the tool call shape.
- `meridian/risk_server.py`, `meridian/finance_server.py`, `northgate/server.py`: the partners
  as A2A servers (Northgate has no LangGraph in it).
- `meridian/federated.py`: Chapter 1's graph with three `RemoteAgentNode`s.
- `meridian/claims_server.py`: Claims intake exposed as an A2A agent.
- `meridian/incident.py`: the chapter's opening incident, kept deliberately broken.
- Chapter 1's `claims_intake` no longer defaults a missing fraud score to 0.0, and
  `tests/test_ch01_breakage.py`'s failure test now shows the outage raising.

Every test runs the real A2A SDK client against the real SDK servers in one process over
`httpx.ASGITransport`: no sockets, no model keys.

### Chapter 4: The Agent Card Is a Contract, Not a Brochure

- `seamkit/contract.py`: the seam contract. `Field`, `SkillTerms` and `SeamTerms` are the
  partner's terms, published on its card as a data-only extension (`publish`, read back
  with `terms_of`); `Expectations` is the caller's half, loaded from a TOML file; `check`
  holds one against the other; `seam_spec` derives Chapter 3's `SeamSpec`; `args_model`
  generates a remote skill tool's argument schema from the partner's input terms.
- `meridian/expectations/*.toml`: Claims' expectations, one file per seam.
  `meridian/federated.py` derives `SEAMS` from them.
- `meridian/data_classes.py`: Meridian's data classes and which may cross to each owner.
- `seamkit/remote.py`: `connect` checks the contract; a breached seam raises
  `ContractBreach` before sending; a completed artifact is checked against the shapes the
  seam reads. `seamkit/server.py`: `agent_card` takes `extensions`.
- `northgate/server.py`, `meridian/risk_server.py`, `meridian/finance_server.py`: each
  partner's published terms; Northgate's v8 revision requires the vehicle's VIN.
- `tests/test_ch04_contract.py`: the v8 incident, found the day it is announced; the
  data-class refusal of Chapter 1's whole-state send; and the 0-1 score that passes every
  declared check.

Run only this chapter's tests with `uv run pytest tests/test_ch04_contract.py`.

### Chapter 5: Which Cards to Trust

- `seamkit/trust.py`: the card trust policy. `PartnerTrust` is what a caller pinned about one
  owner's card (origins, keys by id or exact digests, approved skills); `decide` refuses a card
  from an unpinned origin, without a signature by a pinned key (or not a pinned digest), or
  without the skills the seam calls, and records any other skill as unapproved. `Gradient` and
  `TrackRecord` are the trust gradient: a skill's answer is capped while it is unproven under a
  card digest, and one malformed answer suspends it.
- `seamkit/cards.py`: `card_digest`, the sha256 of the bytes a card signature covers, and
  `sign_card`, the partner-side signer.
- `seamkit/remote.py`: `connect(trust=...)` decides the card before the contract check;
  `refresh` re-decides a refetched card; a refused seam raises `CardUntrusted` before sending.
- `meridian/trust.py`: Meridian's policy. Northgate is pinned by key; Risk and Finance by the
  digests their pipelines publish with each release.
- `meridian/registry.py`, `meridian/lookalike.py`: the chapter's opening incident, a look-alike
  Northgate found through the partner registry, kept deliberately wrong.
- `tests/test_ch05_trust.py`: the incident before and after the policy, capability inflation,
  and the trust gradient.

Run only this chapter's tests with `uv run pytest tests/test_ch05_trust.py`.

### Chapter 6: On Whose Authority: Delegation Chains Across Owners

- `seamkit/delegation.py`: the delegation chain. `Link` is one signed link (issuer, subject, claims,
  audience, skills, fields, hops left, expiry); `narrows` and `mint` add a link no
  wider than its parent; `DelegationAuth` is the caller's transport, which runs the trust hook,
  reads the run's chain and mints the next link on every request; `verify` checks a chain
  whole; `Guard` wraps a partner server, refuses a chain it cannot verify, and binds it to the
  workload that presents it. `inbound_chain` reads the verified chain inside a partner's run.
- `meridian/identity.py`: `MeridianIssuer`, Meridian's identity provider, which exchanges an
  adjuster's session for the root link of a chain.
- `meridian/federated.py`: `connect_partners` gives each seam a `DelegationAuth`;
  `in_process_http` presents each request as a named workload; `with_authority` fills the seam
  inventory's authority column from each seam's grant.
- `meridian/risk_server.py`, `meridian/finance_server.py`, `northgate/server.py`: every partner
  server behind a `Guard`; Northgate's `estimate_history` reads the insurer from the verified
  chain, not from the request.
- `meridian/incident.py`: `ForwardingAuth` and the per-partner `WORKLOAD_TOKENS`, the
  chapter's opening incident (a partner forwarding the caller's token), kept deliberately
  wrong.
- `tests/test_ch06_delegation.py`: the incident, a chain minted and refused whole, narrowing,
  workload binding, expiry, per-insurer history, and the inventory's authority column.

Run only this chapter's tests with `uv run pytest tests/test_ch06_delegation.py`.

### Chapter 7: Revoking Mid-Task

- `seamkit/revocation.py`: the signed revocation list (`RevocationList`, `revoked`), a
  partner's copy that fails closed when stale (`RevocationSource`), the guard's hook
  (`checker`), the act points (`ActPoint`, `still_holds`), and the onward check (`checked`).
- `meridian/identity.py`: `MeridianIssuer.revoke` ends a person's sessions, drops her cached
  roots, and signs a new list; `cancel_root` mints the root a cancel is sent under, already
  revoked; `revocation_sources` is what a partner holds to pull the list.
- `seamkit/delegation.py`: `Guard(revoked=...)` refuses a revoked chain on every call except
  a cancel of the chain's own task from the workload that sent it.
- `meridian/resume.py`: Claims' resume gate in front of each A2A call node; it cancels a task
  whose recorded chain is revoked, and renews an `auth-required` task for the same person and
  claim.
- `northgate/server.py`: the assessor desk for frame damage, and the act points before an
  estimate is committed. Terms revision `v7.1` states `revocation_seconds = 90`;
  `meridian/expectations/repair_estimate.toml` accepts at most 120.
- `seamkit/drill.py`, `meridian/drill.py`: the delegation revocation drill, on one clock.
  `meridian/incident.py`'s `CachingIssuer` is the chapter's opening incident.
- `tests/test_ch07_revocation.py`: the incident, the drill, the cancel exception, renewal,
  and a stale list.

Run only this chapter's tests with `uv run pytest tests/test_ch07_revocation.py`.

### Chapter 8: Remote Output Is Untrusted Input

- `seamkit/ingress.py`: the seam ingress filter. `IngressRule` closes a field as a value or
  quarantines it as text; `rules_of` reads the rules from the expectations (`pattern`,
  `choices`, `max_chars` in a `[reads.*]` table); `admit` cleans, caps, refuses and tags;
  `signals` is the alarm that decides nothing; `Provenance` and `derive` are the provenance
  tag; `egress` refuses a send carrying partner text or a class its receiver may not hold.
- `seamkit/remote.py`: `_outcome` and `call_once` run the filter after the shape check; the
  tool shape hands a model closed values only; `question_of` and refusal details are cleaned
  and capped; `FilteredNode` checks egress before a send and attaches the tags.
  `UntrustedEgress` is in `seamkit/errors.py`.
- `meridian/review.py`: Claims' review, split into the quarantined reader (`read_notes`,
  `NoteFindings`) and the acting node (`reviewer`), with the memo from a template.
- `northgate/server.py`: terms revision `v7.2` adds the shop's `notes`; history rows may carry
  a shop's `remark`. `meridian/finance_server.py`: release `2026.11` accepts a `payment_memo`.
- `meridian/risk_server.py`: Risk's own rules for `estimate_history` (`HISTORY_RULES`).
- `meridian/incident.py`: `naive_reviewer` and `obeys_notes`, the chapter's opening incident.
- `tests/test_ch08_ingress.py`: the incident, the filter, quarantine against the most
  credulous model, egress, the peer seam, and the question a person reads.

Run only this chapter's tests with `uv run pytest tests/test_ch08_ingress.py`.

### Chapter 10: Interrupt, Resume, and the Human on the Other Side

- `seamkit/server.py`: `ask(question, owner, role)` puts an addressee on an interrupt;
  `GraphExecutor(own=...)` keeps a question for the partner's own people inside, with the task
  `working`, and `answer` finishes it from the partner's inbox through `StoreWriter`, outside
  any request, re-checking the chain at the commit act point.
- `seamkit/approval.py`: the approval credential (`Approval`, `refuse_approval`).
- `seamkit/resume.py`: the cross-owner resume state machine (`TRANSITIONS`, `resume_step`),
  `wake`, the polling `sweep`, and the `fork` guard.
- `seamkit/ledger.py`, `seamkit/remote.py`, `seamkit/contract.py`: the entry's `waiting_on`,
  `since`, `asked` and `terms`; routing by `SeamSpec.roles`; the wait horizon
  (`wait_seconds`) in both halves of the contract. `WaitExpired` is in `seamkit/errors.py`.
- `meridian/resume.py`: Claims' resume gate asks `resume_step` before Chapter 7's `recheck`.
- `meridian/finance_server.py`: release `2026.12`, two-party approval over 2,500
  (`addressed=False` builds the opening incident), Finance's inbox (`approve`), and `ship`.
  `meridian/identity.py`: `MeridianIssuer.approve`.
- `tests/test_ch10_resume.py`: the incident, routing, approvals, every paused case, six days on
  one clock, a card and a question that change while a person is away, and the fork.

Run only this chapter's tests with `uv run pytest tests/test_ch10_resume.py`.

### Chapter 11: Long-Running Tasks: Streaming, Push, and Polling

- `seamkit/delivery.py`: the delivery-mode matrix (`MODES`) and `uncovered`, which refuses a
  seam with no mode, a push seam with no backstop, a backstop as slow as the wait horizon, and
  a push seam whose partner's card cannot push. Each expectations file names its `delivery`
  and `backstop_seconds`.
- `seamkit/push.py`: the per-task `push_token`, `PushTo` (the push config a send carries), the
  verified `receiver`, `WakeQueue` (one pending wake per thread, never during a run), the
  `Backstop` poll, and the `doorbell` app.
- `seamkit/remote.py`: `RemoteAgentNode` gains `push`; `_send` sets the push config.
  `seamkit/resume.py`: `sweep` skips a seam it has no node for.
- `seamkit/server.py`: `agent_card(push=...)`, `a2a_app(push=...)` with the SDK's push config
  store and sender, and `StoreWriter(push=...)`, which rings after a write from outside a
  request. `seamkit/revocation.py`: `RevocationSource.ring`.
- `northgate/server.py`: Northgate pushes, and serves Meridian's revocation doorbell at
  `/revocations`. `meridian/finance_server.py`: Finance pushes, from its inbox too.
- `meridian/federated.py`: `push_seams` and the `bells` argument. `meridian/incident.py`:
  `letter_receiver`, the opening incident's receiver.
- `tests/test_ch11_delivery.py`: the incident, a forged push with a real token, per-task tokens,
  coalescing, the rate limit, a lost push found by the backstop, a dropped stream, the
  revocation doorbell, Finance's inbox, and matrix coverage.

Run only this chapter's tests with `uv run pytest tests/test_ch11_delivery.py`.

### Chapter 12: Retries, Idempotency, and Reconciliation

- `seamkit/server.py`: `KeyedHandler`, the partner's half of a key. A repeat of a key from the
  same caller inside the skill's window gets the task the first send opened; a repeat with a
  different request is refused; a concurrent repeat waits for the first. `a2a_app` reads the
  windows from the card's terms (`SkillTerms.key_seconds`) and takes the key index (`keys=`).
- `seamkit/retry.py`: `RetryPolicy`, `refuse_retry` (reads retry on an outage; sends only under
  an honoured key, inside the deadline and the window less the skew), `backoff` (full jitter,
  `Retry-After`), `retrying`, and `onward_key`.
- `seamkit/contract.py`: `SkillTerms.key_seconds`; `Expectations.retry` and `lookup`; `check`'s
  ninth rule. `meridian/expectations/payments.toml` gains `[retry]`, and
  `fraud_screening.toml` gains `lookup = true`.
- `seamkit/remote.py`: `RemoteAgentNode.recover` (an `unknown` entry is resent under its key,
  or looked up), `lookup`, `tasks`, and the pre-send lookup. `call_once` takes a key, and
  `seamkit/tools.py` derives it from the inbound message id.
- `seamkit/delegation.py`: `verify` refuses a chain that comes back to an owner.
- `seamkit/reconcile.py`: `Reconciler`, its inward and outward passes, `Finding`, `ACTIONS`.
- `meridian/finance_server.py`: release 2026.14 (a 24-hour key window), `backup` and
  `restore`. `northgate/server.py`: terms v7.3 (an hour on both skills).
  `meridian/identity.py`: the reconciliation job's session. `meridian/incident.py`:
  `ResendOnUnknown`, the opening incident's node.
- `tests/test_ch12_retries.py`: the incident, both halves of the key, the retry policy,
  LangGraph's node retry, onward keys, the loop rule, the crash window, found and absent tasks,
  an orphan, and a restore from backup.

Run only this chapter's tests with `uv run pytest tests/test_ch12_retries.py`.

## Contributing

Turn on the repository's hooks once per clone:

```
git config core.hooksPath .githooks
```

## License

MIT License. See `LICENSE`.
