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

## Contributing

Turn on the repository's hooks once per clone:

```
git config core.hooksPath .githooks
```

## License

MIT License. See `LICENSE`.
