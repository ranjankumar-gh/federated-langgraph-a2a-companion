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

## Contributing

Turn on the repository's hooks once per clone:

```
git config core.hooksPath .githooks
```

## License

MIT License. See `LICENSE`.
