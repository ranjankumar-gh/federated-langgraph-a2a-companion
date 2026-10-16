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

## Contributing

Turn on the repository's hooks once per clone:

```
git config core.hooksPath .githooks
```

## License

MIT License. See `LICENSE`.
