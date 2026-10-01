# Contributing to TheraGraph

Thanks for helping. This file is the rule set for changes to this repository: what a
change must include, what CI enforces, and how review works.

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE), and you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

**Security problems are not issues.** Report them privately as described in
[SECURITY.md](SECURITY.md).

## Getting set up

Needs Python 3.11, [uv](https://docs.astral.sh/uv/) and Node 20+. No API keys and no
network are needed for development.

```bash
make setup
printf 'DATABASE_URL=sqlite+aiosqlite:///./tg.sqlite3\nREDIS_URL=memory://\n' > .env
make migrate && make seed-smoke && make train
make check            # lint + typecheck + every test: exactly what CI runs
```

`make help` lists every target.

## Ground rules

These are enforced in review, and most of them by CI.

### Scientific integrity

1. **Retrieved facts and model predictions stay separate.** Anything from a public source
   carries its source, identifier, retrieval time and dataset version. Anything a model
   produced carries its model version, uncertainty and applicability-domain status, and is
   labelled as a prediction wherever it appears. A change that blurs the two is rejected.
2. **The agent cannot fabricate evidence.** Tools are its only source of facts, and the
   validator removes claims whose citations were not retrieved in the run. Do not add a
   path around either.
3. **No claim of clinical efficacy**, in code, docs, UI copy or examples.
4. **Metrics are reported honestly.** Scaffold splits are the headline; random splits are
   the labelled optimism reference. Do not replace a measured number with a nicer one.

### Code

1. **Configuration goes through `theragraph.config.Settings`.** Nothing reads
   `os.environ` directly (the MCP transport switch is the one documented exception).
2. **Request bodies are strict pydantic models** (`extra="forbid"`) with bounds on every
   number, string and list.
3. **Every endpoint that spends tokens, starts a job, or writes to the corpus checks a
   role** (`principal.require_role("researcher")` or stricter).
4. **No new network egress in tests.** Ingestion is exercised against committed fixtures
   or in-process fakes; CI runs offline.
5. **Tests come with the change.** A bug fix includes a test that fails without it. A
   security fix includes a test that shows the gate refusing.
6. **Migrations are additive and reversible.** `alembic check` and the downgrade
   round-trip run in CI on PostgreSQL and SQLite.
7. **Comments explain why**, not what. Match the surrounding style.

### Secrets and data

1. Never commit a `.env`, a key, a token, a database file, or a trained model. They are
   gitignored; CI also runs gitleaks over the full history.
2. New upstream fixtures go under `data/seed/fixtures/` via `--mode record`, and only from
   sources whose licence permits redistribution. Add the licence to `NOTICE` if it is new.

## CI gates

A pull request merges only when every required check is green:

| Check | What it enforces |
|---|---|
| Backend (lint, types, tests) | `ruff check`, `ruff format --check`, `mypy` (strict), `pytest`, and `docs/openapi.json` is current |
| Backend on PostgreSQL | the suite against PostgreSQL 16 + Redis |
| Migrations | upgrade → `alembic check` → downgrade → upgrade, on both dialects |
| Frontend (lint, types, tests, build) | eslint (zero warnings), prettier, `tsc --noEmit`, vitest, build |
| Agent evaluation gate | the agent's pass-rate, citation and zero-hallucination thresholds |
| Container images | both images build; compose files validate |
| Security | CodeQL, gitleaks, `pip-audit`, `npm audit` |

If you change an API route or schema, run `make openapi` and commit `docs/openapi.json`.

## Branches, commits and pull requests

* `main` is protected: no direct pushes, no force-pushes, no deletion. All changes land
  through a pull request with green checks, no high-severity CodeQL alerts, and resolved
  review threads, and merge as a squash or rebase (linear history). Release tags `v*`
  cannot be moved or deleted.
* Those rules are versioned in [`.github/rulesets/`](.github/rulesets) and applied with
  `bash scripts/apply-github-settings.sh`, so a change to branch protection is reviewed
  like any other change.
* Branch from `main`: `feat/…`, `fix/…`, `docs/…`, `chore/…`.
* Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/),
  scoped to the module: `feat(ranking): …`, `fix(agent): …`, `docs: …`. The subject says
  what changes for a user, in the present tense.
* Keep a pull request to one concern. Fill in the template, including how you tested it.

## Reporting bugs and proposing features

Use the issue templates. For a bug, include the command or request, what you expected,
what happened, and the `X-Request-ID` from the response if there was one. For anything
larger than a small change, open an issue to agree on the approach before writing code.
