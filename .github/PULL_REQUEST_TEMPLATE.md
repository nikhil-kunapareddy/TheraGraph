## What and why

<!-- One concern per pull request. What changes for a user, and why. Link the issue. -->

Closes #

## How it was tested

<!-- The commands you ran and what they showed. A bug fix needs a test that fails without it. -->

- [ ] `make check` passes locally (lint, types, every test)
- [ ] `make openapi` run and `docs/openapi.json` committed, if a route or schema changed
- [ ] Migration added and `alembic check` clean, if a model changed

## Checklist

- [ ] Retrieved facts and model predictions are still labelled separately
- [ ] New endpoints that spend tokens, start jobs or write data check a role
- [ ] No secrets, `.env`, database files or model artifacts committed
- [ ] No new network calls in tests
- [ ] Docs updated where behaviour changed
