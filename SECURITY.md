# Security policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately, in order of preference:

1. **GitHub private vulnerability reporting** — on this repository, go to
   *Security → Report a vulnerability*
   (<https://github.com/nikhil-kunapareddy/TheraGraph/security/advisories/new>).
2. **Email** — <nikhilkunapareddy@gmail.com>, with `[TheraGraph security]` in the subject.

Please include what you found, how to reproduce it (a request, a payload, or a failing
test is ideal), the version or commit, and the impact as you understand it.

What to expect:

| | |
|---|---|
| Acknowledgement | within 3 working days |
| Assessment and a plan | within 10 working days |
| Fix for a confirmed high or critical issue | as fast as it can be made safely, typically within 30 days |

You will be credited in the advisory unless you ask not to be. Please give a reasonable
window to ship a fix before disclosing publicly; coordinated disclosure through a GitHub
security advisory is the default.

## Supported versions

TheraGraph is pre-1.0. Security fixes land on `main` and in the next release; there are
no back-ported fixes for older versions.

## Scope

In scope: the backend (`backend/`), the frontend (`frontend/`), the MCP server, the
container images and compose files, and the CI configuration in this repository.

Out of scope: the upstream data sources (Open Targets, ChEMBL, PubChem, Europe PMC,
MoleculeNet) and vulnerabilities in third-party dependencies that are already public
(Dependabot tracks those — a report is still welcome if TheraGraph is exploitable through
one in a way the upstream advisory does not cover).

## Deployment security model

The defaults are for a single developer on a trusted machine. Before exposing TheraGraph
to a network, read [docs/operations.md](docs/operations.md#security). In short:

* Run with `ENVIRONMENT=production`. The API then refuses to boot unless
  `AUTH_ENABLED=true` and `JWT_SECRET` is a non-default secret of at least 32 characters.
* Put the API behind the bundled nginx (or an equivalent proxy) and do not publish its
  port directly. `FORWARDED_ALLOW_IPS` must list only proxies you control.
* Set `ALLOW_LIVE_INGESTION=false` on a public deployment.
* Treat the model registry and the S3 bucket as trusted storage. Tree models are joblib
  (pickle) files; anyone who can write there can execute code in the workers.
* The MCP server has no authentication. Keep it on stdio, or behind an authenticating
  proxy if you enable an HTTP transport.
* The compose datastore credentials are development defaults. Replace all of them.

TheraGraph's outputs are research aids, not medical advice; see the README's limitations.
