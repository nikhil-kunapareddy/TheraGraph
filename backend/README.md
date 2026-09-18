# TheraGraph backend

FastAPI modular monolith that powers TheraGraph. See the [repository README](../README.md)
for the product overview and [docs/](../docs) for architecture, ingestion, training,
evaluation and deployment guides.

```
src/theragraph/
  api/         HTTP layer: routers, dependencies, middleware
  agent/       planner / tool-executor research agent + eval harness
  chem/        RDKit-backed molecular processing (descriptors, fingerprints, scaffolds)
  core/        cross-cutting concerns (logging, ids, cache, telemetry, security, storage)
  db/          SQLAlchemy models, session management, repositories
  ingestion/   connectors + normalisers for Open Targets, ChEMBL, PubChem, MoleculeNet
  jobs/        Redis-backed background job queue and worker
  mcp_server/  Model Context Protocol surface over the scientific tools
  ml/          datasets, splits, baseline + graph models, evaluation, inference
  ranking/     transparent multi-objective candidate prioritisation
  schemas/     Pydantic request/response contracts
  services/    domain services orchestrating the modules above
```
