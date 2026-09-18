"""Domain services.

Services are the seam between the HTTP/agent/MCP surfaces and the modules that do the
real work (ingestion, chemistry, ML, ranking). They own orchestration and transactions;
they do not own SQL (repositories) or science (chem/ml/ranking). Every agent tool and
every REST endpoint calls a service, so the two surfaces can never drift apart in
behaviour.
"""

from theragraph.services.context import ServiceContext

__all__ = ["ServiceContext"]
