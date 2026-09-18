"""Model Context Protocol surface over TheraGraph's scientific tools.

The MCP server exposes the *same* tool registry the internal agent uses, so an external
MCP client (Claude Desktop, an IDE, another agent) gets an identical contract: identical
schemas, identical provenance, identical guardrails. There is no second implementation
to drift.

Read-only tools are exposed by default. Ingestion tools mutate the corpus and consume
upstream API quota, so they are opt-in via ``THERAGRAPH_MCP_ALLOW_INGESTION=true``.
"""

from theragraph.mcp_server.server import build_server, main

__all__ = ["build_server", "main"]
