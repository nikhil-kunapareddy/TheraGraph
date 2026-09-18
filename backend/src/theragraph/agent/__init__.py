"""The AI research agent.

An explicit planner/executor workflow, not a single prompt:

    question -> plan -> tool execution -> output validation -> evidence validation -> report

Design constraints, all enforced in code rather than requested in a prompt:

* The model may only obtain facts by calling typed tools backed by the database. It has
  no free-text channel into the report's factual content.
* Every tool call is schema-validated on the way in and on the way out, and persisted.
* Report claims must cite records that were actually retrieved during the run;
  unresolvable citations are removed before the reader sees them.
* Predictions are labelled as predictions, with uncertainty and applicability-domain
  status, everywhere they appear.
* Hidden chain-of-thought is never persisted or exposed. What is stored is the
  structured plan, the tool calls, and a short user-facing rationale the model is
  explicitly asked to produce.
"""

from theragraph.agent.llm import (
    LLMResponse,
    ModelUsage,
    build_llm,
    estimate_cost_usd,
)
from theragraph.agent.orchestrator import execute_run, start_research
from theragraph.agent.tools import TOOL_REGISTRY, ToolResult, tool_schemas

__all__ = [
    "TOOL_REGISTRY",
    "LLMResponse",
    "ModelUsage",
    "ToolResult",
    "build_llm",
    "estimate_cost_usd",
    "execute_run",
    "start_research",
    "tool_schemas",
]
