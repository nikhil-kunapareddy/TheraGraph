"""MCP server exposing the scientific tool registry.

Built on the MCP Python SDK (2.x, ``MCPServer``). Each TheraGraph tool becomes an MCP
tool with the same name, description and JSON schema; the handler opens a database
session, runs the shared tool implementation and returns its structured payload plus
provenance.

Design notes:

* **One session per call.** MCP calls are independent, and a long-lived session would
  hold a transaction open across idle time.
* **Errors are returned, not raised.** A tool failure comes back as a structured error
  payload so the calling model can adapt, matching the internal agent's behaviour.
* **Provenance travels with every result.** The MCP client sees the same source /
  retrieval-date / citation-key fields the UI shows, so a model on the other end can
  cite correctly.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from theragraph.agent.tools import TOOL_REGISTRY, ToolSpec, execute_tool
from theragraph.config import Settings, get_settings
from theragraph.core.cache import build_cache
from theragraph.core.logging import configure_logging, get_logger
from theragraph.core.storage import build_object_store
from theragraph.db.session import init_engine, session_scope
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

SERVER_INSTRUCTIONS = """
TheraGraph exposes an evidence-grounded drug-discovery corpus: diseases, therapeutic
targets, retrieved scientific evidence, compounds with measured bioactivity, and
versioned molecular property models.

Rules for using these tools:

1. Facts come only from tool results. Do not supplement them from memory.
2. Cite retrieved records using the `citation_key` each record carries.
3. Anything from `predict_properties` or `prioritize_candidates` is a model output.
   Report it as a prediction, with its uncertainty and applicability-domain status.
4. Prioritisation ranks compounds for experimental follow-up. It is not a statement
   about clinical efficacy or safety.
5. An empty result means the ingested corpus has nothing, not that the literature is
   empty. Say which.

Start with `describe_corpus` to see what is loaded, or `search_entities` to resolve a
disease, target or compound.
""".strip()


def _allow_ingestion() -> bool:
    return os.environ.get("THERAGRAPH_MCP_ALLOW_INGESTION", "").lower() in (
        "1",
        "true",
        "yes",
    )


def build_server(settings: Settings | None = None) -> Any:
    """Construct the MCP server with one tool per registry entry."""
    from mcp.server.mcpserver import MCPServer

    resolved = settings or get_settings()
    allow_ingestion = _allow_ingestion()

    server = MCPServer(
        name="theragraph",
        title="TheraGraph drug-discovery workbench",
        description=(
            "Evidence-grounded disease, target, compound and molecular-property tools "
            "backed by Open Targets, ChEMBL, PubChem and Europe PMC."
        ),
        instructions=SERVER_INSTRUCTIONS,
        version=resolved.version,
    )

    cache = build_cache(
        resolved.redis_url,
        enabled=resolved.cache_enabled,
        default_ttl_s=resolved.cache_default_ttl_s,
    )
    store = build_object_store(
        backend=resolved.storage_backend,
        root=resolved.storage_root,
        bucket=resolved.s3_bucket,
        endpoint_url=resolved.s3_endpoint_url,
        access_key=resolved.s3_access_key_id,
        secret_key=resolved.s3_secret_access_key,
        region=resolved.s3_region,
    )

    for spec in sorted(TOOL_REGISTRY.values(), key=lambda item: item.name):
        if spec.mutates and not allow_ingestion:
            continue
        _register(server, spec, resolved, cache, store)

    log.info(
        "mcp.server_built",
        tools=len([spec for spec in TOOL_REGISTRY.values() if not spec.mutates])
        + (len(TOOL_REGISTRY) if allow_ingestion else 0),
        allow_ingestion=allow_ingestion,
    )
    return server


def _register(server: Any, spec: ToolSpec, settings: Settings, cache: Any, store: Any) -> None:
    """Register one tool with a *flat* schema derived from its Pydantic model.

    The MCP SDK builds a tool's input schema from the handler's signature. Taking the
    Pydantic model as a single parameter would nest everything under ``payload``, so the
    handler's signature is synthesised from the model's fields instead: one keyword
    parameter per field, with the model's own types and defaults. The result is a flat
    schema identical to what the internal agent validates against -- one definition,
    two surfaces.
    """
    fields = spec.input_model.model_fields
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    for name, field in fields.items():
        if field.is_required():
            default: Any = inspect.Parameter.empty
        elif field.default_factory is not None:  # type: ignore[truthy-function]
            default = field.default_factory()  # type: ignore[misc,call-arg]
        else:
            default = field.default
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation,
            )
        )
        annotations[name] = field.annotation

    # Required parameters must precede optional ones in a valid signature.
    parameters.sort(key=lambda parameter: parameter.default is not inspect.Parameter.empty)
    annotations["return"] = dict[str, Any]

    async def handler(**kwargs: Any) -> dict[str, Any]:
        try:
            payload = spec.input_model.model_validate(kwargs)
        except Exception as exc:
            return {
                "ok": False,
                "tool": spec.name,
                "error": {"code": "validation_error", "message": str(exc)[:600]},
            }

        async with session_scope() as session:
            context = ServiceContext(settings=settings, session=session, cache=cache, store=store)
            execution = await execute_tool(
                context,
                spec.name,
                payload.model_dump(mode="json"),
                allow_mutating=spec.mutates,
            )

        if not execution.succeeded or execution.result is None:
            log.warning("mcp.tool_failed", tool=spec.name, error=execution.error)
            return {
                "ok": False,
                "tool": spec.name,
                "error": {
                    "code": execution.error_code or "tool_error",
                    "message": execution.error or "the tool failed",
                },
            }

        result = execution.result
        return {
            "ok": True,
            "tool": spec.name,
            "summary": result.summary,
            "data": result.data,
            "citation_keys": result.citations[:100],
            "record_ids": result.record_ids,
            "is_empty": result.is_empty,
            "warnings": result.warnings,
            "latency_ms": round(execution.latency_ms, 2),
            "provenance_note": (
                "Model output; not an experimental measurement."
                if spec.category == "ml"
                else "Retrieved from a named public source; see per-record provenance."
            ),
        }

    handler.__name__ = spec.name
    handler.__qualname__ = f"theragraph_mcp.{spec.name}"
    handler.__doc__ = spec.description
    handler.__annotations__ = annotations
    handler.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters, return_annotation=dict[str, Any]
    )

    server.add_tool(handler, name=spec.name, description=spec.description)


def main() -> None:
    """Entry point for ``theragraph mcp`` and for MCP client configuration."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        # stdio transport puts the protocol on stdout, so logs must be structured and
        # go to the same stream only as JSON the client ignores -- console formatting
        # here would corrupt the protocol stream.
        fmt="json",
        service="theragraph-mcp",
    )
    init_engine(settings)
    server = build_server(settings)
    transport = os.environ.get("THERAGRAPH_MCP_TRANSPORT", "stdio")
    log.info("mcp.starting", transport=transport)
    server.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    main()
