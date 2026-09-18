"""Central application configuration.

All runtime configuration is read from the environment (12-factor). Nothing in the
codebase may read ``os.environ`` directly: everything goes through :func:`get_settings`
so that configuration is typed, validated once at boot, and trivially overridable in
tests.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

Environment = Literal["local", "test", "staging", "production"]


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "TheraGraph"
    environment: Environment = "local"
    version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ------------------------------------------------------------- datastores
    database_url: str = "postgresql+asyncpg://theragraph:theragraph@localhost:5432/theragraph"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False
    database_statement_timeout_ms: int = 30_000

    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_default_ttl_s: int = 3600
    cache_namespace: str = "tg"

    # --------------------------------------------------------- object storage
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_bucket: str = "theragraph"
    s3_access_key_id: str = "theragraph"
    s3_secret_access_key: str = "theragraph"
    s3_region: str = "us-east-1"
    storage_backend: Literal["s3", "filesystem"] = "filesystem"
    storage_root: Path = REPO_ROOT / "artifacts"

    # ------------------------------------------------------------------- data
    data_dir: Path = REPO_ROOT / "data"
    seed_dir: Path = REPO_ROOT / "data" / "seed"
    model_registry_dir: Path = REPO_ROOT / "artifacts" / "models"

    # -------------------------------------------------------- external source
    opentargets_graphql_url: AnyHttpUrl = AnyHttpUrl(
        "https://api.platform.opentargets.org/api/v4/graphql"
    )
    chembl_base_url: AnyHttpUrl = AnyHttpUrl("https://www.ebi.ac.uk/chembl/api/data")
    pubchem_base_url: AnyHttpUrl = AnyHttpUrl("https://pubchem.ncbi.nlm.nih.gov/rest/pug")
    europepmc_base_url: AnyHttpUrl = AnyHttpUrl("https://www.ebi.ac.uk/europepmc/webservices/rest")
    moleculenet_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
    )
    http_timeout_s: float = 30.0
    http_max_retries: int = 4
    http_backoff_base_s: float = 0.5
    http_user_agent: str = "TheraGraph/0.1 (research workbench; +https://github.com/theragraph)"
    allow_live_ingestion: bool = True
    """When False, ingestion connectors refuse network egress and only replay fixtures."""

    # ------------------------------------------------------------------ agent
    anthropic_api_key: str | None = None
    agent_model: str = "claude-opus-5"
    agent_planner_model: str = "claude-opus-5"
    agent_max_tokens: int = 16000
    agent_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    agent_temperature: float = 0.0
    agent_max_steps: int = 12
    agent_tool_timeout_s: float = 60.0
    agent_llm_backend: Literal["anthropic", "deterministic"] = "anthropic"
    """``deterministic`` swaps the LLM for a rule-based planner so the product (and the
    eval suite) runs with no API key and no network. See docs/design_decisions.md."""

    # --------------------------------------------------------------- security
    auth_enabled: bool = False
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_ttl_s: int = 86_400
    api_keys: list[str] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_window_s: int = 60
    rate_limit_agent_requests: int = 20
    max_request_bytes: int = 2_000_000

    # ----------------------------------------------------------------- jobs
    job_queue_backend: Literal["redis", "inline"] = "redis"
    job_max_attempts: int = 3
    job_visibility_timeout_s: int = 900
    job_worker_concurrency: int = 4
    job_poll_interval_s: float = 0.5

    # ---------------------------------------------------------- observability
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    otel_enabled: bool = False
    otel_exporter_endpoint: str | None = None
    otel_service_name: str = "theragraph-api"
    metrics_enabled: bool = True
    slow_query_threshold_ms: float = 250.0

    @field_validator("cors_origins", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated lists in env vars (``A,B`` -> ``["A", "B"]``)."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def sync_database_url(self) -> str:
        """Alembic and other sync tooling need a non-async driver."""
        return (
            self.database_url.replace("+asyncpg", "+psycopg2")
            .replace("+aiosqlite", "")
            .replace("postgresql+psycopg2", "postgresql")
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (tests only)."""
    get_settings.cache_clear()


SettingsDep = Annotated[Settings, "settings"]
