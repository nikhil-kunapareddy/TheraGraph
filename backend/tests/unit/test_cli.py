"""The operator command line.

The CLI is what CI, the Makefile and an operator actually invoke, so its contract --
exit codes, JSON on stdout, secrets redacted, a typo rejected with the valid options --
is worth asserting rather than discovering in a failed deploy.

These tests drive the Typer app in-process with a throwaway SQLite database, so they
touch no network, no PostgreSQL and no model registry beyond a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from theragraph.cli import _parse_families, app
from theragraph.config import reset_settings_cache

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every path-valued setting at a temporary directory."""
    database = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database}")
    monkeypatch.setenv("REDIS_URL", "memory://")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("ALLOW_LIVE_INGESTION", "false")
    monkeypatch.setenv("AGENT_LLM_BACKEND", "deterministic")
    monkeypatch.setenv("STORAGE_BACKEND", "filesystem")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "artifacts" / "models"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEED_DIR", str(tmp_path / "data" / "seed"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings_cache()
    yield database
    reset_settings_cache()


def invoke(*args: str) -> Any:
    result = runner.invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def payload(result: Any) -> dict[str, Any]:
    """The last JSON object on stdout, ignoring any log lines above it."""
    text = result.stdout
    start = text.index("{")
    return json.loads(text[start:])


class TestInfoAndVersion:
    def test_version_reports_rdkit_too(self, cli_env: Path) -> None:
        """The RDKit build is part of the answer: descriptors change between versions."""
        result = invoke("version")
        assert result.exit_code == 0
        assert "theragraph" in result.stdout
        assert "rdkit" in result.stdout

    def test_info_redacts_secrets(self, cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
        monkeypatch.setenv("JWT_SECRET", "another-secret-value")
        reset_settings_cache()

        result = invoke("info")

        assert result.exit_code == 0
        assert "sk-ant-not-a-real-key" not in result.stdout
        assert "another-secret-value" not in result.stdout
        body = payload(result)
        assert body["anthropic_api_key"] == "[REDACTED]"
        assert body["jwt_secret"] == "[REDACTED]"

    def test_info_reports_the_effective_agent_backend(self, cli_env: Path) -> None:
        """With no key the configured backend is irrelevant; say what will actually run."""
        result = invoke("info")
        body = payload(result)
        assert body["agent_llm_backend"] == "deterministic"
        assert body["agent_backend_effective"] == "deterministic"


class TestDatabaseCommands:
    def test_bootstrap_then_check_then_demo_corpus(self, cli_env: Path) -> None:
        created = invoke("db", "bootstrap")
        assert created.exit_code == 0
        assert payload(created)["tables"] == 28

        checked = invoke("db", "check")
        assert checked.exit_code == 0
        assert payload(checked)["ok"] is True

        sources = invoke("db", "sources")
        assert payload(sources)["registered_sources"] == 5

        corpus = invoke("db", "demo-corpus")
        assert corpus.exit_code == 0
        loaded = payload(corpus)["loaded"]
        assert len(loaded["compound_ids"]) == 4
        assert len(loaded["target_ids"]) == 2
        assert loaded["disease_id"]

    def test_check_exits_non_zero_when_the_database_is_unreachable(
        self, cli_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A health command that always exits 0 is not a health command."""
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:////nonexistent/dir/db.sqlite3")
        reset_settings_cache()

        result = invoke("db", "check")

        assert result.exit_code == 1


class TestAgentCommands:
    def test_tools_are_listed_with_their_schemas(self, cli_env: Path) -> None:
        result = invoke("agent", "tools")
        assert result.exit_code == 0
        body = payload(result)
        assert len(body["tools"]) == 15
        mutating = [tool["name"] for tool in body["tools"] if tool["mutates"]]
        assert sorted(mutating) == ["ingest_disease", "ingest_target_bioactivity"]
        for tool in body["tools"]:
            assert tool["input_schema"]["type"] == "object"
            assert tool["description"]

    def test_eval_rejects_an_unknown_case_before_running_anything(
        self, cli_env: Path, tmp_path: Path
    ) -> None:
        """A typo must fail as a bad parameter, not run the whole suite and report zero."""
        report = tmp_path / "eval.json"

        result = invoke("agent", "eval", "--case", "not-a-case", "--output", str(report))

        assert result.exit_code == 2  # click's exit code for a bad parameter
        assert not report.exists()


class TestOpenApiCommand:
    def test_writes_the_schema_and_reports_the_path_count(
        self, cli_env: Path, tmp_path: Path
    ) -> None:
        destination = tmp_path / "nested" / "openapi.json"

        result = invoke("openapi", "--output", str(destination))

        assert result.exit_code == 0
        assert payload(result)["paths"] == 45
        schema = json.loads(destination.read_text())
        assert schema["openapi"].startswith("3.")
        assert "/api/v1/agent/research" in schema["paths"]


class TestFamilyParsing:
    def test_accepts_the_documented_values(self) -> None:
        assert _parse_families("random_forest,gin") == ("random_forest", "gin")
        assert _parse_families(" gin , xgboost ") == ("gin", "xgboost")

    def test_rejects_a_typo_with_the_allowed_values(self) -> None:
        """Failing here beats failing inside the training loop twenty minutes later."""
        with pytest.raises(typer.BadParameter) as caught:
            _parse_families("random_forset")

        message = str(caught.value)
        assert "random_forset" in message
        assert "random_forest" in message
        assert "gin" in message

    def test_ignores_empty_entries(self) -> None:
        assert _parse_families("gin,,") == ("gin",)
