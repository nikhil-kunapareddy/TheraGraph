"""HTTP contract: status codes, error envelopes, validation, provenance, headers."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestSystemEndpoints:
    async def test_liveness_does_not_touch_dependencies(self, api_client: AsyncClient) -> None:
        """A database blip must not make an orchestrator kill a healthy process."""
        response = await api_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["checks"] == {}

    async def test_readiness_reports_each_dependency(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/readyz")).json()
        assert payload["checks"]["database"]["ok"] is True
        assert "agent_backend" in payload["checks"]

    async def test_metrics_are_exposed_in_prometheus_format(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/metrics")
        assert response.status_code == 200
        assert "theragraph_http_requests_total" in response.text

    async def test_request_id_is_echoed(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/healthz")
        assert response.headers["X-Request-ID"]

    async def test_supplied_request_id_is_preserved(self, api_client: AsyncClient) -> None:
        """So a client can correlate its own logs with the server's."""
        response = await api_client.get("/healthz", headers={"X-Request-ID": "req_client_123"})
        assert response.headers["X-Request-ID"] == "req_client_123"

    async def test_openapi_documents_every_route(self, api_client: AsyncClient) -> None:
        schema = (await api_client.get("/openapi.json")).json()
        assert len(schema["paths"]) > 30
        for path in ("/api/v1/search", "/api/v1/agent/research", "/api/v1/runs/{run_id}"):
            assert path in schema["paths"]

    async def test_root_states_the_product_framing(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/")).json()
        assert "separately" in payload["description"]


class TestSearchEndpoints:
    async def test_search_finds_seeded_entities(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/search", params={"q": "BACE1"})).json()
        assert payload["counts"]["target"] >= 1
        assert payload["hits"][0]["entity_type"] == "target"

    async def test_search_reports_which_term_matched(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """A surprising result must be explainable."""
        payload = (
            await api_client.get(
                "/api/v1/search", params={"q": "BACE1 evidence in Alzheimer disease"}
            )
        ).json()
        assert payload["matched_term"]
        assert payload["terms_tried"]

    async def test_search_hits_carry_provenance(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/search", params={"q": "Aspirin"})).json()
        assert payload["hits"][0]["source"]
        assert payload["hits"][0]["metadata"]["retrieved_at"]

    async def test_entity_type_filter(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (
            await api_client.get(
                "/api/v1/search", params={"q": "Alzheimer disease", "entity_types": "disease"}
            )
        ).json()
        assert all(hit["entity_type"] == "disease" for hit in payload["hits"])

    async def test_missing_query_is_a_validation_error(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/search")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_corpus_summary_lists_sources_and_licences(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/search/corpus")).json()
        assert payload["counts"]["compounds"] == 4
        assert any(source["license"] for source in payload["sources"])
        assert payload["dataset_versions"][0]["content_hash"]


class TestEntityEndpoints:
    async def test_disease_returns_associated_targets_with_scores(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/diseases/MONDO_0004975")).json()
        assert payload["disease"]["name"] == "Alzheimer disease"
        assert payload["associated_targets"][0]["association"]["overall_score"] == 0.87
        assert (
            "retrieved verbatim" in payload["associated_targets"][0]["association"]["score_meaning"]
        )

    async def test_unknown_disease_is_404_with_a_hint(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/diseases/EFO_0000000")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert "hint" in error["details"]

    async def test_target_groups_evidence_by_type_with_meanings(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (
            await api_client.get(
                "/api/v1/targets/APP", params={"disease": "MONDO_0004975", "min_pchembl": 0}
            )
        ).json()
        groups = {group["evidence_type"]: group for group in payload["evidence"]["groups"]}
        assert "genetic_association" in groups
        assert groups["genetic_association"]["meaning"]
        # Genetics must be presented before literature.
        order = [group["evidence_type"] for group in payload["evidence"]["groups"]]
        assert order.index("genetic_association") < order.index("literature")

    async def test_evidence_records_carry_citation_keys_and_urls(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (
            await api_client.get("/api/v1/targets/APP/evidence", params={"limit": 10})
        ).json()
        item = payload["items"][0]
        assert item["citation_key"].startswith("open_targets:")
        assert item["provenance"]["retrieved_at"]
        assert item["is_retrieved_fact"] is True

    async def test_absent_score_is_labelled_not_zeroed(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """The literature record in the fixture has no source score."""
        payload = (
            await api_client.get(
                "/api/v1/targets/APP/evidence", params={"evidence_types": "literature"}
            )
        ).json()
        item = payload["items"][0]
        assert item["score"] is None
        assert item["score_provided_by_source"] is False

    async def test_target_without_chemistry_says_so(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """Silence would read as "no compounds exist"; the note says what it means."""
        payload = (await api_client.get("/api/v1/targets/APP", params={"min_pchembl": 14.0})).json()
        assert payload["known_compounds"] == []
        assert payload["chemistry_summary"]["note"]

    async def test_compound_separates_measurements_from_computed_properties(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/compounds/CHEMBL25")).json()
        assert payload["compound"]["name"] == "Aspirin"
        assert payload["properties"]["basis"].startswith("computed")
        assert payload["bioactivity"]["measurements"][0]["is_experimental_measurement"] is True

    async def test_compound_max_phase_meaning_is_explained(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/compounds/CHEMBL25")).json()
        assert payload["compound"]["max_phase_meaning"] == "Approved"

    async def test_descriptor_dictionary_explains_every_value(
        self, api_client: AsyncClient
    ) -> None:
        payload = (await api_client.get("/api/v1/compounds/descriptors")).json()
        assert len(payload["descriptors"]) > 10
        assert all(entry["explanation"] for entry in payload["descriptors"])


class TestChemistryEndpoints:
    async def test_properties_for_a_pasted_structure(self, api_client: AsyncClient) -> None:
        response = await api_client.post(
            "/api/v1/compounds/properties", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["properties"]["values"]["molecular_weight"] == pytest.approx(
            180.159, abs=0.01
        )
        assert payload["persisted"] is False

    async def test_invalid_structure_is_422_with_a_typed_code(
        self, api_client: AsyncClient
    ) -> None:
        response = await api_client.post(
            "/api/v1/compounds/properties", json={"smiles": "not-a-molecule"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_chemistry"

    async def test_similarity_search_reports_its_own_efficiency(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/compounds/similarity",
            json={"smiles": "CC(=O)Oc1ccccc1C(=O)O", "threshold": 0.3, "limit": 5},
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["hits"]
        assert payload["search_stats"]["candidates_examined"] >= 1
        assert "popcount" in payload["search_stats"]["method"]

    async def test_similarity_rejects_an_out_of_range_threshold(
        self, api_client: AsyncClient
    ) -> None:
        response = await api_client.post(
            "/api/v1/compounds/similarity", json={"smiles": "CCO", "threshold": 9}
        )
        assert response.status_code == 422

    async def test_unknown_field_is_rejected(self, api_client: AsyncClient) -> None:
        """Strict request models catch typos rather than silently ignoring them."""
        response = await api_client.post(
            "/api/v1/compounds/similarity", json={"smiles": "CCO", "treshold": 0.7}
        )
        assert response.status_code == 422

    async def test_substructure_search_confirms_every_match(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/compounds/substructure", json={"smarts": "c1ccccc1", "limit": 10}
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["count"] >= 1
        assert payload["search_stats"]["confirmed_by_graph_match"] == payload["count"]


class TestPredictionEndpoints:
    async def test_endpoints_list_is_empty_without_models(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/predictions/endpoints")).json()
        assert payload["endpoints"] == []

    async def test_predicting_without_a_model_is_a_clear_error(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/predictions", json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}
        )
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "model_unavailable"
        assert "train one with" in error["details"]["hint"]

    async def test_no_input_is_rejected(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/api/v1/predictions", json={})
        assert response.status_code == 422
        assert "compound_ids" in response.json()["error"]["message"]


class TestRankingEndpoints:
    async def test_presets_expose_full_configurations(self, api_client: AsyncClient) -> None:
        """The UI renders sliders and explanations from this, so it must be complete."""
        payload = (await api_client.get("/api/v1/ranking/presets")).json()
        assert len(payload["presets"]) >= 3
        config = payload["configs"]["general_oral_small_molecule"]
        assert all(objective["explanation"] for objective in config["objectives"])
        assert all(objective["desirability"]["description"] for objective in config["objectives"])

    async def test_prioritisation_returns_attributable_scores(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/ranking",
            json={
                "target": "BACE1",
                "disease": "MONDO_0004975",
                "min_pchembl": 0.0,
                "limit": 10,
                "run_predictions": False,
            },
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["candidates"]
        candidate = payload["candidates"][0]
        assert candidate["factors"]
        assert "not a prediction of clinical efficacy" in candidate["disclaimer"]

    async def test_weights_are_validated_against_the_config(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/ranking",
            json={"target": "BACE1", "weights": {"not_an_objective": 1.0}},
        )
        assert response.status_code == 422
        assert "unknown" in response.json()["error"]["message"].lower()

    async def test_unknown_target_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/api/v1/ranking", json={"target": "NOPE"})
        assert response.status_code == 404

    async def test_comparison_requires_at_least_one_compound(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/api/v1/ranking/compare", json={"compound_ids": []})
        assert response.status_code == 422

    async def test_comparison_without_a_target_says_what_it_excluded(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/ranking/compare",
            json={"compound_ids": seeded["compound_ids"][:3]},
        )
        payload = response.json()
        assert response.status_code == 200
        assert "No target supplied" in payload["note"]


class TestGraphEndpoint:
    async def test_graph_edges_name_their_source(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (
            await api_client.get("/api/v1/graph", params={"disease": "MONDO_0004975"})
        ).json()
        assert payload["stats"]["n_nodes"] > 0
        for edge in payload["edges"]:
            assert edge["evidence_source"] or edge["relation"]
            assert "score_provided_by_source" in edge

    async def test_prediction_nodes_are_marked_as_model_output(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """The renderer relies on this to keep predictions visually distinct."""
        payload = (
            await api_client.get("/api/v1/graph", params={"disease": "MONDO_0004975"})
        ).json()
        natures = {node["nature"] for node in payload["nodes"]}
        assert natures <= {"retrieved", "model_output"}
        assert payload["legend"]["natures"]["model_output"]

    async def test_graph_requires_a_root(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/graph")
        assert response.status_code == 404


class TestAgentEndpoints:
    async def test_tool_surface_is_documented(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/agent/tools")).json()
        assert len(payload["tools"]) >= 10
        assert payload["guarantees"]
        assert all(tool["input_schema"] for tool in payload["tools"])

    async def test_write_tools_are_marked(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/agent/tools")).json()
        mutating = [tool for tool in payload["tools"] if tool["mutates"]]
        assert mutating
        assert payload["default_exposed"] < len(payload["tools"])

    async def test_research_run_executes_inline(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        response = await api_client.post(
            "/api/v1/agent/research",
            json={
                "query": "What targets are associated with Alzheimer disease?",
                "parameters": {"n_targets": 5},
                "execute_inline": True,
            },
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["run_id"].startswith("RUN-")
        assert payload["status"] == "succeeded"

    async def test_short_query_is_rejected(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/api/v1/agent/research", json={"query": "hi"})
        assert response.status_code == 422

    async def test_idempotency_key_returns_the_same_run(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """Re-submitting must not spend model tokens twice."""
        body = {
            "query": "Which targets matter for Alzheimer disease?",
            "execute_inline": True,
            "idempotency_key": "test-key-1",
        }
        first = (await api_client.post("/api/v1/agent/research", json=body)).json()
        second = (await api_client.post("/api/v1/agent/research", json=body)).json()
        assert first["run_id"] == second["run_id"]
        assert second["idempotent_hit"] is True


class TestRunEndpoints:
    async def test_run_trace_contains_everything_needed_to_reproduce(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        created = (
            await api_client.post(
                "/api/v1/agent/research",
                json={
                    "query": "What targets are associated with Alzheimer disease?",
                    "execute_inline": True,
                },
            )
        ).json()
        payload = (await api_client.get(f"/api/v1/runs/{created['run_id']}")).json()

        assert payload["reproducibility_hash"]
        assert payload["pins"]["dataset_versions"]
        assert payload["prompts"]
        assert payload["executions"]
        assert payload["tool_calls"]
        assert payload["retrieved_record_ids"]
        assert payload["plan"]["steps"]

    async def test_report_claims_are_validated(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        created = (
            await api_client.post(
                "/api/v1/agent/research",
                json={
                    "query": "What targets are associated with Alzheimer disease?",
                    "execute_inline": True,
                },
            )
        ).json()
        report = (await api_client.get(f"/api/v1/runs/{created['run_id']}/report")).json()
        assert report["validation"]["rejected_claims"] == []
        assert report["markdown"].startswith("#")
        assert report["limitations"]

    async def test_replay_plan_reports_drift(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        created = (
            await api_client.post(
                "/api/v1/agent/research",
                json={"query": "Targets for Alzheimer disease?", "execute_inline": True},
            )
        ).json()
        plan = (await api_client.get(f"/api/v1/runs/{created['run_id']}/replay-plan")).json()
        assert plan["drift"]["reproducible"] is True
        assert plan["query"]

    async def test_unknown_run_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/runs/RUN-1999-00001")
        assert response.status_code == 404

    async def test_comparing_one_run_is_rejected(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/api/v1/runs/compare", json={"run_ids": ["RUN-1"]})
        assert response.status_code == 422

    async def test_usage_summary(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/runs/usage")).json()
        assert "runs" in payload
        assert "tools" in payload


class TestModelEndpoints:
    async def test_model_list_states_the_reporting_convention(
        self, api_client: AsyncClient
    ) -> None:
        payload = (await api_client.get("/api/v1/models")).json()
        assert "scaffold-split" in payload["reporting_note"]

    async def test_unknown_model_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/models/does-not-exist")
        assert response.status_code == 404

    async def test_experiments_list(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/experiments")).json()
        assert payload["count"] == 0


class TestIngestionEndpoints:
    async def test_sources_include_licences_and_citations(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        payload = (await api_client.get("/api/v1/ingestion/sources")).json()
        assert any(source["citation"] for source in payload["sources"])

    async def test_live_ingestion_is_refused_when_disabled(self, api_client: AsyncClient) -> None:
        """The deployed demo serves a pre-ingested corpus and must not call upstream."""
        response = await api_client.post(
            "/api/v1/ingestion/disease", json={"efo_id": "MONDO_0004975", "mode": "live"}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "live_ingestion_disabled"

    async def test_replay_mode_ingestion_is_permitted(self, api_client: AsyncClient) -> None:
        response = await api_client.post(
            "/api/v1/ingestion/features/backfill", params={"limit": 10}
        )
        assert response.status_code == 200
        assert response.json()["task"] == "backfill_features"

    async def test_ingestion_runs_history(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/ingestion/runs")).json()
        assert "runs" in payload


class TestJobEndpoints:
    async def test_job_list_reports_the_registry(self, api_client: AsyncClient) -> None:
        payload = (await api_client.get("/api/v1/jobs")).json()
        assert "train_endpoint" in payload["registered_tasks"]
        assert "queue_stats" in payload

    async def test_unknown_job_is_404(self, api_client: AsyncClient) -> None:
        assert (await api_client.get("/api/v1/jobs/nope")).status_code == 404

    async def test_training_is_enqueued_not_run_inline(self, api_client: AsyncClient) -> None:
        """Training takes minutes; it must never block a request."""
        response = await api_client.post(
            "/api/v1/jobs/train",
            json={"endpoint": "solubility", "dataset_key": "esol", "families": ["random_forest"]},
        )
        # The response is always a job envelope, never a training result -- and a job
        # that fails (no dataset ingested here) must not fail the submission.
        assert response.status_code == 200
        payload = response.json()
        assert payload["task"] == "train_endpoint"
        assert payload["poll"].endswith(payload["job_id"])

        job = (await api_client.get(payload["poll"])).json()
        assert job["status"] in ("pending", "running", "succeeded", "failed")

    async def test_training_rejects_an_unknown_dataset(self, api_client: AsyncClient) -> None:
        response = await api_client.post(
            "/api/v1/jobs/train", json={"endpoint": "solubility", "dataset_key": "nope"}
        )
        assert response.status_code == 422


class TestSecurityEnforcement:
    """The gates themselves, not just the helpers behind them.

    Every other test in this file runs with ``auth_enabled=False`` and rate limiting
    off, which is right for exercising the science. That leaves the two controls a
    public deployment depends on untested, and an auth gate nobody has ever seen refuse
    a request is how a deployment ends up open.
    """

    async def test_unauthenticated_request_is_refused_when_auth_is_enabled(
        self, secured_client_factory: Any
    ) -> None:
        client = await secured_client_factory(auth_enabled=True, api_keys=["test-key-one"])

        response = await client.get("/api/v1/search", params={"q": "alzheimer"})

        assert response.status_code == 401
        body = response.json()
        # The message must say how to authenticate, not merely that you did not.
        assert "X-API-Key" in body["error"]["message"]

    async def test_a_valid_api_key_is_accepted(self, secured_client_factory: Any) -> None:
        client = await secured_client_factory(auth_enabled=True, api_keys=["test-key-one"])

        response = await client.get(
            "/api/v1/search",
            params={"q": "alzheimer"},
            headers={"X-API-Key": "test-key-one"},
        )

        assert response.status_code == 200

    async def test_a_wrong_api_key_is_refused(self, secured_client_factory: Any) -> None:
        client = await secured_client_factory(auth_enabled=True, api_keys=["test-key-one"])

        response = await client.get(
            "/api/v1/search",
            params={"q": "alzheimer"},
            headers={"X-API-Key": "test-key-two"},
        )

        assert response.status_code == 401

    async def test_a_valid_bearer_token_is_accepted(self, secured_client_factory: Any) -> None:
        from theragraph.core.security import issue_token

        secret = "test-secret-long-enough-to-be-plausible"
        client = await secured_client_factory(auth_enabled=True, jwt_secret=secret)
        token = issue_token("researcher@example.test", secret=secret)

        response = await client.get(
            "/api/v1/search",
            params={"q": "alzheimer"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200

    async def test_a_token_signed_with_another_secret_is_refused(
        self, secured_client_factory: Any
    ) -> None:
        from theragraph.core.security import issue_token

        client = await secured_client_factory(auth_enabled=True, jwt_secret="the-real-secret-value")
        forged = issue_token("attacker@example.test", secret="a-different-secret")

        response = await client.get(
            "/api/v1/search",
            params={"q": "alzheimer"},
            headers={"Authorization": f"Bearer {forged}"},
        )

        assert response.status_code == 401

    async def test_liveness_stays_open_when_auth_is_enabled(
        self, secured_client_factory: Any
    ) -> None:
        """A probe that needs a credential is a probe that reports the wrong thing."""
        client = await secured_client_factory(auth_enabled=True, api_keys=["test-key-one"])

        assert (await client.get("/healthz")).status_code == 200

    async def test_rate_limit_throttles_and_says_when_to_retry(
        self, secured_client_factory: Any
    ) -> None:
        client = await secured_client_factory(
            rate_limit_enabled=True, rate_limit_requests=3, rate_limit_window_s=60
        )

        statuses = [
            (await client.get("/api/v1/search", params={"q": "alzheimer"})).status_code
            for _ in range(5)
        ]

        assert statuses[:3] == [200, 200, 200]
        assert statuses[3:] == [429, 429]

        throttled = await client.get("/api/v1/search", params={"q": "alzheimer"})
        assert throttled.headers["Retry-After"]
        assert throttled.headers["X-RateLimit-Remaining"] == "0"
