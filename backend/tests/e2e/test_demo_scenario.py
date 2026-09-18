"""The demo scenario, driven entirely through the HTTP API.

This is the walk-through a recruiter or a scientist would follow:

    disease -> targets -> evidence -> compounds -> similarity -> predictions
            -> comparison -> research agent -> report -> run trace -> replay

If this test passes, the product works end to end over its real interface. It uses the
in-process corpus (no network) and the deterministic agent backend (no API key), so it
runs in CI exactly as it runs on a laptop.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.asyncio, pytest.mark.e2e]


class TestDemoScenario:
    async def test_complete_research_workflow(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        # 1. Enter a disease.
        search = (await api_client.get("/api/v1/search", params={"q": "Alzheimer disease"})).json()
        disease_hit = next(hit for hit in search["hits"] if hit["entity_type"] == "disease")
        assert disease_hit["external_id"] == "MONDO_0004975"

        # 2. Retrieve associated therapeutic targets, ordered by the source's score.
        disease = (await api_client.get(f"/api/v1/diseases/{disease_hit['external_id']}")).json()
        targets = disease["associated_targets"]
        assert len(targets) == 2
        assert (
            targets[0]["association"]["overall_score"] >= targets[1]["association"]["overall_score"]
        )

        # 3. Inspect the evidence for a target, with provenance on every record.
        symbol = targets[0]["target"]["symbol"]
        target = (
            await api_client.get(
                f"/api/v1/targets/{symbol}",
                params={"disease": "MONDO_0004975", "min_pchembl": 0},
            )
        ).json()
        assert target["evidence"]["total"] >= 1
        for group in target["evidence"]["groups"]:
            for record in group.get("records") or group.get("items") or []:
                assert record["provenance"]["source"]
                assert record["provenance"]["retrieved_at"]
                assert record["citation_key"]

        # 4. Retrieve known compounds with measured activity (BACE1 has the chemistry).
        chemistry_target = "BACE1"
        bace = (
            await api_client.get(f"/api/v1/targets/{chemistry_target}", params={"min_pchembl": 0})
        ).json()
        known = bace["known_compounds"]
        assert known
        assert all(compound["basis"].startswith("experimental") for compound in known)

        # 5. Search structurally similar compounds.
        seed_compound = (
            await api_client.get(f"/api/v1/compounds/{known[0]['compound_id']}")
        ).json()
        similarity = (
            await api_client.post(
                "/api/v1/compounds/similarity",
                json={"smiles": seed_compound["compound"]["smiles"], "threshold": 0.3, "limit": 5},
            )
        ).json()
        assert similarity["hits"]
        assert similarity["search_stats"]["corpus_compounds"] >= 4

        # 6. Property predictions are unavailable without a trained model, and the API
        #    says so rather than inventing numbers.
        prediction = await api_client.post(
            "/api/v1/predictions", json={"compound_ids": [known[0]["compound_id"]]}
        )
        assert prediction.status_code == 503
        assert prediction.json()["error"]["code"] == "model_unavailable"

        # 7. Compare candidates with the full factor breakdown.
        ranking = (
            await api_client.post(
                "/api/v1/ranking",
                json={
                    "target": chemistry_target,
                    "disease": "MONDO_0004975",
                    "min_pchembl": 0.0,
                    "run_predictions": False,
                    "limit": 10,
                },
            )
        ).json()
        assert ranking["candidates"]
        leader = ranking["candidates"][0]
        contributions = sum(
            factor["contribution"]
            for factor in leader["factors"]
            if factor["raw_value"] is not None
        )
        assert contributions == pytest.approx(leader["total_score"], abs=1e-6)
        assert "not a prediction of clinical efficacy" in leader["disclaimer"]

        comparison = (
            await api_client.post(
                "/api/v1/ranking/compare",
                json={
                    "compound_ids": [item["compound_id"] for item in ranking["candidates"][:3]],
                    "target": chemistry_target,
                },
            )
        ).json()
        assert len(comparison["candidates"]) >= 2
        assert comparison["highlights"]

        # 8. Run the AI research assistant.
        run = (
            await api_client.post(
                "/api/v1/agent/research",
                json={
                    "query": "Which compounds should we prioritise for Alzheimer disease?",
                    "parameters": {"n_targets": 5},
                    "execute_inline": True,
                },
            )
        ).json()
        assert run["status"] == "succeeded"

        # 9. Generate an evidence-backed report with no unsupported claims.
        report = (await api_client.get(f"/api/v1/runs/{run['run_id']}/report")).json()
        assert report["validation"]["rejected_claims"] == []
        assert report["limitations"]
        assert "Methods and provenance" in report["markdown"]
        for citation in report["citations"]:
            assert citation["source"]
            assert citation["retrieved_at"]

        # 10. Open the research-run page and reproduce the workflow.
        trace = (await api_client.get(f"/api/v1/runs/{run['run_id']}")).json()
        assert trace["reproducibility_hash"]
        assert trace["pins"]["dataset_versions"]
        assert trace["tool_calls"]
        assert trace["prompts"]

        replay_plan = (await api_client.get(f"/api/v1/runs/{run['run_id']}/replay-plan")).json()
        assert replay_plan["drift"]["reproducible"] is True

        replay = (
            await api_client.post(
                f"/api/v1/runs/{run['run_id']}/replay", json={"execute_inline": True}
            )
        ).json()
        assert replay["run_id"] != run["run_id"]
        assert replay["replay_of"] == run["run_id"]

        replayed = (await api_client.get(f"/api/v1/runs/{replay['run_id']}")).json()
        # Same pinned inputs means the same reproducibility hash.
        assert replayed["reproducibility_hash"] == trace["reproducibility_hash"]

        difference = (
            await api_client.post(
                "/api/v1/runs/compare", json={"run_ids": [run["run_id"], replay["run_id"]]}
            )
        ).json()["differences"][0]
        assert difference["same_reproducibility_hash"] is True
        assert "Identical pinned inputs" in difference["verdict"]

    async def test_graph_is_navigable_from_the_disease(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        graph = (
            await api_client.get(
                "/api/v1/graph", params={"disease": "MONDO_0004975", "max_targets": 5}
            )
        ).json()
        kinds = graph["stats"]["node_counts"]
        assert kinds["disease"] == 1
        assert kinds["target"] >= 1
        assert kinds.get("evidence", 0) >= 1
        assert kinds.get("compound", 0) >= 1

        # The graph must be connected: every edge references nodes that exist.
        ids = {node["id"] for node in graph["nodes"]}
        for edge in graph["edges"]:
            assert edge["source"] in ids
            assert edge["target"] in ids

    async def test_a_question_the_corpus_cannot_answer_is_answered_honestly(
        self, api_client: AsyncClient, seeded: dict[str, Any]
    ) -> None:
        """The failure mode that matters most: confident nonsense about missing data."""
        run = (
            await api_client.post(
                "/api/v1/agent/research",
                json={
                    "query": "What targets are associated with Kawasaki disease?",
                    "execute_inline": True,
                },
            )
        ).json()
        assert run["status"] == "succeeded"
        assert run["readiness"]["ready"] is False

        report = (await api_client.get(f"/api/v1/runs/{run['run_id']}/report")).json()
        assert report["citations"] == []
        assert report["validation"]["validated_claims"] == 0
        # Nothing about Kawasaki disease may appear as a retrieved fact.
        assert "Kawasaki" not in " ".join(
            claim["text"] for claim in report["claims"] if claim["kind"] == "retrieved_fact"
        )
