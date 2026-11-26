from __future__ import annotations

import json
import sys
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from sciintel.semantic import paper_evidence
from sciintel.semantic.auto_topic_discovery import OllamaTopicLabeler, build_profiles


class PaperEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_non_doi_document_id_is_not_misclassified_as_doi(self) -> None:
        self.assertEqual(paper_evidence._clean_doi("doc-1"), "")
        self.assertEqual(paper_evidence._identifier_keys("doc-1", "document_id"), ["raw:doc-1"])

    def test_reconstruct_openalex_abstract_orders_words(self) -> None:
        index = {
            "Research": [0], "papers": [1], "study": [2], "stochastic": [3],
            "optimization": [4], "methods": [5], "for": [6], "large": [7],
            "scale": [8], "control": [9], "systems": [10],
        }
        self.assertEqual(
            paper_evidence.reconstruct_openalex_abstract(index),
            "Research papers study stochastic optimization methods for large scale control systems",
        )

    def test_local_abstract_alignment_by_doi_not_row_order(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "doi": "10.1234/one", "title": "Paper One"},
            {"document_id": "doc-2", "doi": "10.1234/two", "title": "Paper Two"},
        ])
        # Deliberately reverse rows to ensure matching happens by DOI, not position.
        pd.DataFrame([
            {"doi": "https://doi.org/10.1234/two", "abstract": "Second paper abstract with enough text to be accepted as a valid abstract entry."},
            {"doi": "10.1234/one", "abstract": "First paper abstract with enough text to be accepted as a valid abstract entry."},
        ]).to_csv(self.root / "data" / "papers.csv", index=False)

        bundle = paper_evidence.collect_paper_evidence(
            self.root, base, ["doc-1", "doc-2"], source="local",
        )
        self.assertIn("First paper abstract", bundle.by_id["doc-1"]["abstract"])
        self.assertIn("Second paper abstract", bundle.by_id["doc-2"]["abstract"])
        self.assertEqual(bundle.stats["documents_with_abstract"], 2)
        self.assertEqual(bundle.text_sources, ["abstract", "abstract"])

    def test_main_pipeline_sqlite_and_processed_text_are_primary_local_sources(self) -> None:
        import sqlite3

        processed = self.root / "data" / "processed"
        docs = {
            "doc-1": "Canonical abstract for document one. " * 5,
            "doc-2": "Canonical abstract for document two. " * 5,
        }
        db_path = self.root / "data" / "silab.sqlite3"
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("""
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    abstract TEXT NOT NULL DEFAULT '',
                    text_path TEXT NOT NULL DEFAULT ''
                )
            """)
            for doc_id, abstract in docs.items():
                folder = processed / doc_id
                folder.mkdir(parents=True)
                text_path = folder / "document.txt"
                text_path.write_text(
                    "Paper title\n\nAbstract\n" + abstract + "\n\n1 Introduction\nFull body.",
                    encoding="utf-8",
                )
                conn.execute(
                    "INSERT INTO documents(id, abstract, text_path) VALUES (?, ?, ?)",
                    (doc_id, abstract, str(text_path)),
                )

        base = pd.DataFrame([
            {"document_id": "doc-1", "title": "Paper One"},
            {"document_id": "doc-2", "title": "Paper Two"},
        ])
        with patch.object(paper_evidence, "_http_json") as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1", "doc-2"], source="auto",
                cache_path=self.root / "cache.json",
            )
        mocked.assert_not_called()
        self.assertEqual(bundle.stats["sqlite_rows_matched"], 2)
        self.assertEqual(bundle.stats["sqlite_abstract_records"], 2)
        self.assertEqual(bundle.stats["processed_text_files_found"], 2)
        self.assertEqual(bundle.stats["documents_with_abstract"], 2)
        self.assertEqual(bundle.stats["documents_with_full_text"], 2)
        self.assertIn("Canonical abstract for document one", bundle.by_id["doc-1"]["abstract"])
        self.assertEqual(bundle.by_id["doc-2"]["source"], "sqlite_abstract")

    def test_processed_text_recovers_abstract_when_sqlite_abstract_is_empty(self) -> None:
        import sqlite3

        folder = self.root / "data" / "processed" / "doc-processed"
        folder.mkdir(parents=True)
        text_path = folder / "document.txt"
        abstract = "This processed PDF abstract contains the scientific objective, methods, and key findings in a sufficiently detailed description."
        text_path.write_text(
            "Paper title\n\nAbstract\n" + abstract + "\n\n1 Introduction\nThe full article begins here.",
            encoding="utf-8",
        )
        with closing(sqlite3.connect(self.root / "data" / "silab.sqlite3")) as conn, conn:
            conn.execute("""
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    abstract TEXT NOT NULL DEFAULT '',
                    text_path TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute(
                "INSERT INTO documents(id, abstract, text_path) VALUES (?, '', ?)",
                ("doc-processed", str(text_path)),
            )
        base = pd.DataFrame([{"document_id": "doc-processed", "title": "Paper title"}])
        bundle = paper_evidence.collect_paper_evidence(
            self.root, base, ["doc-processed"], source="local",
        )
        self.assertTrue(bundle.by_id["doc-processed"]["has_abstract"])
        self.assertEqual(bundle.by_id["doc-processed"]["source"], "sqlite_processed_abstract")
        self.assertIn("scientific objective", bundle.by_id["doc-processed"]["abstract"])
        self.assertEqual(bundle.stats["processed_abstracts_extracted"], 1)

    def test_local_mode_uses_cached_abstracts_without_remote_requests(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-cache", "doi": "10.1234/cached", "title": "Cached paper"},
        ])
        cached_abstract = (
            "A locally cached abstract with sufficient information about the research objective, "
            "methodology, experiments, and principal results."
        )
        cache = self.root / "abstract-cache.json"
        cache.write_text(json.dumps({
            "version": 1,
            "items": {
                "doi:10.1234/cached": {
                    "abstract": cached_abstract,
                    "source": "abstract_crossref",
                    "source_detail": "10.1234/cached",
                }
            },
        }), encoding="utf-8")
        with patch.object(paper_evidence, "_http_json") as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-cache"], source="local", cache_path=cache,
            )
        mocked.assert_not_called()
        self.assertEqual(bundle.stats["cache_hits"], 1)
        self.assertIn("locally cached abstract", bundle.by_id["doc-cache"]["abstract"])

    def test_misaligned_ids_are_rejected(self) -> None:
        base = pd.DataFrame([{"document_id": "doc-1", "title": "Paper One"}])
        with self.assertRaisesRegex(ValueError, "не совпадает"):
            paper_evidence.collect_paper_evidence(self.root, base, ["other-doc"], source="local")

    def test_openalex_batch_retrieval_and_cache(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "openalex_id": "W123", "doi": "10.1234/one", "title": "Paper One"},
        ])
        payload = {
            "results": [{
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/one",
                "ids": {"openalex": "https://openalex.org/W123", "doi": "https://doi.org/10.1234/one"},
                "abstract_inverted_index": {
                    "This": [0], "publication": [1], "studies": [2], "important": [3],
                    "scientific": [4], "questions": [5], "using": [6], "robust": [7],
                    "statistical": [8], "methods": [9], "and": [10], "reproducible": [11], "experiments": [12],
                },
            }]
        }
        cache = self.root / "abstract_cache.json"
        with patch.object(paper_evidence, "_http_json", return_value=payload) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=cache,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(bundle.stats["openalex_batches"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 1)
        self.assertIn("scientific questions", bundle.by_id["doc-1"]["abstract"])
        saved = json.loads(cache.read_text(encoding="utf-8"))
        self.assertIn("doi:10.1234/one", saved["items"])

    def test_openalex_title_lookup_requires_strong_title_match(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "title": "A Study of Robust Stochastic Optimization for Networked Control Systems"},
        ])
        payload = {
            "results": [{
                "id": "https://openalex.org/W456",
                "title": "A Study of Robust Stochastic Optimization for Networked Control Systems",
                "abstract_inverted_index": {
                    "This": [0], "paper": [1], "studies": [2], "robust": [3], "stochastic": [4],
                    "optimization": [5], "for": [6], "networked": [7], "control": [8], "systems": [9],
                    "with": [10], "new": [11], "mathematical": [12], "guarantees": [13],
                },
            }]
        }
        with patch.object(paper_evidence, "_http_json", return_value=payload) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=self.root / "title-cache.json",
                max_openalex_title_lookups=1,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(bundle.stats["openalex_title_matches"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 1)
        self.assertEqual(bundle.by_id["doc-1"]["source"], "abstract_openalex_title")

    def test_negative_title_cache_skips_fresh_no_match(self) -> None:
        from datetime import datetime, timezone

        title = "A Publication Without a DOI"
        key = paper_evidence._title_cache_key(title)
        cache = self.root / "title-cache.json"
        cache.write_text(json.dumps({"version": 1, "items": {
            key: {"status": "no_abstract_match", "checked_at": datetime.now(timezone.utc).isoformat()}
        }}), encoding="utf-8")
        base = pd.DataFrame([{"document_id": "doc-1", "title": title}])
        with patch.object(paper_evidence, "_http_json") as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=cache,
                max_openalex_title_lookups=1,
            )
        mocked.assert_not_called()
        self.assertEqual(bundle.stats["openalex_title_cache_negative_hits"], 1)

    def test_negative_title_cache_expires_after_90_days(self) -> None:
        from datetime import datetime, timedelta, timezone

        title = "A Publication Without a DOI"
        key = paper_evidence._title_cache_key(title)
        cache = self.root / "title-cache.json"
        old = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
        cache.write_text(json.dumps({"version": 1, "items": {
            key: {"status": "no_abstract_match", "checked_at": old}
        }}), encoding="utf-8")
        base = pd.DataFrame([{"document_id": "doc-1", "title": title}])
        payload = {"results": []}
        with patch.object(paper_evidence, "_http_json", return_value=payload) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=cache,
                max_openalex_title_lookups=1,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(bundle.stats["openalex_title_lookups"], 1)

    def test_crossref_stops_only_after_consecutive_failures(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "doi": "10.1234/one", "title": "Paper One"},
            {"document_id": "doc-2", "doi": "10.1234/two", "title": "Paper Two"},
            {"document_id": "doc-3", "doi": "10.1234/three", "title": "Paper Three"},
        ])
        responses = [
            {"results": []},
            RuntimeError("temporary error"),
            {"message": {"title": ["Paper Two"]}},  # successful response, no abstract
            RuntimeError("temporary error"),
        ]
        with patch.object(paper_evidence, "_http_json", side_effect=responses) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1", "doc-2", "doc-3"], source="openalex",
                cache_path=self.root / "cache.json", max_crossref_lookups=3,
            )
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(bundle.stats["crossref_requests"], 3)
        self.assertEqual(len(bundle.stats["remote_errors"]), 2)

    def test_openalex_title_lookup_rejects_weak_match(self) -> None:
        base = pd.DataFrame([{"document_id": "doc-1", "title": "A Study of Robust Stochastic Optimization"}])
        payload = {"results": [{
            "id": "https://openalex.org/W999", "title": "A Survey of Deep Learning for Medical Images",
            "abstract_inverted_index": {"A": [0], "sufficiently": [1], "long": [2], "abstract": [3], "for": [4], "testing": [5], "matching": [6], "logic": [7]},
        }]}
        with patch.object(paper_evidence, "_http_json", return_value=payload):
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=self.root / "title-cache.json",
                max_openalex_title_lookups=1,
            )
        self.assertEqual(bundle.stats["openalex_title_rejected"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 0)

    def test_title_lookup_budget_is_round_robin_across_domains(self) -> None:
        import urllib.parse

        base = pd.DataFrame([
            {"document_id": "a1", "title": "Alpha domain paper one", "domain": "alpha"},
            {"document_id": "a2", "title": "Alpha domain paper two", "domain": "alpha"},
            {"document_id": "b1", "title": "Beta domain paper one", "domain": "beta"},
            {"document_id": "b2", "title": "Beta domain paper two", "domain": "beta"},
        ])
        queried_titles = []

        def fake_lookup(url, *_args, **_kwargs):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("search", [""])[0]
            queried_titles.append(query)
            return {"results": []}

        with patch.object(paper_evidence, "_http_json", side_effect=fake_lookup):
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["a1", "a2", "b1", "b2"], source="openalex",
                cache_path=self.root / "cache.json", max_openalex_title_lookups=2,
            )
        self.assertEqual(len(queried_titles), 2)
        self.assertTrue(any("Alpha" in title for title in queried_titles))
        self.assertTrue(any("Beta" in title for title in queried_titles))
        self.assertEqual(bundle.stats["openalex_title_selection"], "round_robin_by_domain")

    def test_openalex_failure_does_not_disable_crossref_doi_fallback(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "openalex_id": "W123", "doi": "10.1234/one", "title": "Paper One"},
        ])
        cache = self.root / "abstract_cache.json"
        abstract = "Crossref abstract survives an independent OpenAlex rate-limit or network failure and is long enough for matching."
        with patch.object(
            paper_evidence, "_http_json",
            side_effect=[RuntimeError("OpenAlex HTTP 429"), {"message": {"abstract": abstract}}],
        ) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=cache,
                max_crossref_lookups=10,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(bundle.stats["crossref_requests"], 1)
        self.assertEqual(bundle.stats["crossref_abstracts"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 1)
        self.assertTrue(bundle.stats["remote_errors"])

    def test_crossref_title_search_fills_record_after_openalex_rate_limit(self) -> None:
        import urllib.parse

        title = "A Precise Study of Robust Stochastic Optimization Methods"
        abstract = "This abstract describes robust stochastic optimization, the evaluation method, and the principal results in sufficient detail for semantic matching."
        base = pd.DataFrame([{"document_id": "doc-title", "title": title, "domain": "optimization"}])

        def fake_lookup(url, *_args, **_kwargs):
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            if parsed.netloc == "api.openalex.org":
                raise RuntimeError("HTTP 429: daily budget exhausted")
            self.assertEqual(params.get("query.bibliographic", [""])[0], title)
            return {"message": {"items": [{"DOI": "10.5555/example", "title": [title], "abstract": abstract}]}}

        with patch.object(paper_evidence, "_http_json", side_effect=fake_lookup) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-title"], source="openalex",
                cache_path=self.root / "cache-crossref-title.json",
                max_openalex_title_lookups=1, max_crossref_title_lookups=1,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(bundle.stats["crossref_title_lookups"], 1)
        self.assertEqual(bundle.stats["crossref_title_matches"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 1)
        self.assertEqual(bundle.by_id["doc-title"]["source"], "abstract_crossref_title")

    def test_crossref_title_search_rejects_unrelated_record(self) -> None:
        import urllib.parse

        title = "A Study of Robust Stochastic Optimization"
        base = pd.DataFrame([{"document_id": "doc-title", "title": title}])

        def fake_lookup(url, *_args, **_kwargs):
            if urllib.parse.urlparse(url).netloc == "api.openalex.org":
                raise RuntimeError("HTTP 429")
            return {"message": {"items": [{
                "DOI": "10.5555/unrelated",
                "title": ["A Survey of Deep Learning for Medical Image Segmentation"],
                "abstract": "This unrelated abstract is sufficiently long but must never be attached to the source paper.",
            }]}}

        with patch.object(paper_evidence, "_http_json", side_effect=fake_lookup):
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-title"], source="openalex",
                cache_path=self.root / "cache-crossref-reject.json",
                max_openalex_title_lookups=1, max_crossref_title_lookups=1,
            )
        self.assertEqual(bundle.stats["crossref_title_rejected"], 1)
        self.assertEqual(bundle.stats["documents_with_abstract"], 0)

    def test_crossref_fallback_fills_missing_doi(self) -> None:
        base = pd.DataFrame([
            {"document_id": "doc-1", "doi": "10.1234/one", "title": "Paper One"},
        ])
        responses = [
            {"results": []},
            {"message": {"abstract": "Crossref abstract provides a sufficiently long description of the publication and its main scientific method."}},
        ]
        with patch.object(paper_evidence, "_http_json", side_effect=responses) as mocked:
            bundle = paper_evidence.collect_paper_evidence(
                self.root, base, ["doc-1"], source="openalex", cache_path=self.root / "cache.json",
                max_crossref_lookups=1,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(bundle.stats["crossref_abstracts"], 1)
        self.assertIn("Crossref abstract", bundle.by_id["doc-1"]["abstract"])



class OllamaLabelerValidationTests(unittest.TestCase):
    def _generation(self, evidence_title: str) -> dict:
        return {
            "name_ru": "Тема научного исследования",
            "name_en": "Scientific Research Topic",
            "description_ru": "Описание темы на основе переданных публикаций.",
            "scope_note_ru": "",
            "research_core_ru": "Исследовательская задача.",
            "subtopics_ru": ["Подтема"], "methods_ru": ["Метод"],
            "research_objects_ru": ["Объект"],
            "heterogeneity_level": "low", "heterogeneity_note_ru": "",
            "evidence_titles": [evidence_title],
        }


    @staticmethod
    def _final_check(**overrides) -> dict:
        result = {
            "decision": "accepted",
            "name_support_level": "high",
            "profile_support_level": "high",
            "unsupported_name_claims": [],
            "unsupported_profile_claims": [],
            "missing_directions": [],
            "review_note_ru": "Финальная версия подтверждена тестовыми evidence.",
        }
        result.update(overrides)
        return result

    def test_chat_json_retries_truncated_response_with_larger_token_budget(self) -> None:
        labeler = OllamaTopicLabeler("http://127.0.0.1:11434", "test-model")
        calls = []
        responses = [
            {"message": {"content": '{"label": "cut'} , "done": True, "done_reason": "length", "eval_count": 500},
            {"message": {"content": '{"label": "complete"}'}, "done": True, "done_reason": "stop", "eval_count": 10},
        ]

        def fake_request(_path, payload):
            calls.append(payload)
            return responses.pop(0)

        labeler._request = fake_request
        schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
        result = labeler._chat_json("system", "return JSON", schema, num_predict=500)
        self.assertEqual(result["label"], "complete")
        self.assertEqual(calls[0]["options"]["num_predict"], 500)
        self.assertEqual(calls[1]["options"]["num_predict"], 1200)
        self.assertEqual(labeler.runtime_stats()["ollama_runtime"]["retries"], 1)
        self.assertEqual(labeler.runtime_stats()["ollama_runtime"]["invalid_json_responses"], 1)

    def test_compact_evidence_keeps_abstracts_before_title_only_papers(self) -> None:
        evidence = {
            "community_size": 100,
            "candidate_keyphrases": [f"term {i}" for i in range(30)],
            "representative_papers": [
                {"title": f"Title only {i}", "domain": "x", "abstract_or_text_excerpt": ""}
                for i in range(7)
            ] + [
                {"title": "Annotated paper", "domain": "y", "abstract_or_text_excerpt": "Abstract evidence."}
            ],
        }
        compact = OllamaTopicLabeler._compact_evidence(evidence)
        self.assertEqual(compact["representative_papers"][0]["title"], "Annotated paper")
        self.assertEqual(compact["representative_abstract_count"], 1)
        self.assertEqual(len(compact["candidate_keyphrases"]), 14)

    def test_exact_long_supporting_title_is_preserved(self) -> None:
        long_title = ("A Very Long Scientific Publication Title " * 8).strip()
        labeler = OllamaTopicLabeler("http://127.0.0.1:11434", "test-model")
        responses = [
            self._generation(long_title),
            {
                "decision": "accepted", "support_level": "high", "unsupported_claims": [],
                "missing_directions": [], "review_note_ru": "Подтверждено.",
                "heterogeneity_level": "low", "heterogeneity_note_ru": "",
                "revised_name_ru": "", "revised_name_en": "", "revised_description_ru": "", "revised_scope_note_ru": "",
                "revised_research_core_ru": "", "revised_subtopics_ru": [], "revised_methods_ru": [], "revised_research_objects_ru": [],
            },
            self._final_check(),
        ]
        labeler._chat_json = lambda *args, **kwargs: responses.pop(0)
        result = labeler.label({"representative_papers": [{
            "document_id": "doc-long-title", "title": long_title, "domain": "test-domain",
            "text_source": "abstract", "abstract_or_text_excerpt": "This abstract provides source evidence.",
        }]})
        self.assertEqual(result["evidence_titles"], [long_title])
        self.assertEqual(result["evidence_references"][0]["document_id"], "doc-long-title")
        self.assertEqual(result["validation"]["decision"], "accepted")
        self.assertEqual(result["validation"]["final_validation_decision"], "accepted")
        self.assertEqual(result["final_validation"]["status"], "completed")

    def test_unsupported_claim_demotes_accepted_review(self) -> None:
        labeler = OllamaTopicLabeler("http://127.0.0.1:11434", "test-model")
        responses = [
            self._generation("Paper title"),
            {
                "decision": "accepted", "support_level": "high", "unsupported_claims": ["Не подтверждённый метод"],
                "missing_directions": [], "review_note_ru": "Есть неподтверждённое утверждение.",
                "heterogeneity_level": "moderate", "heterogeneity_note_ru": "",
                "revised_name_ru": "", "revised_name_en": "", "revised_description_ru": "", "revised_scope_note_ru": "",
                "revised_research_core_ru": "", "revised_subtopics_ru": [], "revised_methods_ru": [], "revised_research_objects_ru": [],
            },
            self._final_check(
                decision="review_required", name_support_level="medium",
                unsupported_profile_claims=["Не подтверждённый метод"],
                review_note_ru="Неподтверждённый метод не должен остаться в профиле.",
            ),
        ]
        labeler._chat_json = lambda *args, **kwargs: responses.pop(0)
        result = labeler.label({"representative_papers": [{"title": "Paper title"}]})
        self.assertEqual(result["validation"]["decision"], "review_required")
        self.assertEqual(result["validation"]["support_level"], "medium")
        self.assertEqual(result["validation"]["final_validation_decision"], "review_required")
        self.assertIn("Не подтверждённый метод", result["validation"]["unsupported_claims"])


    def test_revision_replaces_all_profile_fields_and_final_check_accepts_clean_result(self) -> None:
        labeler = OllamaTopicLabeler("http://127.0.0.1:11434", "test-model")
        generation = self._generation("Paper title")
        generation.update({
            "name_ru": "Слишком узкая тема одного метода",
            "research_core_ru": "Неподтверждённое ядро.",
            "subtopics_ru": ["Неподтверждённая подтема"],
            "methods_ru": ["Неподтверждённый метод"],
            "research_objects_ru": ["Неподтверждённый объект"],
        })
        revision = {
            "decision": "revised", "support_level": "high",
            "unsupported_claims": ["Неподтверждённый метод", "Неподтверждённый объект"],
            "missing_directions": ["В названии не отражено общее ядро"],
            "review_note_ru": "Сузить профиль до подтверждённого содержания.",
            "heterogeneity_level": "moderate", "heterogeneity_note_ru": "Два устойчивых аспекта.",
            "revised_name_ru": "Общая подтверждённая исследовательская тема",
            "revised_name_en": "Shared Evidence-Based Research Topic",
            "revised_description_ru": "Описание без неподтверждённых деталей.",
            "revised_scope_note_ru": "Границы определены предоставленными аннотациями.",
            "revised_research_core_ru": "Подтверждённое исследовательское ядро.",
            "revised_subtopics_ru": ["Подтверждённая подтема"],
            "revised_methods_ru": ["Подтверждённый метод"],
            "revised_research_objects_ru": ["Подтверждённый объект"],
        }
        responses = [generation, revision, self._final_check()]
        labeler._chat_json = lambda *args, **kwargs: responses.pop(0)
        result = labeler.label({"representative_papers": [{"title": "Paper title", "abstract_or_text_excerpt": "Evidence."}]})
        self.assertEqual(result["name_ru"], "Общая подтверждённая исследовательская тема")
        self.assertEqual(result["research_core_ru"], "Подтверждённое исследовательское ядро.")
        self.assertEqual(result["subtopics_ru"], ["Подтверждённая подтема"])
        self.assertEqual(result["methods_ru"], ["Подтверждённый метод"])
        self.assertEqual(result["research_objects_ru"], ["Подтверждённый объект"])
        self.assertEqual(result["validation"]["initial_unsupported_claims"], ["Неподтверждённый метод", "Неподтверждённый объект"])
        self.assertEqual(result["validation"]["unsupported_claims"], [])
        self.assertEqual(result["validation"]["revision_status"], "revised")
        self.assertEqual(result["validation"]["final_validation_decision"], "accepted")

    def test_final_validation_does_not_accept_remaining_unsupported_title_claim(self) -> None:
        labeler = OllamaTopicLabeler("http://127.0.0.1:11434", "test-model")
        revision = {
            "decision": "revised", "support_level": "high", "unsupported_claims": [],
            "missing_directions": [], "review_note_ru": "Исправлено.",
            "heterogeneity_level": "low", "heterogeneity_note_ru": "",
            "revised_name_ru": "Название темы", "revised_name_en": "Topic Name",
            "revised_description_ru": "Описание.", "revised_scope_note_ru": "",
            "revised_research_core_ru": "Ядро.", "revised_subtopics_ru": [],
            "revised_methods_ru": [], "revised_research_objects_ru": [],
        }
        final_check = self._final_check(
            decision="review_required", name_support_level="low",
            unsupported_name_claims=["Узкое предметное уточнение не подтверждено"],
            review_note_ru="Название остаётся чрезмерно узким.",
        )
        responses = [self._generation("Paper title"), revision, final_check]
        labeler._chat_json = lambda *args, **kwargs: responses.pop(0)
        result = labeler.label({"representative_papers": [{"title": "Paper title", "abstract_or_text_excerpt": "Evidence."}]})
        self.assertEqual(result["validation"]["final_validation_decision"], "review_required")
        self.assertIn("Узкое предметное уточнение не подтверждено", result["validation"]["unsupported_claims"])


class FakeLabeler:
    model = "fake-test-model"

    def label(self, evidence: dict) -> dict:
        titles = [p["title"] for p in evidence["representative_papers"][:2]]
        return {
            "name_en": "Test Topic",
            "name_ru": "Тестовая научная тема",
            "description_ru": "Описание темы, сформированное для проверки конвейера.",
            "scope_note_ru": "",
            "research_core_ru": "Исследовательское ядро тестовой группы.",
            "subtopics_ru": ["Подтема один", "Подтема два"],
            "methods_ru": ["Метод анализа"],
            "research_objects_ru": ["Объект исследования"],
            "heterogeneity_level": "moderate",
            "heterogeneity_note_ru": "Есть несколько близких направлений.",
            "evidence_titles": titles,
            "validation": {
                "review_status": "completed", "decision": "accepted", "support_level": "high",
                "unsupported_claims": [], "missing_directions": [], "review_note_ru": "Подтверждено тестом.",
            },
            "generation_draft": {},
        }


class FailingLabeler:
    model = "failing-test-model"

    def label(self, evidence: dict) -> dict:
        raise TimeoutError("synthetic Ollama timeout")


class SemanticProfileTests(unittest.TestCase):
    def _sample_data(self):
        rng = np.random.default_rng(13)
        centers = np.eye(2, 8, dtype=np.float32)
        vectors = []
        labels = []
        titles = []
        domains = []
        texts = []
        ids = []
        evidence = {}
        for cid, (term, domain) in enumerate((("stochastic optimization gradient descent", "optimization"), ("robot manipulation motion planning", "robotics"))):
            for j in range(6):
                v = centers[cid] + rng.normal(0, 0.025, size=8).astype(np.float32)
                v /= np.linalg.norm(v)
                vectors.append(v)
                labels.append(cid)
                doc_id = f"doc-{cid}-{j}"
                ids.append(doc_id)
                titles.append(f"Research on {term} {j}")
                domains.append(domain)
                texts.append((term + " " + term + " scientific paper methods results") * 2)
                has_abstract = j < (4 if cid == 0 else 1)
                evidence[doc_id] = {
                    "abstract": "This abstract presents a scientific method and results for this research topic." if has_abstract else "",
                    "text_excerpt": "This abstract presents a scientific method and results for this research topic." if has_abstract else "",
                    "source": "abstract_openalex" if has_abstract else "title_only",
                    "source_detail": "test" if has_abstract else "",
                    "has_abstract": has_abstract,
                    "has_full_text": False,
                }
        return np.asarray(labels), np.asarray(vectors), titles, domains, texts, ids, evidence

    def test_build_profiles_emits_evidence_and_review_fields(self) -> None:
        labels, vectors, titles, domains, texts, ids, evidence = self._sample_data()
        profiles, centroids, reps, audit = build_profiles(
            labels, vectors, titles, domains, texts, ids, "en",
            labeler=FakeLabeler(), text_sources=["abstract_openalex"] * len(ids),
            paper_evidence_by_id=evidence,
        )
        self.assertEqual(len(profiles), 2)
        self.assertEqual(set(profiles.semantic_review_decision), {"accepted"})
        self.assertEqual(profiles.loc[profiles.cluster_id == 0, "semantic_abstract_coverage_percent"].iloc[0], 100.0 * 4 / 6)
        self.assertEqual(profiles.loc[profiles.cluster_id == 1, "semantic_abstract_coverage_percent"].iloc[0], 100.0 / 6)
        self.assertIn("Тестовая научная тема", profiles.semantic_name_ru.tolist())
        self.assertTrue(profiles.semantic_supporting_titles.str.len().gt(0).all())
        self.assertEqual(set(profiles.semantic_final_validation_decision), {"accepted"})
        self.assertEqual(set(profiles.semantic_name_support_level), {"high"})
        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(reps), 2)
        self.assertEqual(len(audit), 2)

    def test_model_failure_uses_explicit_fallback_without_losing_topic(self) -> None:
        labels, vectors, titles, domains, texts, ids, evidence = self._sample_data()
        profiles, _centroids, _reps, audit = build_profiles(
            labels, vectors, titles, domains, texts, ids, "en",
            labeler=FailingLabeler(), text_sources=["title_only"] * len(ids),
            paper_evidence_by_id=evidence,
        )
        self.assertEqual(len(profiles), 2)
        self.assertTrue((profiles.semantic_label_status == "generation_failed_fallback").all())
        self.assertTrue((profiles.semantic_review_decision == "review_required").all())
        self.assertTrue(all(record.get("error") for record in audit))


class ExistingResultRelabelIntegrationTests(unittest.TestCase):
    def test_relabel_creates_separate_snapshot_and_preserves_assignments_and_projection(self) -> None:
        import contextlib
        import io
        import os
        import sys
        from sciintel.semantic import label_existing_topics
        from sciintel.semantic.paper_evidence import EvidenceBundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "graph-test"
            run_dir = root / "artifacts" / "runs" / run_id
            run_dir.mkdir(parents=True)
            emb_dir = root / "artifacts" / "embeddings" / "sentence-transformers_allenai-specter"
            emb_dir.mkdir(parents=True)
            docs = []
            vectors = []
            assignments = []
            projection = []
            evidence_by_id = {}
            texts = []
            sources = []
            rng = np.random.default_rng(9)
            for cid, (term, domain) in enumerate((("stochastic optimization gradient", "optimization"), ("robot manipulation planning", "robotics"))):
                center = np.zeros(8, dtype=np.float32)
                center[cid] = 1.0
                for j in range(4):
                    doc_id = f"paper-{cid}-{j}"
                    vector = center + rng.normal(0, 0.01, size=8).astype(np.float32)
                    vector /= np.linalg.norm(vector)
                    vectors.append(vector)
                    title = f"A scientific study of {term} number {j}"
                    docs.append({"document_id": doc_id, "title": title, "domain": domain})
                    assignments.append({"document_id": doc_id, "title": title, "domain": domain, "cluster_id": cid, "semantic_name_ru": "old", "cosine_to_own_centroid": 0.95})
                    projection.append({"document_id": doc_id, "title": title, "cluster_id": cid, "x": float(cid * 10 + j), "y": float(j), "semantic_name_ru": "old"})
                    has_abstract = j < 2
                    abstract = "This paper presents a detailed scientific abstract with method, experiment and results for the topic." if has_abstract else ""
                    evidence_by_id[doc_id] = {
                        "abstract": abstract, "text_excerpt": abstract,
                        "source": "abstract_openalex" if has_abstract else "title_only",
                        "source_detail": "mock", "has_abstract": has_abstract, "has_full_text": False,
                    }
                    texts.append((title + ". " + abstract).strip())
                    sources.append("abstract_openalex" if has_abstract else "title_only")

            pd.DataFrame(docs).to_csv(run_dir / "clusters.csv", index=False, encoding="utf-8-sig")
            np.save(emb_dir / "vectors.npy", np.asarray(vectors, dtype=np.float32))
            (emb_dir / "manifest.json").write_text(json.dumps({"ids": [x["document_id"] for x in docs]}), encoding="utf-8")

            result_base = root / "reports" / "auto_topic_discovery" / run_id
            source_dir = result_base / "20261126_030905"
            source_dir.mkdir(parents=True)
            profiles = pd.DataFrame([
                {"cluster_id": cid, "documents": 4, "share_percent": 50.0, "semantic_name_ru": "old", "semantic_name": "Old", "keywords": "test", "keywords_yake": "test", "keywords_rake": "test", "keywords_tfidf_contrast": "test", "dominant_domains": f"{domain}:4", "mean_cosine_to_centroid": 0.99, "mean_pairwise_cosine": 0.99}
                for cid, domain in enumerate(("optimization", "robotics"))
            ])
            profiles.to_csv(source_dir / "topic_profiles.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(assignments).to_csv(source_dir / "document_assignments.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(projection).to_csv(source_dir / "projection_2d.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(assignments).to_csv(source_dir / "boundary_documents.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"resolution": 1.0, "selection_score": 0.5}]).to_csv(source_dir / "resolution_sweep.csv", index=False, encoding="utf-8-sig")
            (source_dir / "summary.json").write_text(json.dumps({
                "run_id": run_id, "documents": 8, "detected_communities": 2,
                "chosen_resolution": 1.0, "selected_candidate_metrics": {"stability_ari_mean": 0.9, "silhouette_mean": 0.1},
            }), encoding="utf-8")
            (result_base / "latest.json").write_text(json.dumps({"output_dir": source_dir.name}), encoding="utf-8")

            fake_bundle = EvidenceBundle(
                texts=texts, text_sources=sources, by_id=evidence_by_id, used_files=[],
                stats={
                    "documents": 8, "documents_with_abstract": 4, "documents_with_any_text": 4,
                    "documents_title_only": 4, "openalex_batches": 0, "openalex_title_lookups": 0,
                    "crossref_requests": 0, "remote_errors": [],
                },
            )
            argv = [
                "label_existing_topics.py", "--root", str(root), "--run-id", run_id,
                "--source-result", "latest", "--abstract-source", "local",
            ]
            with patch.object(sys, "argv", argv), \
                 patch.object(label_existing_topics, "collect_paper_evidence", return_value=fake_bundle), \
                 patch.object(label_existing_topics.OllamaTopicLabeler, "verify", return_value={
                     "provider": "Ollama local API", "model": "fake-test-model", "model_digest": "test-digest",
                 }), \
                 patch.object(label_existing_topics.OllamaTopicLabeler, "label", FakeLabeler().label), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(label_existing_topics.main(), 0)

            latest = json.loads((result_base / "latest.json").read_text(encoding="utf-8"))
            self.assertTrue(latest["output_dir"].endswith("_semantic_reviewed"))
            new_dir = result_base / latest["output_dir"]
            self.assertTrue((new_dir / "paper_evidence.csv").is_file())
            self.assertTrue((new_dir / "topic_label_generation.json").is_file())
            new_assignments = pd.read_csv(new_dir / "document_assignments.csv", encoding="utf-8-sig")
            self.assertEqual(new_assignments.cluster_id.tolist(), [x["cluster_id"] for x in assignments])
            self.assertTrue((new_assignments.semantic_name_ru == "Тестовая научная тема").all())
            self.assertTrue((new_assignments.display_name == "Тестовая научная тема").all())
            new_profiles = pd.read_csv(new_dir / "topic_profiles.csv", encoding="utf-8-sig")
            self.assertTrue((new_profiles.display_name == "Тестовая научная тема").all())
            new_projection = pd.read_csv(new_dir / "projection_2d.csv", encoding="utf-8-sig")
            self.assertEqual(new_projection.x.tolist(), [x["x"] for x in projection])
            original_assignments = pd.read_csv(source_dir / "document_assignments.csv", encoding="utf-8-sig")
            self.assertTrue((original_assignments.semantic_name_ru == "old").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
