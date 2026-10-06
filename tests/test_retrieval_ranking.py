"""Regression checks for retrieval fusion and reranking inputs."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Doc:
    def __init__(self, content, source):
        self.page_content = content
        self.metadata = {"source": source}


class FakeRanker:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs
        return list(range(len(pairs), 0, -1))


class RetrievalRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {
            name: types.ModuleType(name)
            for name in (
                "Hybrid_Dual_Indexing", "Knowledge_Graph", "Vector_Database",
                "sentence_transformers", "torch", "joblib",
            )
        }
        modules["Hybrid_Dual_Indexing"].Keyword_Search = object
        modules["Hybrid_Dual_Indexing"].Semantic_Search = object
        modules["Knowledge_Graph"].Knowledge_Graphbase = object
        modules["Vector_Database"].Vector_DB = object
        modules["sentence_transformers"].CrossEncoder = object
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "retrieval_ranking_test", Path(__file__).resolve().parents[1] / "Retrieval.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        cls.module = module
        cls.Retriever = module.Retriever

    def setUp(self):
        self.retriever = self.Retriever.__new__(self.Retriever)
        self.retriever.rerank_model = FakeRanker()

    def test_rank_fusion_preserves_distinct_sources(self):
        source_a = Doc("same text", "source-a")
        source_b = Doc("same text", "source-b")
        results = self.retriever.merge_multi_query_retrieval(
            [[source_a, source_b], [source_a]]
        )
        self.assertEqual(results, [source_a, source_b])

    def test_rank_fusion_counts_each_document_once_per_query(self):
        first = Doc("first result", "SympScan")
        repeated = Doc("repeated result", "SympScan")
        result = self.retriever.merge_multi_query_retrieval(
            [[first, repeated, repeated, repeated]]
        )
        self.assertEqual(result, [first, repeated])

        across_queries = self.retriever.merge_multi_query_retrieval(
            [[first, repeated, repeated], [repeated]]
        )
        self.assertEqual(across_queries, [repeated, first])

    def test_hybrid_reranking_removes_duplicate_candidates(self):
        source_a = Doc("same text", "source-a")
        source_b = Doc("same text", "source-b")
        results = self.retriever.merge_hybrid_query_retrieval(
            "query", [source_a], [source_a, source_b]
        )
        self.assertEqual(results, [source_a, source_b])
        self.assertEqual(self.retriever.rerank_model.pairs, [
            ["query", "same text"], ["query", "same text"]
        ])

    def test_graph_reranking_uses_query_first(self):
        self.retriever.merge_multi_subgraph_cross_encoder(
            ["graph context"], "user query"
        )
        self.assertEqual(self.retriever.rerank_model.pairs, [
            ["user query", "graph context"]
        ])

    def test_model_loader_restores_torch_after_success(self):
        class Storage:
            def __new__(cls, *args, **kwargs):
                calls.append(kwargs.get("device"))
                return object.__new__(cls)

        class Model:
            def to(self, device):
                self.device = device

        calls = []
        original_new = Storage.__new__
        torch = self.module.torch
        torch.UntypedStorage = Storage
        torch.serialization = types.SimpleNamespace()
        model = Model()

        def load(_path):
            Storage(device="mps")
            return model

        self.module.joblib.load = load
        self.assertIs(self.Retriever.load_semantic_model(), model)
        self.assertEqual(calls, ["cpu"])
        self.assertEqual(model.device, "cpu")
        self.assertIs(Storage.__new__, original_new)
        self.assertFalse(hasattr(torch.serialization, "_mps_deserialize"))

    def test_model_loader_restores_torch_after_failure(self):
        class Storage:
            pass

        original_new = Storage.__new__
        torch = self.module.torch
        torch.UntypedStorage = Storage
        existing_deserializer = object()
        torch.serialization = types.SimpleNamespace(_mps_deserialize=existing_deserializer)
        self.module.joblib.load = lambda _path: (_ for _ in ()).throw(ValueError("bad model"))

        with self.assertRaises(FileNotFoundError):
            self.Retriever.load_semantic_model("missing-semantic-model.pkl")
        self.assertIs(Storage.__new__, original_new)
        self.assertIs(torch.serialization._mps_deserialize, existing_deserializer)

    def test_graph_restore_requires_explicit_opt_in(self):
        restored = []

        class FakeGraph:
            def load_local(self):
                restored.append(True)

        with (
            patch.object(self.Retriever, "load_semantic_model", return_value=object()),
            patch.object(self.module, "CrossEncoder", return_value=object()),
            patch.object(self.module, "Vector_DB", return_value=object()),
            patch.object(self.module, "Knowledge_Graphbase", side_effect=FakeGraph),
            patch.object(self.module.joblib, "load", return_value=object(), create=True),
        ):
            with patch.dict(os.environ, {"RESTORE_GRAPH_SNAPSHOT": "false"}):
                self.Retriever()
            self.assertEqual(restored, [])

            with patch.dict(os.environ, {"RESTORE_GRAPH_SNAPSHOT": "true"}):
                self.Retriever()
            self.assertEqual(restored, [True])

            self.Retriever(restore_graph_snapshot=False)
            self.assertEqual(restored, [True])

    def test_semantic_results_survive_keyword_failure(self):
        passage = Doc("semantic passage", "SympScan")

        def failed_search(*_args):
            raise RuntimeError("index unavailable")

        self.retriever.inverted_index = types.SimpleNamespace(search=failed_search)
        self.retriever.vector_database = types.SimpleNamespace(search=lambda *_args: [passage])
        with patch("logging.getLogger"):
            results, warnings = self.retriever.hybrid_retrieval(
                ["query"], "query", True, True, False, False, return_warnings=True
            )
        self.assertEqual(results, [passage])
        self.assertEqual(len(warnings), 1)
        self.assertIn("keyword searches failed", warnings[0])

    def test_keyword_results_survive_semantic_failure_and_legacy_api(self):
        passage = Doc("keyword passage", "SympScan")

        def failed_search(*_args):
            raise RuntimeError("index unavailable")

        self.retriever.inverted_index = types.SimpleNamespace(search=lambda *_args: [passage])
        self.retriever.vector_database = types.SimpleNamespace(search=failed_search)
        with patch("logging.getLogger"):
            results, warnings = self.retriever.hybrid_retrieval(
                ["query"], "query", True, True, False, False, return_warnings=True
            )
            legacy_results = self.retriever.hybrid_retrieval(
                ["query"], "query", True, True, False, False
            )
        self.assertEqual(results, [passage])
        self.assertEqual(legacy_results, [passage])
        self.assertIn("semantic searches failed", warnings[0])

    def test_both_search_failures_return_no_evidence_with_warnings(self):
        def failed_search(*_args):
            raise RuntimeError("index unavailable")

        self.retriever.inverted_index = types.SimpleNamespace(search=failed_search)
        self.retriever.vector_database = types.SimpleNamespace(search=failed_search)
        with patch("logging.getLogger"):
            results, warnings = self.retriever.hybrid_retrieval(
                ["first", "second"], "query", True, True, False, True, return_warnings=True
            )
        self.assertEqual(results, [])
        self.assertEqual(len(warnings), 2)

    def test_reranking_failure_falls_back_to_unique_interleaving(self):
        shared = Doc("shared", "SympScan")
        semantic = Doc("semantic", "SympScan")
        self.retriever.inverted_index = types.SimpleNamespace(search=lambda *_args: [shared])
        self.retriever.vector_database = types.SimpleNamespace(search=lambda *_args: [shared, semantic])
        self.retriever.rerank_model = types.SimpleNamespace(
            predict=lambda *_args: (_ for _ in ()).throw(RuntimeError("ranker unavailable"))
        )
        with patch("logging.getLogger"):
            results, warnings = self.retriever.hybrid_retrieval(
                ["query"], "query", True, True, False, True, return_warnings=True
            )
        self.assertEqual(results, [shared, semantic])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Reranking was unavailable", warnings[0])

    def test_invalid_passage_reranking_scores_use_fallback(self):
        keyword = Doc("keyword passage", "SympScan")
        semantic = Doc("semantic passage", "SympScan")
        self.retriever.inverted_index = types.SimpleNamespace(search=lambda *_args: [keyword])
        self.retriever.vector_database = types.SimpleNamespace(search=lambda *_args: [semantic])

        for scores in ([], [0.8], [float("nan"), 0.2], [0.8, float("inf")]):
            with self.subTest(scores=scores), patch("logging.getLogger"):
                self.retriever.rerank_model = types.SimpleNamespace(predict=lambda *_args: scores)
                results, warnings = self.retriever.hybrid_retrieval(
                    ["query"], "query", True, True, False, True, return_warnings=True
                )
                self.assertEqual(results, [keyword, semantic])
                self.assertIn("Reranking was unavailable", warnings[0])

    def test_fallback_limits_combined_results_to_five(self):
        keyword = [Doc(f"keyword {index}", "SympScan") for index in range(4)]
        semantic = [Doc(f"semantic {index}", "SympScan") for index in range(4)]
        self.retriever.inverted_index = types.SimpleNamespace(search=lambda *_args: keyword)
        self.retriever.vector_database = types.SimpleNamespace(search=lambda *_args: semantic)

        expected = [keyword[0], semantic[0], keyword[1], semantic[1], keyword[2]]
        results = self.retriever.hybrid_retrieval(
            ["query"], "query", True, True, False, False
        )
        self.assertEqual(results, expected)

        self.retriever.rerank_model = types.SimpleNamespace(
            predict=lambda *_args: (_ for _ in ()).throw(RuntimeError("ranker unavailable"))
        )
        with patch("logging.getLogger"):
            results, warnings = self.retriever.hybrid_retrieval(
                ["query"], "query", True, True, False, True, return_warnings=True
            )
        self.assertEqual(results, expected)
        self.assertIn("Reranking was unavailable", warnings[0])

    def test_empty_passages_do_not_displace_usable_results(self):
        blank = Doc(" \n", "SympScan")
        useful = [Doc(f"passage {index}", "SympScan") for index in range(6)]
        self.retriever.inverted_index = types.SimpleNamespace(search=lambda *_args: [blank, *useful])
        self.retriever.vector_database = None

        results = self.retriever.hybrid_retrieval(
            ["query"], "query", True, False, False, False
        )
        self.assertEqual(results, useful[:5])

    def test_startup_keeps_keyword_path_when_semantic_and_reranker_fail(self):
        passage = Doc("keyword passage", "SympScan")
        keyword_index = types.SimpleNamespace(search=lambda *_args: [passage])
        with (
            patch.object(self.Retriever, "load_semantic_model", side_effect=FileNotFoundError("model missing")),
            patch.object(self.module, "Vector_DB", side_effect=AssertionError("vector should not load")),
            patch.object(self.module, "CrossEncoder", side_effect=RuntimeError("reranker unavailable")),
            patch.object(self.module, "Knowledge_Graphbase", return_value=object()),
            patch.object(self.module.joblib, "load", return_value=keyword_index, create=True),
            patch("logging.getLogger"),
        ):
            retriever = self.Retriever(restore_graph_snapshot=False)

        results, warnings = retriever.hybrid_retrieval(
            ["query"], "query", True, True, True, True, return_warnings=True
        )
        self.assertEqual(results, [passage])
        self.assertIsNone(retriever.vector_database)
        self.assertIsNone(retriever.rerank_model)
        self.assertEqual(len(warnings), 2)

    def test_startup_without_passage_indexes_returns_no_results(self):
        with (
            patch.object(self.Retriever, "load_semantic_model", side_effect=FileNotFoundError("model missing")),
            patch.object(self.module, "CrossEncoder", return_value=object()),
            patch.object(self.module, "Knowledge_Graphbase", side_effect=RuntimeError("graph unavailable")),
            patch.object(self.module.joblib, "load", side_effect=FileNotFoundError("index missing"), create=True),
            patch("logging.getLogger"),
        ):
            retriever = self.Retriever(restore_graph_snapshot=False)

        results, warnings = retriever.hybrid_retrieval(
            ["query"], "query", True, True, True, True, return_warnings=True
        )
        self.assertEqual(results, [])
        self.assertIsNone(retriever.knowledge_database)
        self.assertEqual(len(warnings), 3)

    def test_explicit_graph_restore_failure_still_stops_startup(self):
        with (
            patch.object(self.Retriever, "load_semantic_model", return_value=object()),
            patch.object(self.module, "Vector_DB", return_value=object()),
            patch.object(self.module, "CrossEncoder", return_value=object()),
            patch.object(self.module, "Knowledge_Graphbase", side_effect=RuntimeError("restore unavailable")),
            patch.object(self.module.joblib, "load", return_value=object(), create=True),
        ):
            with self.assertRaises(RuntimeError):
                self.Retriever(restore_graph_snapshot=True)

    def test_graph_context_survives_missing_reranker(self):
        self.retriever.rerank_model = None
        relationship = {
            "entity1": "disease a", "entity1_type": "Disease",
            "connection": "treated_with", "entity2": "drug b",
            "entity2_type": "Medication",
        }
        self.retriever.knowledge_database = types.SimpleNamespace(
            search=lambda _entities: [relationship]
        )

        graph_text, warning = self.retriever.graph_retrieve(
            [(["disease a"], [])], "question", True, True, return_warning=True
        )
        self.assertIsInstance(graph_text, str)
        self.assertIn("disease a", graph_text)
        self.assertIn("drug b", graph_text)
        self.assertIn("Graph reranking was unavailable", warning)

    def test_graph_reranking_failure_preserves_unique_relationships(self):
        relationships = [
            {
                "entity1": f"disease {index}", "entity1_type": "Disease",
                "connection": ("alert", "treated_with", "has_context_of")[index % 3],
                "entity2": f"related {index}",
                "entity2_type": "Medication",
            }
            for index in range(7)
        ]
        self.retriever.knowledge_database = types.SimpleNamespace(
            search=lambda _entities: relationships
        )
        self.retriever.rerank_model = types.SimpleNamespace(
            predict=lambda *_args: (_ for _ in ()).throw(RuntimeError("ranker unavailable"))
        )
        with patch("logging.getLogger"):
            graph_text, warning = self.retriever.graph_retrieve(
                [(["disease"], []), (["disease"], [])], "question", True, True,
                return_warning=True,
            )
        lines = graph_text.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(set(lines)), 5)
        self.assertIn("disease 0", lines[0])
        self.assertIn("Graph reranking was unavailable", warning)

    def test_invalid_graph_reranking_scores_keep_relationships(self):
        relationship = {
            "entity1": "disease a", "entity1_type": "Disease",
            "connection": "treated_with", "entity2": "drug b",
            "entity2_type": "Medication",
        }
        self.retriever.knowledge_database = types.SimpleNamespace(search=lambda _entities: [relationship])

        for scores in ([], [float("nan")], [float("inf")]):
            with self.subTest(scores=scores), patch("logging.getLogger"):
                self.retriever.rerank_model = types.SimpleNamespace(predict=lambda *_args: scores)
                graph_text, warning = self.retriever.graph_retrieve(
                    [(["disease a"], [])], "question", True, True, return_warning=True
                )
                self.assertIn("disease a", graph_text)
                self.assertIn("Graph reranking was unavailable", warning)

    def test_graph_ranking_separates_relationships(self):
        graph_text = self.retriever.merge_multi_subgraph_cross_encoder(
            ["first relationship.", "second relationship."], "question"
        )
        self.assertEqual(graph_text, "first relationship.\nsecond relationship.")


if __name__ == "__main__":
    unittest.main()
