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


if __name__ == "__main__":
    unittest.main()
