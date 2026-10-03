"""Regression checks for retrieval fusion and reranking inputs."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
