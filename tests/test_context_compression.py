"""Checks for passage-level context compression without loading Ollama."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class ContextCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {
            "langchain_ollama": types.ModuleType("langchain_ollama"),
            "runtime_config": types.ModuleType("runtime_config"),
        }
        modules["langchain_ollama"].OllamaLLM = object
        modules["runtime_config"].ollama_settings = lambda model, base_url: (model, base_url)
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "context_compression_test",
                Path(__file__).resolve().parents[1] / "PreRetrival_and_PostRetrieval.py",
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        cls.ContextProcesser = module.Context_Processer

    def test_compresses_each_passage_and_discards_irrelevant_results(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.side_effect = [
            "SUMMARY: Relevant finding.",
            "SUMMARY: NOT_RELEVANT",
            "Another finding",
        ]
        chunks = [types.SimpleNamespace(page_content=text) for text in ("first", "second", "third")]

        context = processor.context_retrieval_processing(chunks, "query", False, True)

        self.assertEqual(context, "- Relevant finding.\n- Another finding.")
        self.assertEqual(len(processor.llm.invoke.call_args_list), 3)
        for chunk, call in zip(chunks, processor.llm.invoke.call_args_list):
            self.assertIn(f"DOCUMENT CHUNK: {chunk.page_content}", call.args[0])

    def test_compression_reports_only_passages_used_in_prompt(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.side_effect = ["Relevant finding", "NOT_RELEVANT", "Other finding"]
        chunks = [types.SimpleNamespace(page_content=text) for text in ("first", "second", "third")]

        context, used_chunks = processor.context_retrieval_processing(
            chunks, "query", False, True, return_used_chunks=True
        )

        self.assertEqual(context, "- Relevant finding.\n- Other finding.")
        self.assertEqual(used_chunks, [chunks[0], chunks[2]])

    def test_all_irrelevant_passages_leave_no_compressed_evidence(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.side_effect = ["NOT_RELEVANT", "SUMMARY: NOT_RELEVANT."]
        chunks = [types.SimpleNamespace(page_content=text) for text in ("first", "second")]

        self.assertEqual(processor.context_retrieval_processing(chunks, "query", False, True), "")

    def test_intent_detection_requires_explicit_category(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        for response, expected in (
            ("CHITCHAT", "CHITCHAT"),
            ("CATEGORY: CHITCHAT", "CHITCHAT"),
            ("CATEGORY: RAG_SEARCH", "RAG_SEARCH"),
            ("Medical query; the prompt also mentions CHITCHAT", "RAG_SEARCH"),
            ("CATEGORY: (TYPE)\nRAG_SEARCH", "RAG_SEARCH"),
        ):
            with self.subTest(response=response):
                processor.llm.invoke.return_value = response
                self.assertEqual(processor.intent_detection("medical question", ""), expected)

    def test_empty_rewrite_preserves_user_query(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        for response in ("", "REWRITE:", "REWRITE:   "):
            with self.subTest(response=response):
                processor.llm.invoke.return_value = response
                self.assertEqual(processor.rewrite("original medical question", ""), "original medical question")

    def test_expansion_keeps_original_and_ignores_blank_variation(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.return_value = "EXPANSIONS:\n1. Different wording"
        self.assertEqual(processor.expansion("original", ""), ["Different wording", "original"])
        processor.llm.invoke.return_value = "EXPANSIONS:\n1.   "
        self.assertEqual(processor.expansion("original", ""), ["original"])
        processor.llm.invoke.return_value = "EXPANSIONS:\n1. original"
        self.assertEqual(processor.expansion("original", ""), ["original"])

    def test_blank_hyde_uses_query_and_preserves_other_candidates(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.return_value = "  "
        self.assertEqual(processor.HyDE(["variation", "original"], "", 2), ["variation", "original"])

    def test_query_pipeline_never_sends_blank_candidate_to_retrieval(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.intent_detection = Mock(return_value="RAG_SEARCH")
        processor.rewrite = Mock(return_value="")
        processor.expansion = Mock(return_value=["", "  "])
        processor.HyDE = Mock(return_value=[""])

        candidates, display_query, intent = processor.user_query_understanding(
            "original question", "", True, True, True, True
        )
        self.assertEqual((candidates, display_query, intent), (["original question"], "original question", "RAG_SEARCH"))

    def test_hyde_retrieval_keeps_rewritten_query(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.intent_detection = Mock(return_value="RAG_SEARCH")
        processor.rewrite = Mock(return_value="rewritten question")
        processor.expansion = Mock(return_value=["alternate wording", "rewritten question"])
        processor.HyDE = Mock(return_value=["hypothetical answer", "rewritten question"])

        candidates, display_query, intent = processor.user_query_understanding(
            "original question", "", True, True, True, True
        )
        self.assertEqual(candidates, ["hypothetical answer", "rewritten question"])
        self.assertEqual(display_query, "hypothetical answer")
        self.assertEqual(intent, "RAG_SEARCH")

    def test_entity_extraction_ignores_echoed_prompt_example(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.side_effect = [
            'Output: DISEASE: ["Migraine"] & MEDICATION: ["Ibuprofen"]\n'
            'Query: "actual question"\nOutput: (YOUR RESPONSE)',
            'Output: DISEASE: ["Migraine"] & MEDICATION: ["Ibuprofen"]\n'
            'Query: "actual question"\nOutput: DISEASE: ["Asthma"] & MEDICATION: []',
            'DISEASE: ["Asthma"] & MEDICATION: ["Albuterol"]',
        ]

        self.assertEqual(processor.entity_extraction(["query"], True), [[[], []]])
        self.assertEqual(processor.entity_extraction(["query"], True), [[['asthma'], []]])
        self.assertEqual(
            processor.entity_extraction(["query"], True),
            [[['asthma'], ['albuterol']]],
        )


if __name__ == "__main__":
    unittest.main()
