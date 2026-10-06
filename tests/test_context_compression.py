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

    def test_all_irrelevant_passages_leave_no_compressed_evidence(self):
        processor = self.ContextProcesser.__new__(self.ContextProcesser)
        processor.llm = Mock()
        processor.llm.invoke.side_effect = ["NOT_RELEVANT", "SUMMARY: NOT_RELEVANT."]
        chunks = [types.SimpleNamespace(page_content=text) for text in ("first", "second")]

        self.assertEqual(processor.context_retrieval_processing(chunks, "query", False, True), "")


if __name__ == "__main__":
    unittest.main()
