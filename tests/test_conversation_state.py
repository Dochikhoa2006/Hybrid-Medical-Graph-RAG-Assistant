"""Regression checks for conversation state without loading model artifacts."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeLLM:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return "Updated summary"


class FakeContextProcesser:
    def __init__(self, **kwargs):
        self.options = kwargs


class ConversationStateTests(unittest.TestCase):
    def setUp(self):
        self.logging_patch = patch("logging.basicConfig")
        self.logging_patch.start()

    def tearDown(self):
        self.logging_patch.stop()

    @classmethod
    def setUpClass(cls):
        modules = {
            "Hybrid_Dual_Indexing": types.ModuleType("Hybrid_Dual_Indexing"),
            "PreRetrival_and_PostRetrieval": types.ModuleType("PreRetrival_and_PostRetrieval"),
            "Retrieval": types.ModuleType("Retrieval"),
            "langchain_ollama": types.ModuleType("langchain_ollama"),
        }
        modules["Hybrid_Dual_Indexing"].Keyword_Search = object
        modules["Hybrid_Dual_Indexing"].Semantic_Search = object
        modules["PreRetrival_and_PostRetrieval"].Context_Processer = FakeContextProcesser
        modules["Retrieval"].Retriever = object
        modules["langchain_ollama"].OllamaLLM = FakeLLM
        cls.module_patch = patch.dict(sys.modules, modules)
        cls.module_patch.start()
        spec = importlib.util.spec_from_file_location(
            "rag_conversation_test", Path(__file__).resolve().parents[1] / "Augmented_Generation.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.RAG = module.RAG

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()

    def test_shared_retriever_does_not_share_conversation(self):
        retriever = object()
        first = self.RAG(retriever=retriever)
        second = self.RAG(retriever=retriever)

        first.user_query = "First session question"
        first.rewritten_query = "First rewritten question"
        first.final_response = "First session answer"
        first.status = "success"
        second.user_query = "Second session question"
        second.rewritten_query = "Second rewritten question"
        second.final_response = "Second session answer"
        second.status = "success"

        first.Summarize_Chat_History()
        second.Summarize_Chat_History()

        self.assertIs(first.retriever, second.retriever)
        self.assertIn("First session answer", first.llm.prompts[0])
        self.assertNotIn("Second session answer", first.llm.prompts[0])
        self.assertIn("Second session answer", second.llm.prompts[0])
        self.assertNotIn("First session answer", second.llm.prompts[0])

    def test_failed_response_does_not_enter_history(self):
        rag = self.RAG(retriever=object())
        rag.status = "fail"
        rag.Summarize_Chat_History()
        self.assertEqual(rag.chat_history, "No prior conversation")
        self.assertEqual(rag.llm.prompts, [])

    def test_ollama_endpoint_reaches_generation_and_query_processing(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://example.local:11434"}):
            rag = self.RAG(retriever=object())
        self.assertEqual(rag.llm.options["base_url"], "http://example.local:11434")
        self.assertEqual(rag.context_processer.options["base_url"], "http://example.local:11434")

        explicit = self.RAG(retriever=object(), base_url="http://override.local:11434")
        self.assertEqual(explicit.llm.options["base_url"], "http://override.local:11434")
        self.assertEqual(explicit.context_processer.options["base_url"], "http://override.local:11434")


if __name__ == "__main__":
    unittest.main()
