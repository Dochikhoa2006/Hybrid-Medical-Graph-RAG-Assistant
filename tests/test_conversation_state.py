"""Regression checks for conversation state without loading model artifacts."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


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

    def test_chat_log_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.log"
            with patch.dict(os.environ, {"ENABLE_CHAT_LOGGING": "false"}):
                rag = self.RAG(retriever=object(), log_file_path=str(path))
            rag.user_query = "Sensitive question"
            rag.Logging()
            self.assertFalse(path.exists())
            self.assertEqual(rag.llm.prompts, [])

    def test_chat_log_can_be_enabled_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.log"
            with patch.dict(os.environ, {"ENABLE_CHAT_LOGGING": "true"}):
                rag = self.RAG(retriever=object(), log_file_path=str(path))
            rag.user_query = "Example question"
            rag.Logging()
            record = json.loads(path.read_text().strip())
            self.assertEqual(record["RAW USER QUERY"], "Example question")

    def test_response_formatting_flattens_nested_values(self):
        rag = self.RAG.__new__(self.RAG)
        self.assertEqual(
            rag.process_valid_response('["Aspirin", {"other": ["Rest", "Water"]}]'),
            "Aspirin, Rest, Water.",
        )
        self.assertEqual(rag.process_valid_response("Already complete?"), "Already complete?")
        self.assertEqual(rag.process_valid_response("[]"), "")

    def test_json_parser_handles_extra_braces_and_validates_fields(self):
        rag = self.RAG.__new__(self.RAG)
        rag.intent = "CHITCHAT"
        self.assertEqual(
            rag.response_format_check('```json\n{"chitchat": "Hello"}\n``` Extra {note}'),
            {"chitchat": "Hello"},
        )
        self.assertEqual(
            rag.response_format_check('{"chitchat": 2} {"chitchat": "Hello"}'),
            {"chitchat": "Hello"},
        )
        rag.intent = "RAG_SEARCH"
        self.assertEqual(rag.response_format_check('{"answer": "Only one field"}'), "")
        self.assertEqual(rag.response_format_check(None), "")

    def test_new_request_clears_previous_retrieval_context(self):
        rag = self.RAG(retriever=object())
        rag.chat_history = "Previous summary"
        rag.hybrid_text = "Previous medical chunks"
        rag.graph_text = "Previous graph facts"
        rag.first_response = "Previous raw answer"
        rag.context_processer.user_query_understanding = lambda *_args: (None, "hello", "CHITCHAT")
        rag.Generation = lambda _prompt: ("new raw answer", "new answer", "success")

        self.assertEqual(rag.RAG_Online_Phase("hello"), "new answer")
        self.assertEqual(rag.hybrid_text, "")
        self.assertEqual(rag.graph_text, "")
        self.assertEqual(rag.first_response, "new raw answer")
        self.assertEqual(rag.chat_history, "Previous summary")

    def test_request_state_is_cleared_even_when_retrieval_fails(self):
        rag = self.RAG(retriever=object())
        rag.hybrid_text = "Old chunks"
        rag.status = "success"
        rag.Retrieval = Mock(side_effect=RuntimeError("retrieval failed"))

        with self.assertRaises(RuntimeError):
            rag.RAG_Online_Phase("new question")
        self.assertEqual(rag.user_query, "new question")
        self.assertEqual(rag.hybrid_text, "")
        self.assertEqual(rag.status, "")

    def test_summary_runs_before_optional_logging_failure(self):
        rag = self.RAG(retriever=object())
        events = []
        rag.Summarize_Chat_History = lambda: events.append("summary")
        rag.Caching = lambda: events.append("cache")

        def failed_logging():
            events.append("logging")
            raise OSError("disk full")

        rag.Logging = failed_logging
        with self.assertRaises(OSError):
            rag.RAG_PostOnline_Phase()
        self.assertEqual(events, ["summary", "cache", "logging"])

    def test_medical_query_abstains_without_retrieved_evidence(self):
        rag = self.RAG(retriever=object())

        def empty_retrieval():
            rag.intent = "RAG_SEARCH"
            rag.rewritten_query = "medical question"
            rag.hybrid_text = "  "
            rag.graph_text = ["", "  "]

        rag.Retrieval = empty_retrieval
        rag.Augmentation = Mock(side_effect=AssertionError("augmentation should be skipped"))
        rag.Generation = Mock(side_effect=AssertionError("generation should be skipped"))

        answer = rag.RAG_Online_Phase("medical question")
        self.assertIn("couldn't find supporting information", answer)
        self.assertEqual(rag.status, "no_evidence")
        self.assertEqual(rag.first_response, "")
        rag.RAG_PostOnline_Phase()
        self.assertEqual(rag.chat_history, "No prior conversation")
        self.assertEqual(rag.llm.prompts, [])

    def test_medical_query_with_evidence_reaches_generation(self):
        rag = self.RAG(retriever=object())

        def evidence_retrieval():
            rag.intent = "RAG_SEARCH"
            rag.rewritten_query = "medical question"
            rag.hybrid_text = "Retrieved chunk"

        rag.Retrieval = evidence_retrieval
        rag.Augmentation = Mock(return_value="grounded prompt")
        rag.Generation = Mock(return_value=("raw", "answer", "success"))

        self.assertEqual(rag.RAG_Online_Phase("medical question"), "answer")
        rag.Generation.assert_called_once_with("grounded prompt")

    def test_abstention_log_has_no_model_generated_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.log"
            rag = self.RAG(retriever=object(), log_file_path=str(path), log_interactions=True)
            rag.status = "no_evidence"
            rag.final_response = "No supporting information found"
            rag.Logging()
            record = json.loads(path.read_text().strip())
            self.assertIsNone(record["RESPONSE CONFIDENCE (0-1)"])
            self.assertIsNone(record["RETRIEVAL CONFIDENCE (0-1)"])
            self.assertEqual(rag.llm.prompts, [])


if __name__ == "__main__":
    unittest.main()
