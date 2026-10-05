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
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://example.local:11434",
            "OLLAMA_MODEL": "custom-answer-model",
        }):
            rag = self.RAG(retriever=object())
        self.assertEqual(rag.model_name, "custom-answer-model")
        self.assertEqual(rag.llm.options["model"], "custom-answer-model")
        self.assertEqual(rag.context_processer.options["model"], "custom-answer-model")
        self.assertEqual(rag.llm.options["base_url"], "http://example.local:11434")
        self.assertEqual(rag.context_processer.options["base_url"], "http://example.local:11434")

        explicit = self.RAG(retriever=object(), model="explicit-model", base_url="http://override.local:11434")
        self.assertEqual(explicit.model_name, "explicit-model")
        self.assertEqual(explicit.context_processer.options["model"], "explicit-model")
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
        self.assertEqual(
            rag.response_format_check('{"outer": {"chitchat": "Nested answer"}, invalid}'),
            "",
        )
        self.assertEqual(
            rag.response_format_check(
                '{"outer": {"chitchat": "Nested answer"}, invalid} '
                '{"chitchat": "Real answer"}'
            ),
            {"chitchat": "Real answer"},
        )
        self.assertEqual(
            rag.response_format_check('{"chitchat": "Hello \\"{name}\\""}'),
            {"chitchat": 'Hello "{name}"'},
        )
        self.assertEqual(rag.response_format_check(
            '{"chitchat": "brief and redirect 2-5 sentences."}'
        ), "")
        self.assertEqual(rag.response_format_check(
            '{"chitchat": "First", "chitchat": "Second"}'
        ), "")
        rag.intent = "RAG_SEARCH"
        self.assertEqual(rag.response_format_check('{"answer": "Only one field"}'), "")
        self.assertEqual(rag.response_format_check(
            '{"answer": " ", "disease": "Disease A", "medication": "Drug B", "advice": "Rest"}'
        ), "")
        self.assertEqual(rag.response_format_check(
            '{"answer": "concise 2-5 sentences", "disease": "Disease A", "medication": "", "advice": ""}'
        ), "")
        self.assertEqual(rag.response_format_check(
            '{"answer": "A real answer", "disease": "Disease mentioned in the text", "medication": "", "advice": ""}'
        ), "")
        self.assertEqual(rag.response_format_check(
            '{"answer": "First", "answer": "Second", "disease": "", "medication": "", "advice": ""}'
        ), "")
        self.assertEqual(rag.response_format_check(None), "")

    def test_duplicate_json_key_triggers_repair(self):
        rag = self.RAG(retriever=object())
        rag.intent = "CHITCHAT"
        rag.llm.invoke = Mock(side_effect=[
            '{"chitchat": "First", "chitchat": "Second"}',
            '{"chitchat": "Hello"}',
        ])

        _, answer, status = rag.Generation("BASE PROMPT", format_fail=2)
        self.assertEqual(status, "success")
        self.assertEqual(rag.llm.invoke.call_count, 2)
        self.assertIn("Hello", answer)
        self.assertIn("Do not repeat keys", rag.llm.invoke.call_args_list[1].args[0])

    def test_schema_placeholder_echo_triggers_repair(self):
        rag = self.RAG(retriever=object())
        rag.intent = "RAG_SEARCH"
        rag.llm.invoke = Mock(side_effect=[
            '{"answer": "concise 2-5 sentences", "disease": "Disease mentioned in the text", "medication": "", "advice": ""}',
            '{"answer": "The passage describes Disease A.", "disease": "Disease A", "medication": "", "advice": ""}',
        ])

        _, answer, status = rag.Generation("BASE PROMPT", format_fail=2)
        self.assertEqual(status, "success")
        self.assertEqual(rag.llm.invoke.call_count, 2)
        self.assertIn("The passage describes Disease A", answer)
        self.assertIn("copy placeholder text", rag.llm.invoke.call_args_list[1].args[0])

    def test_blank_primary_answer_is_repaired_before_rendering_medical_fields(self):
        rag = self.RAG(retriever=object())
        rag.intent = "RAG_SEARCH"
        rag.llm.invoke = Mock(side_effect=[
            '{"answer": "", "disease": "Disease A", "medication": "Drug B", "advice": "Rest"}',
            '{"answer": "The passage describes Disease A.", "disease": "Disease A", "medication": "", "advice": ""}',
        ])

        _, answer, status = rag.Generation("BASE PROMPT", format_fail=2)
        self.assertEqual(status, "success")
        self.assertEqual(rag.llm.invoke.call_count, 2)
        self.assertIn("The passage describes Disease A", answer)
        self.assertIn("answer value must not be blank", rag.llm.invoke.call_args_list[1].args[0])

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
            self.assertIsNone(record["RESPONSE HEURISTIC SCORE (0-1)"])
            self.assertIsNone(record["RETRIEVAL HEURISTIC SCORE (0-1)"])
            self.assertEqual(rag.llm.prompts, [])

    def test_telemetry_scores_require_one_finite_unit_value(self):
        rag = self.RAG(retriever=object())
        self.assertEqual(rag.parse_unit_score("0.75"), 0.75)
        self.assertEqual(rag.parse_unit_score(" 1.0 "), 1.0)
        for invalid in ("5/10", "Score: 0.7", "1.2", "-0.1", "nan", "inf", ""):
            with self.subTest(value=invalid):
                self.assertIsNone(rag.parse_unit_score(invalid))

        rag.llm.invoke = Mock(return_value="5/10")
        self.assertEqual(rag.response_content_check("answer", "question"), (None, False))
        rag.hybrid_text = "Retrieved passage"
        self.assertIsNone(rag.how_retrieval_helpful())

        rag.llm.invoke = Mock(return_value="0.6")
        self.assertEqual(rag.response_content_check("answer", "question", relevance_threshold=0.7), (0.6, False))

    def test_retrieved_context_keeps_distinct_sources_and_graph(self):
        rag = self.RAG(retriever=object())
        first = types.SimpleNamespace(
            page_content="Same passage", metadata={"source": "SympScan", "disease_name": "Disease A"}
        )
        duplicate = types.SimpleNamespace(
            page_content="Same passage", metadata={"source": "SympScan", "disease_name": "Disease A"}
        )
        second = types.SimpleNamespace(
            page_content="Same passage", metadata={"source": "Other source", "disease_name": "Disease B"}
        )
        rag.context_processer.user_query_understanding = lambda *_args: (["question"], "question", "RAG_SEARCH")
        rag.context_processer.context_retrieval_processing = lambda *_args: "Same passage"
        rag.context_processer.entity_extraction = lambda *_args: []
        rag.retriever = types.SimpleNamespace(
            hybrid_retrieval=lambda *_args, **_kwargs: ([first, duplicate, second], []),
            graph_retrieve=lambda *_args: ["Disease A treated_with X"],
        )

        rag.Retrieval()
        snapshot = rag.response_context_snapshot()
        self.assertEqual(len(snapshot), 3)
        self.assertEqual([item["source"] for item in snapshot], [
            "SympScan", "Other source", "SympScan graph snapshot"
        ])
        self.assertEqual(snapshot[-1]["kind"], "Graph relationships")

        rag.reset_request_state()
        self.assertEqual(rag.retrieved_context, [])
        self.assertEqual(len(snapshot), 3)
        snapshot[0]["text"] = "changed in chat history"
        self.assertEqual(first.page_content, "Same passage")

    def test_displayed_passages_match_default_prompt_limit(self):
        rag = self.RAG(retriever=object())
        chunks = [
            types.SimpleNamespace(page_content=f"Passage {index}", metadata={"source": "SympScan"})
            for index in range(6)
        ]
        rag.context_processer.user_query_understanding = lambda *_args: (["question"], "question", "RAG_SEARCH")
        rag.context_processer.context_retrieval_processing = lambda *_args: "Prompt uses first five passages"
        rag.context_processer.entity_extraction = lambda *_args: []
        rag.retriever = types.SimpleNamespace(
            hybrid_retrieval=lambda *_args, **_kwargs: (chunks, []),
            graph_retrieve=lambda *_args: "",
        )

        rag.Retrieval()
        self.assertEqual([item["text"] for item in rag.retrieved_context], [
            "Passage 0", "Passage 1", "Passage 2", "Passage 3", "Passage 4"
        ])

    def test_graph_failure_uses_passages_and_records_warning(self):
        rag = self.RAG(retriever=object())
        passage = types.SimpleNamespace(
            page_content="Retrieved passage", metadata={"source": "SympScan"}
        )
        rag.context_processer.user_query_understanding = lambda *_args: (["question"], "question", "RAG_SEARCH")
        rag.context_processer.context_retrieval_processing = lambda *_args: "Retrieved passage"
        rag.context_processer.entity_extraction = lambda *_args: []

        def unavailable_graph(*_args):
            raise ConnectionError("Neo4j unavailable")

        rag.retriever = types.SimpleNamespace(
            hybrid_retrieval=lambda *_args, **_kwargs: ([passage], ["Some keyword searches failed; available results were used."]),
            graph_retrieve=unavailable_graph,
        )
        rag.Generation = Mock(return_value=("raw", "answer", "success"))

        with patch("logging.getLogger"):
            self.assertEqual(rag.RAG_Online_Phase("question"), "answer")
        self.assertEqual(rag.graph_text, "")
        self.assertIn("Graph lookup was unavailable", rag.graph_warning)
        self.assertEqual(len(rag.search_warnings), 1)
        self.assertEqual(len(rag.retrieved_context), 1)
        rag.Generation.assert_called_once()

        rag.reset_request_state()
        self.assertEqual(rag.graph_warning, "")
        self.assertEqual(rag.search_warnings, [])

    def test_graph_failure_without_passages_still_abstains(self):
        rag = self.RAG(retriever=object())
        rag.context_processer.user_query_understanding = lambda *_args: (["question"], "question", "RAG_SEARCH")
        rag.context_processer.context_retrieval_processing = lambda *_args: ""
        rag.context_processer.entity_extraction = lambda *_args: []
        rag.retriever = types.SimpleNamespace(
            hybrid_retrieval=lambda *_args, **_kwargs: ([], []),
            graph_retrieve=Mock(side_effect=ConnectionError("Neo4j unavailable")),
        )
        rag.Generation = Mock(side_effect=AssertionError("generation should be skipped"))

        with patch("logging.getLogger"):
            answer = rag.RAG_Online_Phase("question")
        self.assertEqual(rag.status, "no_evidence")
        self.assertIn("couldn't find supporting information", answer)
        self.assertIn("Graph lookup was unavailable", rag.graph_warning)
        rag.Generation.assert_not_called()

    def test_generation_repairs_from_latest_bounded_output(self):
        rag = self.RAG(retriever=object())
        rag.intent = "CHITCHAT"
        first_invalid = "A" * 5000
        second_invalid = "B" * 5000
        rag.llm.invoke = Mock(side_effect=[
            first_invalid, second_invalid, '{"chitchat": "Hello"}'
        ])

        first, answer, status = rag.Generation("BASE PROMPT", format_fail=3)
        self.assertEqual(first, first_invalid)
        self.assertEqual(status, "success")
        self.assertIn("Hello", answer)
        prompts = [call.args[0] for call in rag.llm.invoke.call_args_list]
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0], "BASE PROMPT")
        self.assertLess(len(prompts[1]), 1600)
        self.assertLess(len(prompts[2]), 1600)
        self.assertIn("B" * 100, prompts[2])
        self.assertNotIn("A" * 100, prompts[2])

    def test_generation_stops_after_format_attempt_limit(self):
        rag = self.RAG(retriever=object())
        rag.intent = "RAG_SEARCH"
        rag.llm.invoke = Mock(return_value="not JSON")

        first, answer, status = rag.Generation("BASE PROMPT", format_fail=2)
        self.assertEqual(first, "not JSON")
        self.assertEqual(status, "fail")
        self.assertIn("couldn't format a response", answer)
        self.assertEqual(rag.llm.invoke.call_count, 2)

        with self.assertRaises(ValueError):
            rag.Generation("BASE PROMPT", format_fail=0)

    def test_clear_conversation_removes_session_state_but_keeps_optional_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.log"
            path.write_text("existing log entry\n")
            retriever = object()
            rag = self.RAG(retriever=retriever, log_file_path=str(path), log_interactions=True)
            rag.chat_history = "Prior discussion"
            rag.user_query = "Sensitive question"
            rag.final_response = "Prior answer"
            rag.hybrid_text = "Prior retrieved passage"
            rag.graph_text = "Prior graph fact"
            rag.retrieved_context = [{"text": "Prior retrieved passage"}]
            rag.graph_warning = "Prior graph warning"
            rag.search_warnings = ["Prior search warning"]

            rag.clear_conversation()

            self.assertEqual(rag.chat_history, "No prior conversation")
            self.assertEqual(rag.user_query, "")
            self.assertEqual(rag.final_response, "")
            self.assertEqual(rag.hybrid_text, "")
            self.assertEqual(rag.graph_text, "")
            self.assertEqual(rag.retrieved_context, [])
            self.assertEqual(rag.graph_warning, "")
            self.assertEqual(rag.search_warnings, [])
            self.assertIs(rag.retriever, retriever)
            self.assertEqual(path.read_text(), "existing log entry\n")


if __name__ == "__main__":
    unittest.main()
