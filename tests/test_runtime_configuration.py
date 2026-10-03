"""Check runtime endpoints without requiring Neo4j or Ollama services."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeConfigurationTests(unittest.TestCase):
    def test_query_processor_uses_configured_ollama_endpoint(self):
        calls = []
        ollama = types.ModuleType("langchain_ollama")
        ollama.OllamaLLM = lambda **kwargs: calls.append(kwargs)
        with patch.dict(sys.modules, {"langchain_ollama": ollama}):
            processor = load_module("query_config_test", "PreRetrival_and_PostRetrieval.py")
            with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://ollama.local:11434"}):
                processor.Context_Processer()
            processor.Context_Processer(base_url="http://explicit.local:11434")
            with patch.dict(os.environ, {"OLLAMA_BASE_URL": ""}):
                processor.Context_Processer()
        self.assertEqual(calls[0]["base_url"], "http://ollama.local:11434")
        self.assertEqual(calls[1]["base_url"], "http://explicit.local:11434")
        self.assertEqual(calls[2]["base_url"], "http://host.docker.internal:11434")

    def test_graph_uses_configured_neo4j_endpoint(self):
        calls = []
        neo4j = types.ModuleType("neo4j")
        neo4j.GraphDatabase = types.SimpleNamespace(
            driver=lambda uri, auth: calls.append((uri, auth))
        )
        pyspark = types.ModuleType("pyspark")
        pyspark.sql = types.ModuleType("pyspark.sql")
        pyspark.sql.SparkSession = object
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        modules = {
            "neo4j": neo4j,
            "pyspark": pyspark,
            "pyspark.sql": pyspark.sql,
            "dotenv": dotenv,
            "joblib": types.ModuleType("joblib"),
        }
        with patch.dict(sys.modules, modules):
            graph = load_module("graph_config_test", "Knowledge_Graph.py")
            with patch.dict(os.environ, {"NEO4J_URI": "bolt://graph.local:7687", "AUTH": "test-password"}):
                graph.Knowledge_Graphbase()
            graph.Knowledge_Graphbase(connection_URI="bolt://explicit.local:7687")
            with patch.dict(os.environ, {"NEO4J_URI": ""}):
                graph.Knowledge_Graphbase()
        self.assertEqual(calls[0], ("bolt://graph.local:7687", ("neo4j", "test-password")))
        self.assertEqual(calls[1][0], "bolt://explicit.local:7687")
        self.assertEqual(calls[2][0], "bolt://my-neo4j:7687")


if __name__ == "__main__":
    unittest.main()
