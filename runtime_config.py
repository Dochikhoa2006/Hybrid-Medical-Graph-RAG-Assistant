"""Shared runtime defaults for local model and service endpoints."""

import os


DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b-instruct-q5_k_m"
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_NEO4J_URI = "bolt://my-neo4j:7687"


def ollama_settings(model=None, base_url=None):
    return (
        model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
    )


def neo4j_uri(connection_uri=None):
    return connection_uri or os.getenv("NEO4J_URI") or DEFAULT_NEO4J_URI
