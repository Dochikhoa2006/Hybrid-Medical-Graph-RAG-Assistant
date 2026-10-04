# Safety and limitations

SympScan is an educational retrieval and generation prototype. This document makes its intended boundary explicit for users, reviewers, and future contributors.

## Intended use

Appropriate uses include:

- Studying hybrid information-retrieval architecture.
- Demonstrating BM25, FAISS, Neo4j, reranking, and local LLM integration.
- Exploring dataset-derived relationships in a controlled local environment.
- Building portfolio evidence for data engineering and applied AI work.

## Prohibited or unsupported use

Do not use SympScan for:

- Diagnosing a person or ruling out a condition.
- Emergency triage or deciding whether care is urgent.
- Selecting, starting, stopping, or dosing medication.
- Treatment planning, contraindication checking, or clinical decision support.
- Processing real patient records or protected health information.
- Public, multi-user, or production healthcare deployment in its current form.

If someone may be experiencing a medical emergency, contact local emergency services or an appropriate healthcare professional. Do not wait for this application.

## Evidence boundary

All retrieval context is derived from one community dataset snapshot. The Neo4j graph restructures information from that same source; it does not independently verify it against clinical guidelines, peer-reviewed literature, drug references, or clinician review.

Generated answers may be incomplete, outdated, incorrectly matched, or fabricated despite retrieval and JSON formatting. Structured output improves presentation consistency, not medical validity.

If both retrieval paths return no usable text for a medical query, the application abstains without asking the generation model to answer. This checks only whether context is present; it does not establish relevance, accuracy, or clinical safety.

## Known implementation limitations

- There is no emergency or red-flag classifier.
- There is no clinical evaluation set, clinician adjudication, or regulated validation.
- The UI shows retrieved context with each answer, but does not verify or cite individual claims against it.
- Graph lookup depends on LLM/regex entity extraction and exact normalized names.
- Graph retrieval is one hop and does not perform clinical reasoning.
- The embedding and reranking models are general-purpose rather than clinically validated for this dataset.
- The generation model is a small local Qwen 2.5 variant.
- JSON checking verifies parseability but not a complete schema or factual content.
- Response and retrieval scores are uncalibrated LLM self-assessments.
- Query expansion, HyDE, chunk ordering, and extractive compression are inactive by default.
- Dependencies and the Neo4j image are not fully version-pinned.
- There are no automated retrieval-quality or medical-correctness tests.

## Privacy limitations

Interaction logging is disabled by default. If `ENABLE_CHAT_LOGGING=true`, the application writes raw questions, retrieved content, answers, and telemetry to `Chat_History.log` in plaintext. It does not provide consent management, encryption at rest, a retention policy, data-subject controls, or audit-grade access controls.

Conversation state is held in each Streamlit session, while the retriever is cached across sessions. When logging is enabled, raw interaction details go to one shared plaintext log, so the current design is suitable for a controlled demonstration, not a shared service.

## Operational limitations

- Graph restoration is opt-in through `RESTORE_GRAPH_SNAPSHOT=true`; when enabled, it deletes all nodes and drops existing indexes and constraints before importing the local Cypher snapshot.
- The default Neo4j password in Docker Compose is a demonstration credential.
- The tracked Compose file installs APOC Core only, while graph restoration requires an APOC Extended procedure on current Neo4j releases.
- Compose exposes Neo4j ports on all host interfaces and bind-mounts the repository read-write into the app container.
- Ollama is expected at a Docker Desktop host bridge address.
- A fresh clone lacks several ignored runtime artifacts.
- Loading local pickle/joblib metadata is safe only when the artifact source is trusted.
- Neo4j readiness is not protected by an application-level health check.

## Responsible demonstration checklist

Before a demo or portfolio recording:

1. Use only fictional, non-identifying questions.
2. Use a dedicated local Neo4j database.
3. Confirm the artifact sources and checksums.
4. Remove or redact old logs.
5. State visibly that the system is educational and not medical advice.
6. Avoid presenting the LLM scores as confidence or accuracy.
7. Do not claim clinical validation, production readiness, or diagnostic performance.

## What would be required for a stronger system

Meaningful progress toward a safety-focused medical-information system would require, at minimum:

- A versioned, reviewed evidence corpus with source citations.
- Medical-domain evaluation sets and clinician-led error analysis.
- Emergency escalation and abstention behavior.
- Medication, interaction, contraindication, and dosage safeguards.
- Source-grounded answers with traceable provenance.
- Session isolation, authentication, encryption, retention controls, and security review.
- Reproducible dependencies, automated tests, monitoring, and deployment hardening.

Those items are future engineering requirements, not capabilities of the current repository.
