# Setup and reproducibility

This guide documents the current setup honestly: a fresh clone contains the source and selected index snapshots, but it does not contain every generated artifact required at runtime.

## Reproducibility status

Full retrieval behavior expects these files at fixed root-relative paths:

```text
Semantic_Model.pkl
Keyword_Model.pkl
FAISS_Database/index.faiss
FAISS_Database/index.pkl
neo4j.cypher
```

`Semantic_Model.pkl`, `Keyword_Model.pkl`, and `FAISS_Database/index.pkl` are ignored by Git in the current repository. They must be generated locally or supplied through a trusted artifact channel for the corresponding retrieval paths to work. The app can start with a missing path, but it warns when that path is unavailable and abstains from a medical answer if no usable context remains.

## Prerequisites

- Git and Git LFS
- Python 3.11
- A Java runtime compatible with the installed PySpark release
- Docker Desktop and Docker Compose
- Neo4j with compatible APOC Core and APOC Extended plugins for graph restoration
- Ollama running on the host
- Network access for initial Python, Hugging Face, NLTK, and Ollama downloads

The dependency file is currently unpinned. For a reproducible release, capture and test a version-locked environment before publishing binary artifacts.

## Clone and create the environment

```bash
git clone https://github.com/Dochikhoa2006/SympScan-Advanced-Medical-RAG-Knowledge-Graph-System.git
cd SympScan-Advanced-Medical-RAG-Knowledge-Graph-System

git lfs install
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt wordnet
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Obtain the source dataset

Download the [SympScan – Symptoms to Disease dataset](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease) under its published terms. Create a root-level `SympScan/` directory containing these filenames:

```text
SympScan/
├── Diseases_and_Symptoms_dataset.csv
├── description.csv
├── diets.csv
├── medications.csv
├── precautions.csv
└── workout.csv
```

The raw source directory is ignored so dataset redistribution remains separate from the software repository.

## Build the local retrieval artifacts

Run each stage from the repository root.

### 1. Preprocess to Parquet

```bash
python Raw_Dataset_PreProcess.py
```

Expected output:

```text
Processed_Dataset.parquet/
```

### 2. Build chunks and BM25/semantic objects

```bash
python Hybrid_Dual_Indexing.py
```

Expected outputs:

```text
Chunks.pkl
Keyword_Model.pkl
Semantic_Model.pkl
```

This stage loads `all-MiniLM-L6-v2` and may download model files on first use.

### 3. Build the FAISS store

```bash
python Vector_Database.py
```

Expected outputs:

```text
FAISS_Database/index.faiss
FAISS_Database/index.pkl
```

Both FAISS files are required. `index.faiss` stores the vector index; `index.pkl` stores LangChain docstore metadata and the ID mapping.

## Neo4j snapshot

The tracked `neo4j.cypher` file is managed through Git LFS. Graph restoration is disabled during normal retriever initialization. Confirm that LFS materialized the file rather than leaving a small pointer before explicitly restoring it:

```bash
git lfs ls-files
ls -lh neo4j.cypher
```

Rebuilding the graph snapshot requires all of the following:

- The processed Parquet dataset and `Chunks.pkl`.
- A Neo4j instance with compatible APOC Core and APOC Extended plugins enabled.
- A Python/PySpark environment with Java available.
- Network name resolution for the implementation's default `my-neo4j` host, or an explicit compatible execution environment.

The repository does not yet provide a portable, one-command graph regeneration environment. `Knowledge_Graph.py` constructs the graph and then exports it through APOC when those prerequisites are satisfied.

> [!CAUTION]
> Setting `RESTORE_GRAPH_SNAPSHOT=true` deletes all nodes and drops existing indexes and constraints in the configured Neo4j database before importing the tracked snapshot. Use only a dedicated disposable database.

The restore path checks that the snapshot is present and is not a Git LFS pointer, then checks that `apoc.cypher.runFile` is available before deleting data. These checks do not guarantee that the import will succeed. The tracked Compose file installs APOC Core with `NEO4J_PLUGINS=["apoc"]`, but current Neo4j releases provide `apoc.cypher.runFile` through APOC Extended. Graph restoration is therefore not a verified path with the current unpinned `neo4j:latest` definition until a compatible Extended plugin is installed and tested.

## Prepare Ollama

Start Ollama on the host, then pull the exact model referenced by the tracked application:

```bash
ollama pull qwen2.5:0.5b-instruct-q5_k_m
```

Set `OLLAMA_MODEL` to another pulled model name to use it for both query processing and answer generation. The default is `qwen2.5:0.5b-instruct-q5_k_m`.

The application container reaches Ollama at `http://host.docker.internal:11434`. Docker Desktop supplies this hostname on macOS and Windows. Additional host-gateway configuration may be required on Linux.

## Start the application

After all required artifacts exist and a compatible Neo4j/APOC Core/Extended environment is available:

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

Graph retrieval requires an already populated Neo4j database. With a compatible Neo4j/APOC Core/Extended installation and a dedicated disposable database, explicitly restore the snapshot by starting the app with:

```bash
RESTORE_GRAPH_SNAPSHOT=true docker compose up --build
```

Do not use this setting against a database containing data you need to keep. Leave it unset for subsequent starts so the graph is not reset again.

The application may download the cross-encoder during the first startup if it is not cached. The `all-MiniLM-L6-v2` embedding model is acquired during artifact construction and subsequently loaded through `Semantic_Model.pkl`. Initial loading can take about one minute or longer depending on the host and cache state.

The application reads `NEO4J_URI`, `OLLAMA_BASE_URL`, and `OLLAMA_MODEL` from the environment. If unset, it uses `bolt://my-neo4j:7687`, `http://host.docker.internal:11434`, and `qwen2.5:0.5b-instruct-q5_k_m`, respectively. Constructor arguments can override these settings when using the Python classes directly.

## Verification checklist

Run these non-destructive checks before publishing a release:

```bash
python -m compileall -q *.py
docker compose config --quiet
git lfs ls-files
```

Then verify manually:

1. Streamlit loads without an exception.
2. Neo4j starts with APOC available.
3. A simple medical-information query retrieves context and renders a response.
4. A greeting follows the chitchat path.
5. No personal health information appears in `Chat_History.log` before sharing logs or screenshots.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| FAISS load reports a missing file | `FAISS_Database/index.pkl` was not built or supplied |
| Joblib load fails | A required model pickle is missing, incompatible, or untrusted |
| Ollama connection fails | Ollama is stopped, the model is absent, or the host bridge is unavailable |
| Neo4j import fails | APOC Extended, the LFS snapshot, credentials, version compatibility, or database readiness is missing |
| PySpark cannot start | Java is unavailable or incompatible with the installed PySpark version |
| First request is slow | The cross-encoder or other runtime resources are loading or downloading |
