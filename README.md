# RAG Chatbot for Self-Study

[![Unit Tests](https://github.com/eliaspardo/rag-chatbot/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/eliaspardo/rag-chatbot/actions/workflows/unit-tests.yml)

A self-study AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) and adapts to any domain or subject matter.

## Features

### 🎓 Domain Expert Mode

- Ask questions about one or multiple provided PDF documents
- Get contextual answers based on your imported PDF materials (local and/or S3)
- Conversational interface with chat history
- Configurable for any domain (medical, legal, technical, academic, etc.)

## Technology Stack

- **LLM**: Together AI or Ollama (select via `LLM_PROVIDER`)
- **Vector Database**: Chroma DB
- **Embeddings**: HuggingFace Sentence Transformers (configurable via `config/params.env`)
- **Framework**: LangChain
- **PDF Processing**: PyMuPDF (fitz) or Docling (select via `RAG_PREPROCESSOR`)
- **Source Loading**: Local files and S3-backed PDFs via `boto3`
- **Database**: PostgreSQL (Document Management Service)
- **Contract Testing**: Pact (consumer-driven contracts)

## Architecture

The application is built as a microservices system with the following services.


| Service | Port | Description |
|---------|------|-------------|
| Streamlit | 8501 | Web frontend |
| Inference Service | 8002 | Chat API |
| Ingestion Service | 8003 | Document ingestion API |
| Document Management Service | 8004 | Document status tracking |
| ChromaDB | 8001 | Vector store |
| PostgreSQL (dms-db) | 5432 | DMS database |
| Pact Broker | 9292 | Contract testing (dev only) |
| MLflow | 5000 | Evaluation tracking and visualization (dev only)|
| LocalStack | 4566 | S3 bucket emulation (dev only)|

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker (for running services and integration tests)
  - **Docker Desktop on Linux users**: Testcontainers requires special configuration. Set these environment variables:
    ```bash
    export DOCKER_HOST=unix://~/.docker/desktop/docker.sock
    export TESTCONTAINERS_RYUK_DISABLED=true
    ```
    (If using `direnv`, these are already configured in `.envrc`)
- LLM provider setup:
  - Together AI: API key in `.env` (`TOGETHER_API_KEY`)
  - Ollama: running local server (default `http://localhost:11434`)

### Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd rag-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: install pre-commit hooks**

   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

   To run all hooks on demand:

   ```bash
   pre-commit run --all-files
   ```

4. **Create your secrets file**

   ```bash
   cp .env.example .env
   ```

5. **Create and configure your runtime parameters**

   ```bash
   cp config/params.env.example config/params.env
   ```

   - `config/params.env` (untracked, example provided): set your PDF path, models, chunking, retrieval, and eval settings here.
     - Important: customize the chatbot for your use case by updating:
       - CHATBOT_ROLE
       - USE_CASE
   - Add secrets to `.env` (untracked, example provide): set `TOGETHER_API_KEY=` when using Together AI.
     - If you use S3 sources, also set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
   - Choose your LLM provider in `config/params.env` via `LLM_PROVIDER=together` or `LLM_PROVIDER=ollama`.

   The app always loads `.env` first and `config/params.env` second (no profiles to manage).

## Usage

### Running with Docker Compose (Recommended)

The full stack runs via Docker Compose:

```bash
# Start all services in detached mode
make up
# Or: docker compose up -d

# View logs (attached mode)
make debug
# Or: docker compose up

# Stop all services
make down
# Or: docker compose down
```

### Running Locally (Development)

For local development without Docker:

**Ingestion Service** (run once to build the vector store):
```bash
python -m src.ingestion_service.main
```

**Inference Service** (run to serve the API):
```bash
python -m src.inference_service.main
```

**Document Management Service**:
```bash
python -m src.document_management_service.main
```

**UI Service** (Streamlit frontend):
```bash
streamlit run src/ui_service/streamlit_app.py
```

### Document Management Service

The Document Management Service (DMS) tracks document processing status (pending, completed, error):
- Documents are tracked by path hash
- Status persists in PostgreSQL
- Only new/failed documents are reprocessed
- Single document ingestion available via `POST /ingestion/document/`

### Startup

On startup, the ingestion service will process the PDF documents in PDF_PATH and ingest only the ones that are new/pending.
Delete the database if you want to rebuild context from different source documents.

### Source Files

- `PDF_PATH` supports a comma-separated list of source PDF paths.
- Sources can be local file paths and/or S3 paths (`s3://...`).
- Example:
```env
PDF_PATH=data/guide.pdf,data/appendix.pdf,s3://my-docs/training/reference.pdf
```

⚠ Warning: cleanup logic recreates `AWS_TEMP_FOLDER` and deletes everything currently inside it on each startup. Do not place permanent files there.

### Local S3 with Docker Compose (LocalStack)

For local integration testing of S3-backed sources, this repo includes `docker-compose.yaml` with a LocalStack service.

1. Start LocalStack:

```bash
docker compose up -d
```

2. Configure `config/params.env` for local endpoint:

```env
PDF_PATH=s3://sample-bucket/your_file.pdf
AWS_ENDPOINT_URL=http://127.0.0.1:4566
AWS_REGION=us-east-1
AWS_TEMP_FOLDER=data/temp/
```

3. Configure `.env` credentials for local access:

```env
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

4. Stop LocalStack when done:

```bash
docker compose down
```

Data persists under `./my-localstack-data` as configured in `docker-compose.yaml`.


## Project Structure

```
rag-chatbot/
|- README.md
|- Makefile                    # Development commands
|- docker-compose.yaml         # Service orchestration
|- docker-compose.override.yml # Local dev overrides
|- requirements.txt
|- .env.example
|- .gitignore
|- config/
|  \- params.env              # Tunable runtime parameters
|- docker/                    # Dockerfiles per service
|- src/
|  |- ingestion_service/      # Builds the vector store from PDFs
|  |- document_management_service/  # Document status tracking
|  |- inference_service/      # Serves the chat API
|  |- shared/                 # Shared utilities
|  \- ui_service/             # Streamlit frontend   
|- tests/
|  |- unit/                   # Unit tests (mocked dependencies)
|  |- contract/               # Pact consumer & provider tests
|  \- data/                   # Test fixtures
|- pacts/                     # Generated Pact files (gitignored)
\- data/
   \- your_document.pdf       # Place PDFs here for context
```

## Configuration Options

- Secrets live in `.env` (untracked): `TOGETHER_API_KEY` and `DMS_DATABASE_URL`. Additionally, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` if you're serving S3 files.
- Tunables live in `config/params.env` (tracked): table below.

| Variable          | Default                                         | Description                           |
| ----------------- | ----------------------------------------------- | ------------------------------------- |
| `LLM_PROVIDER`    | `together`                                      | LLM provider: `together` or `ollama`  |
| `OLLAMA_BASE_URL` | `http://localhost:11434`                        | Ollama server URL (Ollama only)       |
| `CHATBOT_ROLE`    | expert tutor                                    | Chatbot's role                        |
| `USE_CASE`        | learn from the provided materials               | Learning goal                         |
| `MODEL_NAME`      | `mistralai/Mistral-7B-Instruct-v0.1`            | LLM model to use                      |
| `PDF_PATH`        | -                                               | Comma-separated PDF paths (local and/or `s3://...`) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Embedding model                       |
| `DB_DIR`          | `chroma_db`                                     | Directory for vector database         |
| `AWS_TEMP_FOLDER` | `data/temp/`                                    | Local temp folder used for downloaded S3 files (cleared on startup) |
| `AWS_REGION`      | -                                               | AWS region for S3 client              |
| `AWS_ENDPOINT_URL`| -                                               | Optional custom S3 endpoint URL (for S3-compatible providers/LocalStack) |
| `CHUNK_SIZE`      | `500`                                           | Text chunk size for processing        |
| `CHUNK_OVERLAP`   | `50`                                            | Overlap between text chunks           |
| `RETRIEVAL_K`     | `4`                                             | Number of relevant chunks to retrieve |
| `TEMPERATURE`     | `0.3`                                           | LLM temperature (creativity)          |
| `MAX_TOKENS`      | `512`                                           | Maximum tokens in LLM response        |
| `RAG_PREPROCESSOR`| `legacy`                                        | PDF preprocessor: `legacy` or `docling` |
| `DOCLING_EXPORT_TYPE` | `doc_chunks`                                 | Docling export: `markdown` or `doc_chunks` |
| `DMS_URL` | `http://localhost:8004` | Document Management Service URL |
| `CHAT_TIMEOUT` | `120` | Seconds to wait for a chat response before timing out (frontend) |

## Dependencies

Listed in `requirements.txt` file and within each service's folder.

## Testing

### Quick Reference

```bash
make test          # Run all tests (unit + contract)
make test-unit     # Run unit tests only
make test-contract # Run contract tests only
make test-e2e      # Run E2E tests (starts services, waits for readiness)
make test-eval     # Run eval tests (needs mlflow container running)
```

### Test Layers

| Layer | Command | What it tests |
|-------|---------|---------------|
| Unit | `pytest tests/unit` | Business logic with mocked dependencies |
| Contract | `pytest tests/contract` | Consumer-driven contracts (Pact) |
| Eval | `pytest -m deepeval` | LLM response quality (see below) |

### Contract Testing (Pact)

This project uses [Pact](https://pact.io/) for consumer-driven contract testing between services.

**Prerequisites:**
- Pact Broker must be running: `docker compose up -d pact-broker`
- Broker UI available at http://localhost:9292
- Install dev deps: `pip install -r requirements-dev.txt`

**Configuration:**

The provider tests read the broker URL from the `PACT_BROKER_URL` environment variable (default: `http://localhost:9292/`). Override this in CI:

```bash
export PACT_BROKER_URL=http://your-broker-host:9292/
```

**Workflow:**
1. Consumer tests generate pact files in `pacts/`
2. Publish to broker: `make pact-publish`
3. Provider tests verify against broker contracts

**Running contract tests:**
```bash
# Consumer tests (generate pacts)
pytest tests/contract/consumer/

# Provider tests (verify against broker)
pytest tests/contract/provider/
```

### E2E Tests (Playwright)
>**⚠️ These tests are meant to be ran locally or on ephemeral environments as they clear the vector and DMS databases**.

UI tests use Playwright to test the Streamlit frontend through browser automation.

**Prerequisites:**
- Install dev deps: `pip install -r requirements-dev.txt`
- Install deps: `sudo playwright install-deps`
- Install browser binaries (required once): `playwright install`


**Running E2E tests:**
```bash
# Recommended: use make target (handles service startup and readiness checks)
make test-e2e

# Or manually:
docker compose up -d
pytest tests/e2e/
```

### Unit Tests

Standard suite: `pytest` (eval tests are marked and excluded by default, see below). Make sure `.env` and `config/params.env` exist so env loading succeeds.

### Evals (local only)

Eval tests are disabled by default when running pytest to avoid breaking CI/CD runs as they need a source document and a golden dataset.

Prereq: install dev deps to run evals: `pip install -r requirements-dev.txt`

Dataset schema: JSON array of objects with `question` and `ground_truth` strings (see `tests/data/golden_set.json.example`).

#### DeepEval

DeepEval tests log results to MLflow.

Relevant variables in `config/params.env`:
- `EVAL_PDF_PATH`, `EVAL_DB_DIR`, `EVAL_GOLDEN_SET_PATH`
- `EVAL_NO_OF_QUESTIONS_TO_TEST` — limit how many questions from the golden set to evaluate; `0` runs the full dataset (useful to set a small number for quick local runs)
- `EVAL_LLM_PROVIDER`, `EVAL_MODEL_NAME`, `EVAL_OLLAMA_BASE_URL`, `EVAL_TOGETHER_API_KEY`
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`

##### MLflow UI Patch (Parent Compare)

This repo includes a small patch to add a custom MLflow compare page (`/compare-parents`) for parent-run child comparisons.

Apply after creating your virtualenv and installing dependencies (uses the active venv if `VIRTUAL_ENV` is set, otherwise defaults to `./.venv`):

```bash
./tools/mlflow_patches/scripts/apply_mlflow_compare_patch.sh
```

Windows (PowerShell):

```powershell
.\tools\mlflow_patches\scripts\apply_mlflow_compare_patch.ps1
```

Notes:
- The script copies `tools/mlflow_patches/patches/mlflow_server_init.py` into the active venv as `mlflow/server/__init__.py` (and backs up any existing file to `tools/mlflow_patches/backup/`).
- If the template is missing, it falls back to applying `tools/mlflow_patches/patches/mlflow-compare-parents.patch`.
- Reapply after recreating the venv or upgrading MLflow.

How to run:

```bash
pytest -m deepeval --run-name "prompt-tweak-2026-01-23"
```

The `--run-name` flag controls the MLflow parent run name. If omitted, the default is `deepeval-YYYY-MM-DD-HH-MM-SS` (UTC).

Example (quick regression after a prompt tweak):
- You update `src/shared/prompts.py` to refine `domain_expert_prompt`
- Run: `pytest -m deepeval --run-name "domain-expert-prompt-v2-2026-01-23"`
- The run and nested per-question results are tracked under that name in MLflow

## Troubleshooting

### Common Issues

**PDF Processing Errors**

- Ensure your PDF path is correct in `config/params.env`
- Check that the PDF is readable and not password-protected

**Together AI API Errors**

- Verify your API key is valid and has sufficient credits
- Check network connectivity

**Ollama Errors**

- Ensure the Ollama server is running at `OLLAMA_BASE_URL`
- Verify the model is available locally (e.g., `ollama list`)

**Memory Issues**

- Reduce `CHUNK_SIZE` if processing large documents
- Consider using smaller embedding models

**Vector Database Issues**

- Delete the `chroma_db` directory to rebuild the database
- Ensure sufficient disk space

### Debugging

Enable verbose mode by adding the `verbose=True` line in the chain configurations:

```python
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True,  # Add for debugging
)
```

## Support

For issues create an issue in the GitHub repository.
