# RAG Chatbot for Self-Study

A self-study AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) and adapts to any domain or subject matter.

## Features

### 🎓 Domain Expert Mode

- Ask questions about one or multiple provided PDF documents
- Get contextual answers based on your imported PDF materials (local and/or S3)
- Conversational interface with chat history
- Configurable for any domain (medical, legal, technical, academic, etc.)

### 📝 Exam Prep Mode

- Request quiz questions on specific topics or sections
- Receive immediate feedback on your answers
- Adaptive questioning to avoid repetition
- Customizable for any exam or certification

## Technology Stack

- **LLM**: Together AI or Ollama (select via `LLM_PROVIDER`)
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers (configurable via `config/params.env`)
- **Framework**: LangChain
- **PDF Processing**: PyMuPDF (fitz) or Docling (select via `RAG_PREPROCESSOR`)
- **Source Loading**: Local files and S3-backed PDFs via `boto3`

## Installation

### Prerequisites

- Python 3.8 or higher
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

   - `config/params.env` (untracked, example provided): set your PDF path, models, chunking, retrieval, and RAGAS settings here.
     - Important: customize the chatbot for your use case by updating:
       - CHATBOT_ROLE
       - USE_CASE
   - Add secrets to `.env` (untracked, example provide): set `TOGETHER_API_KEY=` when using Together AI.
     - If you use S3 sources, also set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
   - Choose your LLM provider in `config/params.env` via `LLM_PROVIDER=together` or `LLM_PROVIDER=ollama`.

   The app always loads `.env` first and `config/params.env` second (no profiles to manage).

## Usage

### Running the Application

From the repo root, after activating your virtualenv:

```bash
python -m src.main
```

### First Run

On the first run, the application will:

1. Process your PDF document
2. Split text into chunks
3. Create embeddings
4. Store vectors in FAISS database

Subsequent runs will load the existing vector database. Delete the database if you want to rebuild context from different source documents.

### Source Files

- `PDF_PATH` supports a comma-separated list of source PDF paths.
- Sources can be local file paths and/or S3 paths (`s3://...`).
- Example:

```env
PDF_PATH=data/guide.pdf,data/appendix.pdf,s3://my-docs/training/reference.pdf
```

⚠ Warning: if S3 URLs are detected, cleanup logic recreates `AWS_TEMP_FOLDER` and deletes everything currently inside it on each startup. Do not place permanent files there.

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

### Operational Modes

#### Domain Expert Mode (Option 1)

```
❓ Your question: What are the key principles of risk-based testing?
💡 Answer: Based on the context, risk-based testing involves...
```

#### Exam Prep Mode (Option 2)

```
🧠 Section / topic: Risk Management
❓ Your question: What are the main categories of product risks in software testing?
❓ Your answer: Functional failures and quality issues
💡 Answer: Good start! The main categories include functional failures...
```

## Project Structure

```
rag-chatbot/
|- README.md
|- requirements.txt
|- .env.example
|- .gitignore
|- config/
|  \- params.env          # Tunable, tracked runtime parameters
|- src/
|  |- main.py             # Main application entry point
|  |- domain_expert.py    # Domain expert logic
|  |- exam_prep.py        # Exam prep logic
|  \- prompts.py          # System and sentence condense prompts
|- data/
|  \- your_document.pdf   # Place any file you want to use for context here
\- faiss_db/              # Vector store - autogenerated upon first run.
```

## Configuration Options

- Secrets live in `.env` (untracked): `TOGETHER_API_KEY` (Together AI only).
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
| `DB_DIR`          | `faiss_db`                                      | Directory for vector database         |
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

## Dependencies

Listed in `requirements.txt` file.

## Commands

During chat sessions:

- Type your question or topic normally
- `mode` - Return to operational mode selection
- `quit`, `exit`, `no`, or `stop` - Exit the application
- `Ctrl+C` - Force quit

## Testing

Standard suite: `pytest` (eval tests are marked and excluded by default, see below). Make sure `.env` and `config/params.env` exist so env loading succeeds.

### Evals (local only)

Eval tests are disabled by default when running pytest to avoid breaking CI/CD runs as they need a source document and a golden dataset.

Prereq: install dev deps to run evals: `pip install -r requirements-dev.txt`

Dataset schema: JSON array of objects with `question` and `ground_truth` strings (see `tests/data/golden_set.json.example`).

#### RAGAS

Relevant variables in `config/params.env`:
- EVAL_PDF_PATH, EVAL_DB_DIR, EVAL_GOLDEN_SET_PATH
- EVAL_LLM_PROVIDER, EVAL_MODEL_NAME, EVAL_OLLAMA_BASE_URL, EVAL_TOGETHER_API_KEY
- EVAL_RESULTS_DIR, EVAL_*_THRESHOLD, EVAL_*_MIN


How to run:

```bash
pytest -m ragas
```

Expected behavior: if files are missing, tests skip/fail with a clear message; results are saved under EVAL_RESULTS_DIR (CSV/JSON, plus plots if matplotlib/plotly are installed).

#### DeepEval

DeepEval tests log results to MLflow.

Relevant variables in `config/params.env`:
- EVAL_PDF_PATH, EVAL_DB_DIR, EVAL_GOLDEN_SET_PATH
- EVAL_LLM_PROVIDER, EVAL_MODEL_NAME, EVAL_OLLAMA_BASE_URL, EVAL_TOGETHER_API_KEY
- MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

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
- You update `src/core/prompts.py` to refine `domain_expert_prompt`
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

- Delete the `faiss_db` directory to rebuild the database
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
