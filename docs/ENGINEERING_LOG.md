# Engineering Log
> Informal record of work done, problems solved, and optimizations made. Used as source material for case studies and retrospectives.

---

## 2026-04-10

### Surfacing no-documents 503 error end-to-end (issue #74)
- **Goal**: Make the empty-vector-store failure path informative for users instead of showing a generic error
- **Solution**: Three-layer change — inference service error message made explicit and user-facing; UI client adds `NoDocumentsIngestedError` named exception that intercepts the 503 before `raise_for_status()`; Streamlit caller shows `st.warning` with the server-supplied message
- **Pattern**: Detail string lives in exactly one place (inference service); client and UI derive from it at runtime so there's no string duplication
- **Tests**: Pact consumer test added for the 503 interaction; Pact provider state handler added to verify the exact response body under the "no documents have been ingested" state

---

## 2026-04-09

### LangChain 1.x migration — broken imports, pinned deps, and torch fix
- **Problem**: `make up` failed at runtime with `ModuleNotFoundError` on `langchain.memory`, then `langchain.llms`, as the inference_service started
- **Root cause**: Unpinned requirements — each fresh Docker build pulled the latest langchain, which in 1.x removed `langchain.memory`, `langchain.llms.base`, and moved chain/prompt classes out of the top-level `langchain` package
- **Fix 1 — import migration** in `chain_manager.py`:
  - `langchain.memory.ConversationBufferMemory` → `langchain_classic.memory`
  - `langchain.llms.base.LLM` → `langchain_core.language_models.llms`
  - `langchain.chains.*` → `langchain_classic.chains`
  - `langchain.prompts.PromptTemplate` → `langchain_core.prompts`
  - `langchain.chains.base.Chain` → `langchain_classic.chains.base`
- **Fix 2 — CPU-only torch**: `inference_service` Dockerfile was pulling the full CUDA torch wheel (~530MB) via `sentence-transformers`. Added an explicit CPU-only torch pre-install step (same pattern as `ingestion_service`), reducing the download to ~200MB. Added `--timeout 300 --retries 5` to both Dockerfiles to handle flaky PyPI connections
- **Fix 3 — pinned dependencies**: Captured `pip freeze` from all running containers and pinned direct deps in each service `requirements.txt`. Updated root `requirements.txt` to langchain 1.x and rebuilt local `.venv` to match, resolving conflicts with `instructor` (removed — unused), `openai` (1.x → 2.x), `aiohttp`, `urllib3`, and `langchain-docling` (`<2.0.0` → `==2.0.0`)

---

## 2026-03-20

### Integration Test Data Seeding Pattern - Hybrid Approach
- **Problem**: Integration tests need to pre-seed ChromaDB with test data. Initial approaches had tradeoffs:
  - Pre-seeded fixtures: Hide what data tests use (readability issue)
  - Using app endpoints for setup: Couples test setup to implementation (if ingestion breaks, all tests break)
  - No helper: Duplication across tests
- **Solution**: Hybrid approach combining best practices:
  1. Helper function `seed_chromadb_documents()` - reusable seeding logic
  2. Fixture `chromadb_client` - handles client creation + cleanup
  3. Tests call helper explicitly with specific data
- **Implementation pattern**:
  ```python
  # Fixture provides client + cleanup
  @pytest.fixture
  def chromadb_client(integration_env):
      client = chromadb.HttpClient(...)
      yield client
      client.delete_collection(...)  # Cleanup

  # Test explicitly seeds its data
  def test_health_with_docs(self, chromadb_client, integration_env):
      seed_chromadb_documents(
          chromadb_client,
          texts=["Specific test data"],
          metadatas=[{"source": "test.pdf"}]
      )
      # ... test logic
  ```
- **Why it works**:
  - Flexible: Each test controls its own data
  - Readable: Test clearly shows its dependencies
  - Isolated: No shared state between tests
  - Reusable: Helper eliminates duplication
  - Clean: Fixture handles cleanup automatically
- **Result**: Clear, maintainable integration tests with explicit test data. Pattern is reusable for other services (inference, DMS) that need data seeding.
- **Related**: ADR-043 (integration test strategy)

---

## 2026-03-18

### Autonomous feature removal via Claude Code (case study)
- **Task**: Remove Exam Prep mode from Inference Service and Frontend (Issue #51)
- **Process**: Single prompt → Claude Code autonomously executed entire workflow:
  1. Read GitHub issue via `gh issue view`
  2. Created todo list (12 items) to track progress
  3. Read all affected files to understand scope
  4. Deleted/modified 10 files across 4 directories
  5. Ran tests, verified grep for remaining references
  6. Created commit with descriptive message
  7. Pushed branch, created PR with detailed summary
  8. Added clarifying comment about unrelated file
- **Human involvement**: Zero during execution; review phase only
- **Stats**: 559 lines deleted, 16 lines added (net -543 LOC)
- **Time**: ~5 minutes from prompt to PR
- **Observations**:
  - Todo list helped maintain focus across multi-file changes
  - Pre-commit hooks caught unrelated issues (existing broken test file)
  - Claude correctly scoped changes to issue requirements (no over-engineering)
  - PR description matched issue checklist format for easy review
- **Takeaway**: Well-scoped deletion tasks with clear acceptance criteria are ideal for autonomous execution. The issue's explicit checklist format translated directly into actionable steps.

---

## 2026-03-17

### Removed legacy (non-DMS) ingestion
- Deleted `DMS_ENABLED` feature flag and all legacy code paths
- Removed `prepare_vector_store` and `update_vector_store` from bootstrap.py
- Simplified lifespan.py and main.py - single code path now
- DMS_URL is now required; service fails fast if not set
- Deleted `tests/unit/ingestion_service/test_bootstrap.py` (tested removed code)
- Net result: -211 lines of code

### Batch ingestion response pattern
- Changed `/ingestion/documents/` to return per-document results
- Added `DocumentIngestionResult` dataclass in document_ingestor.py
- Added `BatchIngestionResponse` model with total/succeeded/failed/results
- Updated unit tests to verify return values
- Breaking API change - documented in ADR-040

### Docker build optimization
- **Problem**: Docker rebuilds triggered by unrelated file changes
- **Root cause**: No `.dockerignore`, and `COPY data/` in Dockerfile copied volatile data
- **Fix**: Created `.dockerignore` excluding data/, .git/, __pycache__/, .venv/, docs/
- **Fix**: Removed `COPY data/ data/` from Dockerfile.ingestion_service (data is volume-mounted anyway)
- **Result**: Code changes now only require restart, not rebuild

### Contract test fixes (DMS provider verification)
- **Problem**: `monkeypatch` doesn't work with session-scoped fixtures
- **Solution**: Use `patch.dict(os.environ, ...)` or patch module variables directly
- **Problem**: SQLite in-memory not persisting tables across connections
- **Root cause**: `DMS_DATABASE_URL` read at import time, patch was too late
- **Solution**: Patch the module-level variable: `patch("...lifespan.DMS_DATABASE_URL", ...)`
- **Problem**: `get_db_client` mock not working (it's a generator)
- **Solution**: Use FastAPI's `app.dependency_overrides[get_db_client] = lambda: mock`

### Inference service folder structure refactor
- **Problem**: inference_service had `api/` and `core/` subfolders while other services were flat
- **Goal**: Improve consistency across services while keeping meaningful abstractions
- **Decision**: Remove `api/` folder (flatten HTTP layer), keep `core/` folder (LangChain business logic)
- **Changes**:
  - Moved `api/main.py`, `api/lifespan.py`, `api/bootstrap.py`, `api/session_manager.py` to service root
  - Updated imports in moved files and tests
  - Updated `README.md` run command and `Dockerfile.inference_service` CMD
- **Rationale**: The `api/` folder added navigation overhead for only 4 files. The `core/` folder remains because it groups 4 related LangChain orchestration files that are genuinely distinct from the HTTP layer.
- Other services stay flat — they're simpler and don't need the separation

---

## 2026-03-19

### UI Service - Health Status Component
- **Goal**: Provide visibility into inference service health and document availability before users attempt to chat
- **Implementation**:
  - Added expandable health status panel above chat interface (collapsed by default)
  - Created `InferenceServiceClient` abstraction to decouple UI from direct API calls
  - Renamed `API_BASE_URL` → `INFERENCE_SERVICE_URL` for consistency with other services
  - Used Streamlit's `@st.fragment` decorator for auto-refresh every 30 seconds without full page reload
- **Testing**:
  - Unit tests for InferenceServiceClient (mocked requests)
  - Pact consumer contract tests defining UI → Inference health endpoint expectations
  - Provider verification tests in inference service confirming contract satisfaction
- **Result**: Users can now see service status, ChromaDB document count, and DMS document count before chatting. Reduces confusion when service is up but no documents are loaded.
- **Contract-first approach validated**: Consumer contract defined what UI needs → provider verification drove implementation → full test coverage without manual integration tests

---

## 2026-03-25

### Fixed ChromaDB test data persistence with autouse fixture
- **Problem**: Integration tests in `TestIngestionService` experienced data accumulation across test methods. The final test (`test_ingestion_2_documents`) expected exactly 2 documents but received more because previous tests' data persisted in the shared class-scoped ChromaDB container.
- **Root cause**: The `chromadb_client` fixture included cleanup logic (delete collection after test) but was only used as a parameter by one of four tests. The other three tests wrote data to ChromaDB without cleaning up, causing data to leak into subsequent tests.
- **Solution**: Added `autouse=True` to the `chromadb_client` fixture definition. This makes the cleanup logic run automatically after every test, regardless of whether the test declares the fixture as a parameter.
- **Why autouse over function-scoped container**: Container startup takes 2-5 seconds per test; collection deletion takes <100ms. For 4 tests, autouse cleanup saves 8-20 seconds compared to recreating the container for each test.
- **Result**: Clean state between tests with minimal performance overhead. Tests that need direct ChromaDB access (e.g., for seeding data) still get the client object by declaring the fixture parameter. Tests that only use the ingestion API don't need to change—they automatically benefit from cleanup.
- **Related**: ADR-045 (decision to use autouse fixture), ADR-043 (integration test patterns with testcontainers)

### Ingestion service test coverage expansion
- **Goal**: Increase integration test coverage for error scenarios and service unavailability
- **Work done**:
  - Added tests for DMS unavailable and Vector Store unavailable scenarios
  - Added tests for successful and failed document ingestion
  - Refactored multi-document ingestion to use `ingest_documents` helper
  - Added `make test-integration` target for running integration tests
  - Enabled parallel test execution with pytest-xdist
  - Added ingestion service lifespan testing
  - Removed unused `process_documents` function
- **Result**: More comprehensive integration test suite covering happy path and error conditions. Faster test execution with parallel runs.

---
## 2026-03-29

### Test callback factory refactoring
- **Problem**: Integration tests for ingestion service had 6+ repeated `status_callback` function definitions
  - Each callback: parse request body, dispatch on status (PENDING → 201 with pending response, COMPLETED/ERROR → 204)
  - Only differences were: pending response data and terminal status value
  - ~100 lines of duplicated code across test methods
- **Solution**: Extracted `make_status_callback(pending_response, terminal_status)` factory function
  - Takes pending response data and terminal status as parameters
  - Returns configured callback with pattern baked in
  - Replaced all inline callback definitions with factory calls
- **Implementation**:
  ```python
  def make_status_callback(pending_response, terminal_status):
      def callback(request):
          body = json.loads(request.body)
          if body["status"] == DocumentStatus.PENDING:
              return (201, {}, json.dumps(pending_response))
          elif body["status"] == terminal_status:
              return (204, {}, "")
      return callback
  
  # Usage - before: 15 lines, after: 1 line
  mock_dms.add_callback(
      responses.PUT,
      f"http://localhost:8004/documents/{doc_hash}/status/",
      callback=make_status_callback(dms_documents_pending, DocumentStatus.COMPLETED),
  )
  ```
- **Result**: 
  - Eliminated ~100 lines of duplicated code
  - Each callback definition reduced from 15 lines to 1 line
  - Pattern is now explicit and centralized
  - Future status callback changes only need one place updated
- **Related**: ADR-046 (decision to use factory pattern for test callbacks)

---


---

## 2026-04-14

### MLflow evaluation query tool (issue #78)
- **Goal**: Replace manual MLflow UI browsing with a scriptable CLI for querying parent run summaries and drilling into per-question child run scores
- **Solution**: `tools/mlflow_query.py` — standalone script using `MlflowClient`; `list` sub-command shows recent parent runs with metric means; `show` sub-command shows per-question child runs with optional status and field filters
- **Key details**: SQLite URI resolution walks `__file__` upward to project root (handles any working directory); metric names discovered dynamically from `run.data.metrics`; child runs filtered with backtick-quoted `mlflow.parentRunId` tag; natural sort for `question-N` run names
- **Companion skill**: `.claude/skills/mlflow-evals/SKILL.md` teaches Claude how to invoke and interpret the tool
