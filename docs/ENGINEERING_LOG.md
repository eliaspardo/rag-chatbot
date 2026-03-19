# Engineering Log
> Informal record of work done, problems solved, and optimizations made. Used as source material for case studies and retrospectives.

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
