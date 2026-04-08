# Architectural Decision Log
> Auto-generated during Claude Code sessions. Used as input for positioning documentation.

---

**ID**: ADR-001
**Date**: 2026-02-27
**Context**: The `VectorStoreBuilder` class instantiated a `chromadb.HttpClient` directly in its constructor, causing test failures when the ChromaDB container wasn't running. Eval tests needed to use an in-memory `EphemeralClient` instead.
**Decision**: Apply constructor injection by adding an optional `chroma_client` parameter to `VectorStoreBuilder.__init__` and its subclasses, with a default fallback to `HttpClient`.
**Rationale**: Constructor injection is the simplest pattern that requires no changes to existing method signatures, maintains backward compatibility, and allows tests to inject an `EphemeralClient` directly.
**Tradeoffs**: Alternative approaches considered: (1) separate `connect()` method (adds state management complexity), (2) factory/strategy pattern (more abstract than needed). Constructor injection chosen for simplicity.
**Tags**: [architecture, testing, dependency-injection, chromadb]

---

**ID**: ADR-002
**Date**: 2026-03-01
**Context**: The ingestion service only supported seeding the vector store at startup via the `PDF_PATH` environment variable. There was no way to add documents at runtime.
**Decision**: Add a POST `/ingestion/documents/` endpoint that accepts a list of document paths (local or S3) and ingests them into the vector store.
**Rationale**: Decouples document ingestion from service deployment. Allows dynamic addition of documents without restarting the service.
**Tradeoffs**: Ingestion is synchronous—large document sets will block the request. Async/background processing deferred for simplicity.
**Tags**: [API, ingestion, architecture]

---

**ID**: ADR-003
**Date**: 2026-03-01
**Context**: The ingestion endpoint needs access to `VectorStoreBuilder` and `FileLoader` instances that were previously scoped to the lifespan function.
**Decision**: Store `vector_store_builder` and `file_loader` in `app.state` during startup for reuse across request handlers.
**Rationale**: FastAPI's `app.state` is the idiomatic way to share dependencies initialized at startup with route handlers without global variables.
**Tradeoffs**: State is mutable and shared—concurrent requests must be careful. Current implementation is stateless per-request so this is acceptable.
**Tags**: [architecture, FastAPI, dependency-management]

---

**ID**: ADR-004
**Date**: 2026-03-01
**Context**: Previously, the service failed to start if `PDF_PATH` was empty or produced no documents. With the new ingestion endpoint, an empty vector store at startup is a valid state.
**Decision**: Allow the service to start with an empty vector store. Log a warning instead of raising `ServerSetupException` when no documents are found at startup.
**Rationale**: The ingestion endpoint enables adding documents post-startup, so an empty initial state is no longer an error condition.
**Tradeoffs**: Service may run with no documents if neither `PDF_PATH` nor the ingestion endpoint are used—requires operational awareness.
**Tags**: [architecture, resilience, startup]

---

**ID**: ADR-005
**Date**: 2026-03-01
**Context**: `FileLoader` previously checked `PDF_PATH` at init to determine if S3 paths were configured, and only required `AWS_TEMP_FOLDER` when S3 was detected. With request-driven ingestion, S3 paths can arrive at any time.
**Decision**: Always require `AWS_TEMP_FOLDER` configuration and initialize the temp directory at startup, regardless of `PDF_PATH` content.
**Rationale**: Cannot predict at startup whether incoming requests will contain S3 paths. Requiring the temp folder upfront avoids runtime configuration failures.
**Tradeoffs**: Local-only deployments must still configure `AWS_TEMP_FOLDER`. Lazy initialization considered but rejected for simplicity.
**Tags**: [configuration, S3, architecture]

---

**ID**: ADR-006
**Date**: 2026-03-02
**Context**: The inference service previously blocked startup waiting for documents in the vector store (polling with retries). This coupled service availability to ingestion timing and prevented independent scaling/deployment.
**Decision**: Decouple inference service startup from document availability. The service starts immediately and returns HTTP 503 on chat endpoints when the vector store is empty. Health endpoint reports document count for observability.
**Rationale**: Allows inference service to start independently of ingestion. HTTP 503 is semantically correct for "service temporarily unavailable" and enables load balancers to route traffic appropriately.
**Tradeoffs**: Clients must handle 503 responses gracefully. Service appears "healthy" but may not be ready to serve requests until documents are ingested.
**Tags**: [architecture, resilience, API, decoupling]

---

**ID**: ADR-007
**Date**: 2026-03-02
**Context**: Ingestion service bootstrap had a single `prepare_vector_store` function handling both startup seeding (from `PDF_PATH`) and runtime ingestion (from API requests), leading to complex branching logic.
**Decision**: Split into two functions: `prepare_vector_store` (startup, uses `PDF_PATH` env var) and `update_vector_store` (API requests, uses provided paths). Renamed `create_vector_store` to `add_documents_to_vector_store` to reflect additive behavior.
**Rationale**: Single Responsibility Principle—each function has one clear purpose. Simplifies testing and reduces cognitive load.
**Tradeoffs**: Slight code duplication in document processing, but clarity outweighs DRYness here.
**Tags**: [architecture, refactoring, ingestion]

---

**ID**: ADR-008
**Date**: 2026-03-02
**Context**: API endpoints accessed Chroma's internal `_collection` attribute to check document count, coupling code to LangChain's implementation details.
**Decision**: Add `get_collection_count()` method to `VectorStoreLoader` and `VectorStoreBuilder` that uses Chroma client's public API. Store loaders in `app.state` for reuse in endpoints.
**Rationale**: Public APIs are stable; private attributes (prefixed `_`) may change without notice. Centralizing the check also eliminates code duplication.
**Tradeoffs**: Adds a method to each class, but provides a stable interface for collection introspection.
**Tags**: [architecture, API, maintainability]

---

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 2 — Document Management Service & Contract Testing
# ═══════════════════════════════════════════════════════════════════════════════

---

**ID**: ADR-009
**Date**: 2026-03-03
**Context**: Document state (pending, processed, failed) is currently implicit—services check ChromaDB directly or infer status from presence of embeddings. No single source of truth exists for document lifecycle, making it difficult to display status in UI or coordinate between services.
**Decision**: Introduce a dedicated Document Management Service that owns document state. Endpoints: update status, get status, list documents. All services query this service instead of inferring state.
**Rationale**: Centralizes document lifecycle management. Enables real-time status display in frontend. Provides clear boundaries for contract testing. Follows the "quality gate" pattern—inference returns 503 if doc management reports no processed documents.
**Tradeoffs**: Adds a new service to deploy and maintain. Introduces network dependency for status checks. Mitigated by caching (30s in inference service).
**Tags**: [architecture, microservices, document-management]

---

**ID**: ADR-010
**Date**: 2026-03-03
**Context**: The system has multiple services communicating over HTTP (ingestion, inference, frontend, document management). Traditional integration tests require all services running, are slow, and fail for unclear reasons.
**Decision**: Adopt Pact for consumer-driven contract testing. Each consumer defines expected HTTP interactions; providers verify they satisfy those contracts. Replace integration tests with contract verification.
**Rationale**: Contracts are versioned and explicit. Tests run in isolation (no network). Failures pinpoint exactly which contract broke. Enables independent deployment of services when contracts are satisfied.
**Tradeoffs**: Requires Pact Broker infrastructure. Team must understand consumer-driven workflow. Initial setup overhead pays off as service count grows.
**Tags**: [testing, contract-testing, pact, microservices]

---

**ID**: ADR-011
**Date**: 2026-03-03
**Context**: Consumer-driven contracts need a central registry for sharing, versioning, and verifying contracts between CI pipelines.
**Decision**: Deploy Pact Broker as a local Docker instance. Consumers publish contracts on every build; providers verify against latest contracts. Use Pact Broker's "Can I Deploy" feature as a deployment gate.
**Rationale**: Pact Broker is the standard solution for contract management. Local Docker instance keeps infrastructure simple for development. "Can I Deploy" prevents deploying a service that would break its consumers or providers.
**Tradeoffs**: Another container to manage. Could use Pactflow (SaaS) in production for reduced ops burden.
**Tags**: [infrastructure, pact, CI/CD, contract-testing]

---

**ID**: ADR-012
**Date**: 2026-03-03
**Context**: Document management service has multiple consumers with different needs: Streamlit (list documents, get status), Ingestion (check if processed, update status), Inference (check documents available).
**Decision**: Define three separate Pact contracts for document management—one per consumer. Each contract specifies only the interactions that consumer requires.
**Rationale**: Consumer-driven contracts should be minimal—each consumer specifies only what it needs. This prevents over-coupling and allows provider flexibility in implementation details not covered by contracts.
**Tradeoffs**: Multiple contracts to maintain. Provider verification runs all consumer contracts, which is correct but increases verification time.
**Tags**: [contract-testing, pact, API-design]

---
---

**ID**: ADR-016
**Date**: 2026-03-04
**Context**: Document Management Service could either validate state transitions (e.g., reject status updates if document isn't in expected prior state) or act as a simple state store where consumers own business logic.
**Decision**: DMS is a simple state store with no state transition validation. Consumers (ingestion, inference) own the business logic for deciding when/how to update status.
**Rationale**: DMS doesn't have context on *why* a status change is happening—only the consumer knows. Different consumers might have different valid transitions. Simpler provider = easier to test and verify contracts.
**Tradeoffs**: Invalid state transitions won't be caught at the DMS level. Consumers must implement their own validation. Acceptable because ingestion service is the only writer.
**Tags**: [architecture, document-management, separation-of-concerns]

---

**ID**: ADR-017
**Date**: 2026-03-04
**Context**: Original design had separate "register document" and "update status" endpoints. This adds complexity and requires consumers to know whether a document exists before updating.
**Decision**: Eliminate separate register endpoint. `PUT /documents/{hash}/status` uses upsert semantics: returns 201 if document is new (created), 204 if document exists (updated).
**Rationale**: Simpler contract—one endpoint handles both cases. Consumer doesn't need to track whether document was previously registered. Idempotent behavior aligns with PUT semantics.
**Tradeoffs**: Cannot distinguish "intentional first registration" from "update"—but this distinction isn't needed for current use cases.
**Tags**: [API-design, document-management, simplification]

---

**ID**: ADR-018
**Date**: 2026-03-04
**Context**: Need to define return types for document management client functions that are minimal but sufficient for consumer logic.
**Decision**: `get_document_status(hash) -> Optional[DocumentStatus]`: returns `None` for 404 (not registered), status value for 200, raises `HTTPError` for 503. `update_document_status(hash, status) -> None`: returns `None` for success (201/204), raises `HTTPError` for 503.
**Rationale**: Minimal return types—consumers only need the status value or success confirmation. Exceptions for infrastructure failures let callers decide retry/fallback strategy. `None` for "not found" allows simple conditional logic: `if status == COMPLETED: skip`.
**Tradeoffs**: No rich error objects. Acceptable for internal service communication where failure modes are limited.
**Tags**: [API-design, client-library, error-handling]

---

**ID**: ADR-019
**Date**: 2026-03-04
**Context**: Pact consumer tests generate JSON pact files. These files accumulate interactions and don't deduplicate when tests change.
**Decision**: Delete and regenerate pact files when contracts change. Do not commit pact files to the repository—publish to Pact Broker from CI instead. Add `pacts/` to `.gitignore`.
**Rationale**: Pact Broker is the source of truth for contracts, not the repo. Committing generated files leads to staleness and merge conflicts. CI publishes fresh pacts on every consumer build; provider CI pulls from broker.
**Tradeoffs**: Local development requires running tests to generate pacts before provider verification. Acceptable since this is the intended workflow.
**Tags**: [pact, workflow, CI/CD, contract-testing]

---

**ID**: ADR-020
**Date**: 2026-03-05
**Context**: Ingestion service DMS integration requires replacing `prepare_vector_store` and `update_vector_store` with new DMS-aware functions. However, merging this work before DMS is deployed would break production since the new code depends on DMS being available.
**Decision**: Use a simple `DMS_ENABLED` environment variable to toggle between legacy and DMS-aware implementations. Legacy code remains the default (flag off). New DMS-aware functions (`ingest_documents`, `ingest_document`) are activated when flag is set.
**Rationale**: Allows merging ingestion integration work independently of DMS deployment. Production stays stable on legacy path. Single toggle point in `main.py` and `lifespan.py` keeps complexity minimal—not a full feature flag system, just a migration toggle.
**Tradeoffs**: Temporary code duplication (legacy + new paths). Must remember to delete legacy code and flag once DMS is stable. Acceptable for a short migration period.
**Tracking**: [#40](https://github.com/eliaspardo/rag-chatbot/issues/40)
**Tags**: [architecture, feature-flag, migration, ingestion, DMS]

---

**ID**: ADR-021
**Date**: 2026-03-05
**Context**: Original design had separate flows for startup (`startup_ingest` with batch `get_documents()` optimization) and request (`request_ingest`). Both flows essentially do the same thing: take a list of paths, check each document's status, and ingest if not completed.
**Decision**: Unify into a single `ingest_documents(paths)` function used by both entry points. Drop the `get_documents()` batch optimization for startup—check each document individually via `get_document_status()` like the request flow.
**Rationale**: Simpler design with one code path. The batch optimization was premature—individual status checks are fast enough. Uniform behavior between startup and request flows reduces cognitive load and testing surface.
**Tradeoffs**: Startup with many documents makes N individual DMS calls instead of 1 batch call. Acceptable for current scale; optimization can be added later if needed.
**Tags**: [architecture, simplification, ingestion, DMS]

---

**ID**: ADR-022
**Date**: 2026-03-05
**Context**: Documents need a unique identifier (hash) for DMS tracking. Two options: content-based hash (hash file contents) or path-based hash (hash the file path/URI). Content-based requires downloading the file before checking status, which wastes bandwidth for already-completed documents—especially problematic for S3-sourced files.
**Decision**: Use path-based hashing. Hash the document path/URI string, not the file contents. Check DMS status before downloading.
**Rationale**: Avoids downloading files just to check if they're already processed. Simpler implementation—no file I/O needed for hash calculation. Trade-off of potential duplicates (same file at different paths) is acceptable for current use case.
**Tradeoffs**: Same file uploaded with different paths will be processed multiple times. Acceptable operationally; true content deduplication can be added later if needed.
**Tags**: [architecture, simplification, ingestion, hashing]

---

**ID**: ADR-023
**Date**: 2026-03-05
**Context**: Documents in PENDING status could be either actively processing or leftovers from a crashed process. Need to decide how to handle them.
**Decision**: Treat PENDING same as not-found — re-ingest. Only COMPLETED documents are skipped.
**Rationale**: Simple, handles crash recovery. Concurrent duplicate processing is acceptable (not data loss). Avoids complexity of timestamps or coordination.
**Tradeoffs**: Risk of duplicate work if two processes handle same doc. Mitigate by using doc hash as ChromaDB ID to overwrite instead of duplicate.
**Tags**: [architecture, ingestion, resilience, idempotency]

---

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 3 — DMS Provider Verification & Two-Layer TDD
# ═══════════════════════════════════════════════════════════════════════════════

---

**ID**: ADR-024
**Date**: 2026-03-09
**Context**: Provider state handlers in Pact tests need to set up the system for each interaction. Could either describe states in database terms ("row exists in table X") or business terms ("document is registered with status pending").
**Decision**: Provider states use business semantics, not database internals. State names like "Document abc-123 is DocumentStatus.COMPLETED" describe the observable behavior, not the implementation.
**Rationale**: Business-semantic states are stable across implementation changes. If we switch from SQLite to Postgres, state names don't change. Makes contracts readable by non-developers. Aligns with Pact best practice of describing "what" not "how".
**Tradeoffs**: State handler implementations must translate business semantics to actual setup logic. Acceptable complexity in test code.
**Tags**: [contract-testing, pact, provider-verification, testing]

---

**ID**: ADR-025
**Date**: 2026-03-09
**Context**: When DMS returns 404 for an unknown document, the client could either throw an exception or return a sentinel value.
**Decision**: 404 for unknown documents returns `None` from the client. It's an expected business scenario (document not yet registered), not an error condition.
**Rationale**: Callers can use simple conditional logic: `if status is None: register_document()`. Exceptions are reserved for infrastructure failures (503) that callers can't handle gracefully. Aligns with Python's "explicit is better than implicit" and common patterns like `dict.get()`.
**Tradeoffs**: Callers must check for `None`. Type signature `Optional[DocumentStatus]` makes this explicit.
**Tags**: [API-design, error-handling, client-library]

---

**ID**: ADR-026
**Date**: 2026-03-09
**Context**: Provider verification tests need to pull contracts. Options: local pact files, or Pact Broker. Multiple consumers will eventually exist (ingestion, inference, Streamlit).
**Decision**: Provider verification uses `broker_source()` to pull pacts from the broker. One test class per consumer (filtered by consumer name) maintains logical separation while keeping the broker in the verification loop.
**Rationale**: Broker is the source of truth. Pulling from broker ensures verification runs against the same contracts used for "Can I Deploy" checks. Per-consumer test classes allow targeted debugging when a specific consumer's contract fails.
**Tradeoffs**: Requires broker to be running for provider tests. Acceptable since broker is infrastructure (always-on in docker-compose).
**Tags**: [contract-testing, pact, provider-verification, CI/CD]

---

**ID**: ADR-027
**Date**: 2026-03-09
**Context**: Provider verification tests need to inject mock dependencies into the DMS FastAPI app. Options: monkey-patching, dependency overrides, or app.state injection.
**Decision**: Mock database client is injected into the FastAPI app via `app.state.db_client`. Provider state handlers configure mock return values per interaction.
**Rationale**: Consistent with existing pattern (ADR-003) of storing dependencies in `app.state`. State handlers have direct access to configure mocks before each interaction runs. No complex DI framework needed.
**Tradeoffs**: State handlers must know the mock API. Coupling is acceptable for test code.
**Tags**: [testing, dependency-injection, FastAPI, provider-verification]

---

**ID**: ADR-028
**Date**: 2026-03-09
**Context**: Pact Broker could be started per-test-session or run as always-on infrastructure in docker-compose.
**Decision**: Broker is infrastructure—always-on in docker-compose with SQLite persistence (named volume). Not a per-session test dependency.
**Rationale**: Broker maintains verification history needed for "Can I Deploy". Restarting per session would lose this state. Always-on matches production-like behavior and simplifies CI pipeline.
**Tradeoffs**: Developers must run `docker-compose up` before running contract tests. Acceptable overhead; documented in README.
**Tags**: [infrastructure, pact, docker, CI/CD]

---

**ID**: ADR-029
**Date**: 2026-03-10
**Context**: Provider state handlers initially had separate functions for each status (document_completed, document_error, document_pending) that differed only by the status value passed to the mock.
**Decision**: Consolidate into one generic handler function using `functools.partial` to bind the status parameter. State handler dictionary uses partial application to create per-state callables.
**Rationale**: Eliminates code duplication. Adding new status values requires only a new dictionary entry, not a new function. Makes the pattern explicit: state handlers differ by data, not by logic.
**Tradeoffs**: Slightly more abstract—readers must understand partial. Acceptable given Python developers' familiarity with functools.
**Tags**: [testing, pact, provider-verification, code-quality]

---

**ID**: ADR-030
**Date**: 2026-03-10
**Context**: Need to distinguish between "record doesn't exist because caller asked for something that might not exist" vs "record should exist but is unexpectedly missing".
**Decision**: Return `None` for expected business scenarios where absence is a valid outcome (e.g., querying status of an unregistered document). Raise exceptions for genuinely unexpected missing records (e.g., referential integrity violations, corrupted state).
**Rationale**: `None` enables simple conditional flows (`if status is None: register()`). Exceptions signal programmer errors or system failures that shouldn't be silently handled. Matches Python idioms like `dict.get()` vs `dict[key]`.
**Tradeoffs**: Requires discipline to categorize each "not found" case correctly. Document the convention clearly in client interfaces.
**Tags**: [API-design, error-handling, conventions]

---

**ID**: ADR-031
**Date**: 2026-03-10
**Context**: Pact provider verification runs all interactions from a consumer contract. Need to understand the verification result granularity.
**Decision**: Acknowledge that Pact Broker records verification at the pact level, not per interaction. All interactions in a contract must pass for verification to succeed—no partial credit.
**Rationale**: A contract is a unit of trust. If any interaction fails, the provider cannot be trusted to satisfy the consumer's expectations. Partial success would create false confidence and complicate "Can I Deploy" logic.
**Tradeoffs**: A single failing interaction blocks the entire verification. Requires fixing all failures before progress is recorded. This is the intended behavior—contracts are all-or-nothing.
**Tags**: [contract-testing, pact, pact-broker, verification]

---

**ID**: ADR-032
**Date**: 2026-03-10
**Context**: Consumer contract tests initially included interactions for 503 responses (e.g., "DMS is returning 503"). Question arose whether infrastructure failures belong in Pact contracts.
**Decision**: Remove 503 error scenarios from Pact contracts. Contract tests model agreed-upon business behaviors, not infrastructure failures. A 503 means "provider unavailable"—the provider never intentionally designs a "return 503" behavior; it either responds or it doesn't.
**Rationale**: Pact tests what the provider *promises to do*, not what happens when the provider *can't do anything*. How the consumer handles provider outages (retry, circuit-break, fail gracefully) is internal resilience logic—tested with unit tests where the HTTP client is stubbed to raise connection errors or return 503, then assert the consumer behaves correctly.
**Tradeoffs**: Consumer resilience to provider failures must be tested separately (unit/integration tests), not via Pact. This is the correct separation—contracts define the happy path and expected business error responses (404, 400), not infrastructure failures.
**Tags**: [contract-testing, pact, testing-strategy, error-handling]

---

**ID**: ADR-033
**Date**: 2026-03-10
**Context**: DB client unit tests need a clean database state for each test. Options: (1) use module/session-scoped fixtures with explicit cleanup/teardown after each test, or (2) use function-scoped fixtures that create a fresh instance per test.
**Decision**: Use function-scoped fixtures for DB client unit tests. Each test gets a fresh DB client instance, eliminating the need for cleanup logic.
**Rationale**: Simpler test code—no teardown, no risk of state leaking between tests, no ordering dependencies. Tests are fully isolated by default. The slight overhead of re-instantiating per test is negligible for unit tests.
**Tradeoffs**: Cannot share expensive setup across tests. Acceptable for DB client tests where instantiation is cheap. For integration tests with real databases, session-scoped fixtures with cleanup may be more appropriate.
**Tags**: [testing, unit-tests, fixtures, isolation]

---

**ID**: ADR-034
**Date**: 2026-03-11
**Context**: DMS uses SQLAlchemy for database access. Need to decide when to create the engine, sessionmaker, and individual sessions to balance performance with request isolation.
**Decision**: Engine and sessionmaker are created once at app startup (stored in `app.state` or module-level). Sessions are created per request and closed when the request ends.
**Rationale**: Engine creation is expensive (connection pool setup, dialect initialization). Creating it once at startup amortizes this cost. Sessions are lightweight and should be request-scoped to avoid shared state between requests—each request gets a clean session with its own transaction boundary.
**Tradeoffs**: Requires discipline to close sessions properly (use context managers or FastAPI dependencies with cleanup). Connection pool is shared across requests, which is the desired behavior for efficiency.
**Tags**: [architecture, database, SQLAlchemy, DMS]

---

**ID**: ADR-035
**Date**: 2026-03-11
**Context**: The `GET /documents/` endpoint currently returns all documents without pagination. As document count grows, this could become a performance issue.
**Decision**: Defer pagination implementation. Flag as future improvement—add `limit` and `offset` query parameters before handling meaningful document volumes.
**Rationale**: At current project scale, returning all documents is not a bottleneck. Premature optimization adds complexity without immediate benefit. The endpoint design can accommodate pagination later without breaking changes (additive query parameters).
**Tradeoffs**: Risk of performance issues if document count grows unexpectedly before pagination is added. Acceptable for portfolio/development phase—monitor and implement when needed.
**Tracking**: [#46](https://github.com/eliaspardo/rag-chatbot/issues/46)
**Tags**: [API-design, DMS, technical-debt, deferred]

---

**ID**: ADR-036
**Date**: 2026-03-12
**Context**: DMS has multiple test layers (unit, contract, integration). Need to define what each layer is responsible for to avoid duplicate coverage and ensure each test type adds unique value.
**Decision**: Establish clear test layer responsibilities:
- **Unit tests** (mock dependencies, test functions directly): Infrastructure failures (503 path), internal business rules (e.g., `DocumentHashConflictException` logic in DBClient), any logic not observable via HTTP.
- **Contract tests** (Pact, consumer-driven): Consumer-facing HTTP behavior—201 on create, 204 on update, 409 on conflict, response shapes the consumer depends on.
- **Integration tests** (TestClient + Testcontainers): Happy paths end-to-end with real DB, 404 when document doesn't exist, anything requiring the full FastAPI stack (middleware, exception handlers, response serialization).
**Rationale**: Each layer tests what it's uniquely positioned to test. Unit tests are fast, isolated, and can force edge cases (DB exceptions) hard to trigger in integration. Contract tests verify the provider's promise to consumers, not internal implementation. Integration tests validate the full stack with real infrastructure. Guiding principle: don't duplicate coverage across layers without a reason.
**Tradeoffs**: Some behaviors (e.g., 409 conflict) appear in multiple layers—unit tests the business rule, contract tests the HTTP mapping. This is acceptable as they test different concerns. Requires discipline to categorize new tests correctly.
**Tags**: [testing, testing-strategy, unit-tests, contract-testing, integration-tests]

---

**ID**: ADR-037
**Date**: 2026-03-12
**Context**: DBClient returns raw values from the database. If the status column contains a malformed value, `DocumentStatus(value)` raises `ValueError`. Question: should the endpoint catch this and return 503?
**Decision**: Let malformed status values bubble up as HTTP 500. Do not catch `ValueError` from enum conversion.
**Rationale**: A malformed status in the database is a data integrity issue—the system is broken, not temporarily unavailable. HTTP 503 means "try again later" (transient failure); HTTP 500 means "something is wrong" (retrying won't help). Corrupt data is a 500 scenario. Masking it as 503 would be semantically incorrect and could hide bugs. Aligns with ADR-030: raise exceptions for genuinely unexpected states.
**Tradeoffs**: Unhandled exceptions surface as 500 with default FastAPI error response. Acceptable—data corruption should fail loudly, not be silently handled.
**Tags**: [error-handling, API-design, DMS, data-integrity]

---

**ID**: ADR-038
**Date**: 2026-03-15
**Context**: API endpoints had inconsistent exception handling. Some logged generic messages ("DB operation failed"), some didn't log at all, and some raised custom exceptions that weren't HTTPExceptions. When unexpected errors occurred (e.g., DMS URL misconfiguration), no useful information appeared in logs, making debugging difficult.
**Decision**: Standardize exception handling across all FastAPI endpoints with a two-part pattern:
1. **Specific exceptions first**: Catch known exceptions (SQLAlchemyError, ValidationError, custom business exceptions) with specific HTTP status codes (503 for infrastructure, 409 for conflicts, 404 for not found).
2. **Generic catch-all last**: Every endpoint ends with `except Exception as e: logger.error(e); raise HTTPException(status_code=500, detail="Processing failed")`.
**Rationale**:
- **Log the real error**: `logger.error(e)` captures the actual exception for debugging in logs/monitoring.
- **Return generic message to client**: "Processing failed" avoids leaking internal details (stack traces, connection strings, service names) to API consumers.
- **Consistent HTTP semantics**: 503 for infrastructure failures (DB, external services), 500 for unexpected errors, specific codes for business errors.
**Tradeoffs**: Generic 500 responses make client-side debugging harder—but that's intentional for security. Developers use server logs for root cause analysis. The pattern adds boilerplate but provides consistency and observability.
**Tags**: [error-handling, API-design, logging, security, observability]

---

**ID**: ADR-039
**Date**: 2026-03-17
**Context**: ADR-020 introduced a `DMS_ENABLED` feature flag to allow merging DMS integration work while keeping legacy ingestion as the default. The flag was explicitly marked as temporary, with a note to delete legacy code once DMS was stable. DMS has been stable in development for two weeks.
**Decision**: Remove the `DMS_ENABLED` feature flag and all legacy (non-DMS) ingestion code. DMS is now the only ingestion pathway. `DMS_URL` environment variable is required—service fails to start if not set.
**Rationale**: Feature flags are technical debt. The migration period is complete; DMS has proven stable. Keeping two code paths increases maintenance burden and cognitive load. Making DMS_URL required fails fast with a clear error rather than silently misbehaving.
**Tradeoffs**: Breaking change for anyone using `DMS_ENABLED=false`. Acceptable because this was always a temporary migration toggle, not a supported configuration.
**Closes**: ADR-020 (tracking issue #40)
**Tags**: [architecture, migration, cleanup, ingestion, DMS]

---

**ID**: ADR-040
**Date**: 2026-03-17
**Context**: The `POST /ingestion/documents/` endpoint returned a simple `{success: true, message: "..."}` response. When batch ingestion included multiple documents, partial failures (some succeed, some fail) were invisible to callers—they only saw overall success/failure.
**Decision**: Change response model to `BatchIngestionResponse` with per-document results:
```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    {"document": "a.pdf", "success": true, "error": null},
    {"document": "b.pdf", "success": false, "error": "File not found"},
    {"document": "c.pdf", "success": true, "error": null}
  ]
}
```
**Rationale**: Callers need visibility into which documents failed and why. The previous response hid partial failures, making debugging difficult. Per-document results enable retry logic, progress tracking, and clear error reporting in UIs.
**Tradeoffs**: Breaking API change—consumers must update to handle the new response shape. HTTP status remains 200 even for partial failures (results contain the failure details). Alternative considered: return 207 Multi-Status, but decided against it as it adds complexity and most HTTP clients don't handle 207 specially.
**Tags**: [API-design, ingestion, error-handling, breaking-change]

---

**ID**: ADR-041
**Date**: 2026-03-17
**Context**: Inference service is being integrated with Document Management Service (DMS) to enable future features like source citation. Question arose whether DMS should be used as an additional readiness gate for chat endpoints (alongside the existing ChromaDB check from ADR-006).
**Decision**: DMS integration is used for **observability only** (health endpoint), not as a request gate. Chat endpoints (`/chat/domain-expert/`) continue to use only the ChromaDB document count check as the readiness dependency.
**Rationale**:
- **Different purposes**: ChromaDB check answers "can I answer questions?" (operational readiness). DMS check answers "what's in the inventory?" (metadata/observability).
- **Latency**: DMS check is an HTTP call; adding it to every chat request adds latency.
- **Coupling**: If DMS is temporarily down, users shouldn't be blocked from chatting when ChromaDB is fully loaded and functional.
- **Redundancy**: Documents in ChromaDB must have come through DMS—checking both is redundant for gating purposes.
- **Future use**: DMS will be used for source citation (retrieved from RAG chain's `return_source_documents`), not as a per-request dependency.
**Tradeoffs**: Health endpoint shows DMS status but chat availability doesn't depend on it. A mismatch between DMS and ChromaDB state won't block chat—acceptable because ChromaDB is the authoritative source for "can I answer?".
**Related**: ADR-006 (original ChromaDB readiness gate decision)
**Tags**: [architecture, inference, DMS, observability, decoupling]

---

**ID**: ADR-042
**Date**: 2026-03-18
**Context**: Development workflow relied on manual implementation of features and fixes. As project complexity grew with multiple microservices, the time spent on repetitive tasks (writing tests, updating docs, fixing linter errors) increased. Need a way to accelerate development while maintaining code quality.
**Decision**: Integrate Claude Code as a GitHub Actions workflow. Claude can be invoked by mentioning @claude in PR or issue comments. Claude has access to full repository context and can create branches, commits, and comments. Allowed tools are limited to file operations by default—additional tools (like running tests) must be explicitly approved in workflow config.
**Rationale**: Claude Code provides autonomous task execution for well-defined work: feature implementation, test writing, refactoring, bug fixes. The GitHub Actions integration keeps Claude's work visible (all runs in action history) and auditable (all changes via PRs). Limiting tool access by default provides safety—explicit approval required for potentially destructive operations.
**Tradeoffs**: Introduces AI-generated code into the codebase—requires human review of all Claude PRs. GitHub Actions minutes usage increases (free tier: 2000 min/month; Claude runs can be 5-10 min each). Team must learn how to write effective prompts and review AI-generated changes critically.
**Tags**: [tooling, automation, workflow, Claude-Code]

---

**ID**: ADR-043
**Date**: 2026-03-20
**Context**: Need integration tests for ingestion service that sit between unit tests (fully mocked) and E2E tests (all services). ADR-036 defined test layer responsibilities but didn't specify implementation patterns for integration tests.
**Decision**: Use FastAPI TestClient + ChromaDB testcontainer + HTTP-mocked DMS. TestClient runs the full FastAPI app in-process, ChromaDB runs in testcontainer (real database), DMS is mocked at HTTP level using `responses` library.
**Rationale**: TestClient exercises full FastAPI stack (middleware, exception handlers, lifespan) without HTTP overhead. ChromaDB testcontainer provides real database behavior (embeddings, vector search) without manual infrastructure. HTTP-mocked DMS is sufficient because contract tests already verify DMS integration. Class-scoped testcontainer fixture balances test isolation (per-class cleanup) with speed (container reuse). **Test Data Seeding Pattern**: Tests use a hybrid approach for pre-seeding ChromaDB - helper functions (`seed_chromadb_documents`) contain seeding logic, fixtures (`chromadb_client`) handle client lifecycle and cleanup, and tests explicitly call helpers with specific data. This provides flexibility (each test controls its own data), readability (test shows what data it uses), and isolation (no shared state). Rejected alternatives: pre-seeded fixtures (hide test data), using app endpoints for setup (couples tests to implementation).

---
**ID**: ADR-044
**Date**: 2026-03-24  
**Context**: Encountered MLflow database corruption with error `Can't locate revision identified by '1b5f0d9ad7c1'`. Investigation revealed the local environment uses MLflow 3.8.1 while the Docker container was using MLflow latest (3.10+). Both access the same `mlflow.db` SQLite file via volume mount. The corruption occurred after installing SQLAlchemy separately, potentially triggering an Alembic migration. GitHub issue #12627 documents similar failures from interrupted Alembic migrations.
**Decision**: Pin MLflow Docker image to version 3.8.1 in `docker/Dockerfile.mlflow` to match the local installation version. Establish practice of keeping MLflow versions synchronized across environments that share database files.
**Rationale**: While the exact cause of corruption is not definitively proven, version mismatches between MLflow installations accessing the same database are a known risk factor for Alembic migration failures. Synchronizing versions eliminates this as a potential cause and ensures consistent schema expectations. May also help prevent future corruption from dependency updates.
**Tradeoffs**: Requires manual coordination when upgrading MLflow versions. Cannot use `latest` tag for automatic updates. However, this is acceptable given the fragility of SQLite shared across environments. Not guaranteed to prevent all corruption scenarios, but reduces risk. Alternative: separate databases per environment (rejected: loses unified experiment tracking).
**Tags**: [mlflow, docker, database, schema-migration, version-management, preventive]

---

**ID**: ADR-045
**Date**: 2026-03-25
**Context**: Integration tests for ingestion service use a class-scoped ChromaDB testcontainer shared across all tests. A `chromadb_client` fixture existed with cleanup logic (deletes collection after test), but it was only used by one of four tests. This caused data to accumulate across tests, leading to assertion failures when tests expected specific document counts.
**Decision**: Make the `chromadb_client` fixture autouse by adding `autouse=True` parameter. This ensures cleanup runs automatically after every test in the class, even if the test doesn't declare the fixture as a parameter.
**Rationale**: Leverages existing cleanup logic without code duplication. Collection deletion is fast (<100ms) compared to container startup (2-5 seconds), so maintaining the class-scoped container with per-test cleanup provides both isolation and performance. Tests that need direct ChromaDB access (like `test_health_check_with_documents`) can still declare the fixture parameter to get the client object; tests that only use the API don't need to change.
**Tradeoffs**: The cleanup fixture now runs for all tests, even those that don't write data (e.g., `test_health_check_with_no_documents`). The try/except in the cleanup handler makes this safe. Slight coupling—all tests implicitly depend on cleanup running—but this is preferable to data leakage between tests. Alternative considered: change fixture scope to function (new container per test) but rejected due to significant performance overhead.
**Tags**: [testing, fixtures, integration-tests, ChromaDB, cleanup, isolation]

---

**ID**: ADR-046
**Date**: 2026-03-29
**Context**: Integration tests for ingestion service contained repeated `status_callback` function definitions (6+ instances) that differed only in the response data (pending/completed/error documents) and terminal status value. Each callback followed the same pattern: parse request body, return 201 with pending response if status is PENDING, return 204 if status matches terminal status.
**Decision**: Extract a factory function `make_status_callback(pending_response, terminal_status)` that generates callback functions with the pattern baked in. Replace all inline callback definitions with calls to the factory.
**Rationale**: Eliminates ~100 lines of duplicated code. Adding new status scenarios requires passing different arguments, not rewriting the callback logic. Makes the pattern explicit: callbacks differ by data, not by logic. Reduces maintenance burden—bugs in the pattern only need fixing once.
**Tradeoffs**: Introduces slight abstraction—readers must understand the factory pattern. However, the factory is straightforward and well-documented. The abstraction pays for itself immediately given the number of call sites. Alternative considered: leave duplication for explicitness—rejected as it would make future changes error-prone (need to update 6+ locations consistently).
**Tags**: [testing, code-quality, refactoring, integration-tests, DRY]

---

**ID**: ADR-047
**Date**: 2026-03-29
**Context**: Codebase lacked consistent docstring coverage and no docstring linting was enforced, making it harder to understand module/class/function contracts at a glance.
**Decision**: Add PEP 257 docstrings to all public modules, classes, and functions in the `src/` tree. Configure `flake8-docstrings` as a pre-commit dependency and add a `.flake8` file with `docstring-convention = pep257`, ignoring `D107` (init methods), `D203`/`D213` (conflicting style rules), and exempting `tests/` and `tools/` from docstring checks.
**Rationale**: PEP 257 is the Python standard; ignoring `D107` keeps constructors clean since the class docstring already describes the object. Exempting test files avoids noise without sacrificing coverage of the production API surface.
**Tradeoffs**: Only one-line docstrings are used for brevity; multi-line parameter descriptions can be added later if the team moves to a richer convention (Google/NumPy). Docstring coverage for test files is not enforced.
**Tags**: [documentation, tooling, pre-commit, flake8]

---

**ID**: ADR-048
**Date**: 2026-03-30
**Context**: E2E smoke tests need to validate the core RAG flow (ingest → query → response) with deterministic results. Three options considered: (1) clear DBs as precondition for known state, (2) test against existing state (production-safe), or (3) use namespaced test data (e.g., test-specific ChromaDB collections). Need to balance determinism vs. production-safety for the Iteration II smoke test.
**Decision**: E2E smoke tests clear ChromaDB and DMS database state as a precondition (fixture) before running. Tests assume a clean environment and cannot run in production environments. This is explicitly a **local development smoke test** for validating the docker-compose setup, not a production monitoring test.
**Rationale**: Aligns with Iteration II Definition of Done: "A new developer can run `make up` and `make test-smoke`" - this targets local lab validation, not production. Clearing state ensures deterministic, repeatable tests that are simple to reason about and debug. Smoke tests should be maximally simple - complexity defeats the purpose. The docker-compose stack already provides isolation. Production smoke tests (if needed) would be a separate test suite with different characteristics (non-destructive, eventual consistency, different assertions).
**Tradeoffs**: Tests are destructive and cannot run against production. Require clean environment (`make down && make up` before running). Alternative approaches (namespaced data, production-safe assertions) add significant complexity for no immediate benefit given the local lab scope. If production smoke tests are needed in future iterations (3-4), they'll be a separate concern with separate implementation.
**Tags**: [testing, e2e, smoke-tests, test-strategy, local-development]

---

**ID**: ADR-049
**Date**: 2026-04-02
**Context**: Need to implement E2E smoke tests for the RAG chatbot to validate the full flow (ingestion → document management → inference → UI). Three main options for E2E UI testing: (1) Selenium (widely used, established), (2) Playwright (modern, faster, better API), (3) Streamlit's built-in testing utilities (limited to component testing, not full browser automation).
**Decision**: Use Playwright for E2E browser automation testing of the Streamlit frontend. Install via `requirements-dev.txt` with `playwright` and `pytest-playwright` packages.
**Rationale**: Playwright provides modern async-first architecture with better performance than Selenium. Built-in auto-wait reduces flakiness. Native support for multiple browsers and headless mode. Excellent Python integration via `pytest-playwright` plugin. Streamlit's native testing utilities are insufficient for full browser flow testing. Selenium considered legacy compared to Playwright's developer experience.
**Tradeoffs**: Requires `playwright install` step for browser binaries (one-time setup) and `playwright install-deps` for system dependencies (documented in README). Adds ~50MB to dev dependencies. Team must learn Playwright API instead of more familiar Selenium. Acceptable for significantly better DX and test reliability.
**Tags**: [testing, e2e, playwright, tooling, browser-automation]

---

**ID**: ADR-050
**Date**: 2026-04-02
**Context**: Streamlit components don't natively expose stable test selectors (no `data-testid` support). E2E tests need reliable selectors to locate UI elements without coupling to display text (which may change) or fragile CSS selectors.
**Decision**: Inject custom HTML with `data-testid` attributes using Streamlit's `st.markdown(unsafe_allow_html=True)` for critical UI elements that need to be tested. Style the custom HTML to match Streamlit's native component appearance (fonts, colors, spacing).
**Rationale**: `data-testid` attributes are the Playwright best practice for stable selectors—they're independent of styling and content changes. Streamlit doesn't provide this natively, so custom HTML is the workaround. Styling custom HTML to match native components maintains visual consistency. Alternative considered: using text-based selectors like `page.get_by_text("Documents in vector store")` rejected because text may change or be localized.
**Tradeoffs**: Custom HTML replaces native Streamlit components (e.g., `st.metric`), requiring manual CSS to match Streamlit's appearance. CSS may break if Streamlit updates its styling. Maintenance overhead acceptable for critical smoke test stability. Only use for elements that need stable test selectors—prefer native Streamlit components elsewhere.
**Tags**: [testing, e2e, streamlit, UI, playwright, testability]

---

**ID**: ADR-051
**Date**: 2026-04-02
**Context**: E2E tests require all services (ChromaDB, DMS, Ingestion, Inference, Streamlit) to be running and ready before tests execute. Starting services with `docker compose up -d` is fast, but services aren't immediately ready to accept requests—tests fail with connection errors if run too soon.
**Decision**: Implement a polling readiness check in the `make test-e2e` target that waits for the inference service's `/health` endpoint to respond (up to 30 attempts with 2-second intervals). Tests only run after health check succeeds or timeout (60 seconds).
**Rationale**: Simple, effective solution that prevents flaky test failures from race conditions. Inference service is chosen as the readiness gate because it's the last service in the dependency chain (depends on ChromaDB and DMS). If inference is healthy, the full stack is ready. Polling with exponential backoff considered but rejected as over-engineering—fixed 2-second intervals are sufficient for local docker-compose startup.
**Tradeoffs**: Adds 60 seconds max delay to test runs if services are unhealthy (fail-fast with clear error message). Normal case: services ready in 4-10 seconds. Alternative considered: docker-compose healthchecks with `depends_on` conditions—rejected because not all services have health endpoints, and Makefile polling is more flexible and debuggable.
**Tags**: [testing, e2e, docker-compose, CI/CD, makefile, readiness]

---

**ID**: ADR-052
**Date**: 2026-04-02
**Context**: E2E test needs to ingest a document during the test flow. Two approaches: (1) use Playwright to interact with the UI to trigger ingestion (click buttons, fill forms), or (2) use FastAPI TestClient to directly call the ingestion API endpoint.
**Decision**: Use hybrid approach—Playwright for UI interactions (chat, system status page), FastAPI TestClient for ingestion API calls. E2E test combines both tools in the same test function.
**Rationale**: Ingestion via API is faster, more reliable, and easier to debug than multi-step UI form interactions. The test's purpose is validating the core RAG flow (ingest → query → response), not testing the ingestion UI specifically. Using TestClient for setup (ingestion) and Playwright for the user-facing flow (chat interaction) optimizes for test speed and clarity. Playwright alone would require navigating to an ingestion form, uploading files, and waiting for async UI updates—adding complexity without testing value for this smoke test.
**Tradeoffs**: E2E test depends on both Playwright and FastAPI TestClient libraries. Test combines browser automation and programmatic API calls, which might seem inconsistent. However, this pragmatic approach balances thoroughness (testing UI flow) with speed (fast document ingestion). Pure UI-only approach would be slower and more brittle. Acceptable for a smoke test focused on validating system integration, not comprehensive UI testing.
**Tags**: [testing, e2e, test-strategy, playwright, fastapi, pragmatism]

---