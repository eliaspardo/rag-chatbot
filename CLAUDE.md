## Learning Context
Currently on Iteration II (Weeks 8–13) of my 2026 upskilling plan.
Full plan: `/docs/2026_Learning_System.md`

## Current Focus
Building a RAG chatbot as a five-service microservices project (Ingestion, Document Management, Inference, Streamlit frontend, ChromaDB).
Primary goal: stable local lab with unit + contract + E2E smoke tests.

## Stack
FastAPI, SQLAlchemy ORM, Pydantic, pact-python (v2+, Rust FFI), pytest, Docker Compose

## Conventions
- TDD at two layers: Pact contract tests for service boundaries, unit tests for internal logic
- Contract tests define what to build — provider verification drives implementation
- Pact Broker running locally in Docker Compose
- SQLite in-memory for unit tests, Postgres for production
- Correctness over performance

## Working agreements

- Always run `pytest` after modifying Python files using the project's virtual environment at `.venv/bin/pytest` (or `.venv/Scripts/pytest` on Windows).
- If tests fail, do not proceed with further changes until the failure is resolved or explicitly acknowledged by the user.


## Architectural Decision Logging

After any exchange where a design or architectural decision is made — including 
technology choices, structural changes, tradeoffs accepted, or patterns adopted — 
append an entry to `docs/ADR_LOG.md` using this format:

---
**ID**: ADR-XXX
**Date**: YYYY-MM-DD  
**Context**: What problem or situation prompted this decision  
**Decision**: What was decided  
**Rationale**: Why this option over alternatives  
**Tradeoffs**: What was accepted or deferred  
**Tags**: [e.g. architecture, testing, infra, API, containerization]  
---

Do this automatically without being asked. Keep entries concise but precise.

## Engineering Log

After sessions involving implementation work, debugging, or optimizations that aren't
architectural decisions but are useful for case studies, append entries to
`docs/ENGINEERING_LOG.md`. Keep it informal and scannable.

Format:
```markdown
## YYYY-MM-DD

### Short title of work done
- What was the problem/goal
- What was the root cause (if debugging)
- What was the fix/solution
- Result/impact (if measurable)
```

Good candidates for logging:
- Bug fixes with interesting root causes
- Performance/build optimizations
- Infrastructure improvements
- Debugging sessions (especially tricky ones)
- Refactoring efforts

Not needed for: trivial changes, typo fixes, routine test additions.

---

## Decision Summaries

When I request a **weekly summary**, run the `.claude/skills/decision-summary/SKILL.md` skill.
When I request an **iteration summary**, run the `.claude/skills/iteration-report/SKILL.md` skill.

Trigger phrases:
- "generate weekly summary" / "weekly summary" → weekly skill
- "generate iteration summary" / "iteration report" → iteration skill

Context to inject into both skills:
- Current iteration: Iteration II (Weeks 8–13, 2026)
- Project: RAG chatbot — 5 services (Ingestion, Document Management, Inference, Streamlit, ChromaDB)
- Full learning plan: `/docs/2026_Learning_System.md` (consult for iteration boundaries and goals)
- Source material: ADR log entries since last summary

## Plan Files
- Store plans in `docs/plans/`
- Name plan files using format: `YYYY-MM-DD_<feature-slug>.md`
  Example: `2026-03-24_pact-contract-ingestion-service.md`
- Never use auto-generated random names