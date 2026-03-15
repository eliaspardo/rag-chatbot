# Master Prompt — 2026 Learning & Execution System

*Use this document as a standing operating prompt to plan, execute, review, and adjust work throughout 2026. It is designed to optimize for employability, momentum, and real-life constraints.*

## Context & Constraints

**Timeframe:** January–September 2026

**Primary Objective:** Employability as a Test Infrastructure / Platform Quality Architect

### Life Constraints

- Two kids → regular sick days (~3 days every 2 months)
- Summer period (late June–early September) will be chaotic
- Expect 20–30% capacity loss during these periods

### Operating Principles

- Momentum over perfection
- Artifacts over theory
- No catching up after bad weeks; resume at iteration boundaries

## North-Star Goal

**By the end of 2026, I am employable as a Test Infrastructure / Platform Quality Architect, responsible for designing CI/CD-integrated quality systems — including AI-assisted quality capabilities — for API-driven and service-oriented backends.**

### Supporting Goal

Develop sufficient hands-on understanding of APIs, AI agents, databases, and observability to design realistic and effective automated test strategies — not to master these domains independently.

## Learning System

**6-week iterations with clear focus:**

- Define one primary and one secondary goal.
- Primary Goal cannot change mid-iteration. Only how you approach it can change.
- Define clear outcomes for primary goals: one tangible artifact and an optional validation signal.
- Primary Goals: 10 hours a week, with a minimum of 6 hours guaranteed unless going to survival mode (low-energy tasks only).
- An iteration is successful if the primary artifact exists — even if incomplete.
- Secondary Goals have optional night hours or free time. They should be doable in less than 2 hours a week.

**Iteration boundaries:**

- At the beginning of each iteration the goal and deliverables are refined (60 mins max).
- At the end of each iteration, a living **"Positioning Doc"** (1 page) will be updated with what kind of problems I solve.

**Weekly cadence:**

- **Monday** — what to focus on (max 30 mins, unless beginning of iteration)
- **Friday** — retrospective (max 30 mins):
  - What actually moved the primary goal forward?
  - What was planned but didn't happen, and why?
  - One concrete adjustment for next week if needed.
  - Why did I choose this tool/pattern over the alternative, and what is its biggest weakness?

We've identified low-energy tasks for when the system breaks (kids fall sick).

There are recurring non-negotiables, which will take some time but will be handled outside this system.

## Planned Iterations

### Iteration 1 — Weeks 2–7 (Feb 15): System AI Evals

#### Primary Goal

Build an automated evaluation pipeline for your RAG chatbot to measure quality and reliability before the Automation Guild Q&A session (Feb 9).

**The "Architect" Twist:** Wrap your chatbot in a FastAPI layer. Even if you test in the same repo, this mimics how a real frontend or service would call it. It allows you to simulate network latency, timeouts, and concurrent requests.

#### Artifacts

- **The RAG API:** A containerized FastAPI service for your chatbot.
- **The Evaluation Engine (RAGAS):** A pytest suite that runs against the API to measure Faithfulness, Relevancy, and Context Recall. → Finally used DeepEval as an evaluation tool.
- **The "Golden Dataset":** A versioned JSON file containing 20 vetted "Ground Truth" question/answer pairs.
- **Guild Case Study:** A 1-page report on your RAG performance (e.g., "Hallucination Rate vs. Temperature") to use for Q&A at the Guild.

#### Underlying Objectives

- Portfolio — AI Evaluation framework
- Automation Guild 2026 — Improve AI automation results. End date coincides with Q&A session so any learning will help.

#### Definition of Done

You can run `make test-ai` and see a table of RAGAS scores for your 20 test cases.

---

### Iteration 2 — Weeks 8–13 (Apr 5): Microservices Local Lab + Test Foundation

#### Primary Goal

Get your RAG chatbot microservices (ingestion, inference, document management) running as a stable local lab with a real test suite started: unit + contract + 1 smoke E2E.

#### Context

Your RAG chatbot is already split into three services. This iteration treats them as the microservices architecture you'll build quality infrastructure around for the rest of the plan. No need to fork someone else's project — you designed these service boundaries and can articulate why they exist.

**Guardrail:** If you can't get the three-service compose setup running in one focused session, simplify to two services first (inference + document management) and add ingestion later.

#### Artifacts

- Docker Compose config for all three services + one-command `make up`
- Unit tests added or improved in one service (e.g., document management)
- Contract tests for one consumer/provider pair using Pact (e.g., inference → document management)
- One E2E smoke test proving a core flow (e.g., ingest document → query → get relevant answer). Deterministic with seeded data.
- README: how to run + how to test

#### Underlying Objectives

- Portfolio — Real microservices project (not a tutorial fork)
- Portfolio — Stepping stone towards CI/CD
- Skills — Service boundaries, contracts, local deployment

#### Definition of Done

A new developer can run `make up` and `make test-smoke` from the README and see unit + contract + smoke pass locally.

#### Secondary Goal

Document the service boundaries and contracts as a lightweight ADR — this becomes the first document the future agent can reference.

---

### Iteration 3 — Weeks 14–19 (May 17): CI Pipeline + Quality Gates

#### Primary Goal

Turn the tests into a pipeline that enforces rules and produces signals (reports, gates, artifacts).

#### Artifacts

A CI pipeline that runs on every PR with:

- Build + unit tests (fast)
- Contract tests (consumer + provider verification) as a merge gate
- Spin-up integration environment (compose in CI or testcontainers)
- E2E smoke (short, reliable, timeboxed)
- pytest reports (HTML + structured output via `--json-report`) + clear failure output
- On any failed CI run: logs + reports uploaded, output tells you how to reproduce locally

Plus one short doc:

- *"Test strategy in CI: what runs when, and why."*

**Design consideration for iteration 5:** Use pytest's `--json-report` plugin and structured logging (JSON lines) from the start. These structured formats will be significantly easier to parse when building the agent's preprocessing tools. This isn't extra work — it's choosing the right output format now.

#### Underlying Objectives

- Portfolio — Fully functional CI/CD pipeline with automated tests
- Skills — Pipeline design, quality gates, failure diagnostics

#### Definition of Done

PR checks block merge on contract break, and a failure produces logs/reports + a copy-paste repro command.

#### Secondary Goal

Add minimal failure diagnostics (logs, traces, test metadata). Baseline diagnostics first. Add OTel tracing for one flow as a stretch goal — if no time, drop it.

---

### Iteration 4 — Weeks 20–25 (Jun 28): Environment Architecture for Reliable Testing

#### Primary Goal

Define and implement an environment and infrastructure strategy that maximizes test reliability and parity between local and CI execution, using lightweight IaC where it adds value.

#### Approach

- IaC only where it helps test reproducibility
- Containers as ephemeral environments
- No fake "staging"

#### Artifacts

- Environment profiles (local, CI): same commands, different config
- Lifecycle commands: `make env-up` / `env-reset` / `env-down` / `smoke`
- Pinned dependencies: images/versions locked (no drift)
- Deterministic data: seed script + documented invariants
- Infra/config repo + README (2 pages max) explaining execution contexts, trade-offs, and how these decisions support test strategy
- *"How to reproduce a CI failure locally" playbook*

#### Underlying Objectives

- Portfolio — Environment architecture as a quality concern
- Skills — IaC, reproducibility, environment parity

#### Definition of Done

Same commands work locally and in CI with only config changes. `env-reset` guarantees a known state.

#### Secondary Goal

Database basics as they affect testing: migrations + seed workflow in CI. Rollback via redeploy + forward fix — document the reasons.

---

### Iteration 5 — Weeks 26–31 (Aug 9): Agent Tooling: Preprocessing & Structured Extraction

#### Primary Goal

Build and test a set of standalone preprocessing tools that extract structured, actionable information from CI artifacts. Each tool takes a build identifier as input and returns structured output with metadata. These tools become the agent's capabilities in iteration 6.

#### Why This Replaced Async/Queue Testing

The agent in iteration 6 needs tested tools to orchestrate, and building both tools and agent orchestration in one iteration is too much for 60 hours. This work is also naturally resilient to the chaotic summer period — each tool is self-contained and can be picked up and put down independently.

#### Artifacts

A Python package (or directory of scripts) with a consistent interface. Each tool:

- Accepts a build identifier as input
- Reads the relevant artifact type
- Preprocesses: strips noise, extracts signal, discards boilerplate
- Returns structured output with metadata (build ID, timestamp, service name, source type, git commit SHA, branch, environment profile)

The specific tools (based on artifacts produced in iterations 2–4):

| Tool | What it does | Source artifact |
|------|-------------|-----------------|
| `get_test_results` | Parses pytest output for a build. Returns failed tests with names, assertion messages, markers, and tracebacks (`--tb=short`). | pytest `--json-report` output |
| `get_log_errors` | Extracts ERROR/WARN entries from service logs. Filters by known patterns (stack traces, timeouts, connection errors). | Structured JSON logs from services |
| `get_code_changes` | Returns the git diff and list of changed files for the commit associated with a build. | Git history |
| `get_env_config` | Returns the environment profile, image versions, and seed data config for a given build. | Compose files, env profiles from iteration 4 |
| `get_test_history` | Compares a specific test's results across the last N builds. Surfaces patterns (flaky, recently broken, consistently failing). | Stored pytest results across builds |

Plus:

- pytest tests for each tool: known input → expected structured output
- **Interface contract ADR:** document the common return shape (`build_id`, `source_type`, `timestamp`, `content`, `metadata`) and why you chose it

#### Design Principle

**Build tools for yourself that the agent can later use.** If you wouldn't write the script purely for your own debugging convenience during iterations 2–4, don't build it here. The discipline is: no scope creep disguised as "preparation."

#### Underlying Objectives

- Portfolio — Preprocessing pipeline design (what to keep, what to discard, what metadata matters)
- Portfolio — Direct preparation for AI agent
- Skills — Structured data extraction, interface design, test tooling

#### Definition of Done

Each tool can be called with a build ID and returns structured, tested output. A new tool can be added by following the established pattern.

#### Secondary Goal

- **Job target definition:** Define your ideal role — kind of company, stage, type of work, deal-breakers. This must be done before September in case you need to start applying.
- **Async/queue exploration (optional):** Light reading on event-driven testing patterns. Maybe add one async interaction between services (e.g., ingestion publishes an event when a document is processed). Enough to speak intelligently in interviews, not deep implementation.

---

### Iteration 6 — Weeks 32–37 (Sep 20): AI Agent for Failure Analysis

#### Primary Goal

Orchestrate the tools from iteration 5 into an AI agent that analyzes test failures, reasons across multiple artifact types, and suggests root causes.

#### Why Agent Over RAG

RAG was evaluated and rejected for this use case. Semantic retrieval doesn't work for CI artifact analysis because: (1) the relationship between a query and relevant artifacts is logical (same build ID) not semantic (similar text), (2) failure analysis requires all artifacts from a build, not the top-k similar chunks, and (3) structured artifacts like test results and code diffs lose critical context when chunked. The agent approach uses preprocessing tools as callable functions, with the LLM deciding which tools to call, in what order, and synthesizing the results.

#### Architecture

- **Agent layer:** LLM with tool-calling capabilities. Receives a user query, decides which tools to invoke, chains results, produces analysis.
- **Tool layer:** The 5 preprocessing tools from iteration 5, exposed as callable functions with the consistent interface contract.
- **Context layer (optional):** ADRs, runbooks, and test strategy docs loaded as static context (context stuffing, not RAG). Provides the "why" behind architectural decisions when the agent needs to explain a failure.

#### Artifacts

- Working agent that can analyze a triggered failure scenario end-to-end
- Prompt design: system prompt that guides tool selection and multi-step reasoning
- At least one scripted demo: break a contract → run pipeline → ask agent what happened → agent calls tools → provides diagnosis with references to ADRs/runbooks
- Evaluation framework for the agent (extending DeepEval experience):
  - Does it select the right tools?
  - Does it call them with correct parameters?
  - Does it interpret results correctly?
  - Is the final diagnosis accurate?
- **ADR:** "Why agent over RAG for CI failure analysis" — documenting the architectural decision and trade-offs

#### Underlying Objectives

- Portfolio — AI-assisted quality capabilities (the differentiator in your north star)
- Portfolio — Demonstrates architectural judgment: evaluated RAG, identified why it was wrong, designed agent approach instead
- Skills — Agent orchestration, tool design, multi-step evaluation

#### Definition of Done

The agent can analyze a deliberately introduced failure, call the appropriate tools, and produce a diagnosis that references the correct root cause. The evaluation framework can measure whether the agent's tool selection and diagnosis are correct.

#### Secondary Goal

Write up the full portfolio narrative: how the RAG chatbot, the microservices quality lab, and the AI agent connect into one coherent story. This becomes content for your GitHub Pages portfolio and potential blog posts/talks.

## Recurring Non-Negotiables (Outside the System)

- Continue promoting BrowserStack Meetups
- Continue writing LinkedIn posts (monthly cadence by default)

These are maintenance activities and should not compete with iteration goals.
