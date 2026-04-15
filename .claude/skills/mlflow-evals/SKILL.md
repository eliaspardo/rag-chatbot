# MLflow Evaluation Query Skill

Use this skill when asked about eval results, MLflow runs, evaluation scores, failed questions, or comparing model runs.

## Tool location

```
tools/mlflow_query.py
```

Run via: `.venv/bin/python tools/mlflow_query.py <command> [options]`

## Format modes

| Command | `--format human` (default) | `--format agent` |
|---------|---------------------------|-----------------|
| `list` | Markdown table, no run IDs | Markdown table + **Run ID** column |
| `show` | Record-per-block, word-wrapped at 90 chars, **no truncation** | Flat markdown sections, no wrapping |
| `show_wide` | Wide table, question truncated to 60 chars, extra fields to 120 chars | Flat markdown sections, no truncation |

Use **`--format agent`** when you need the Run ID from `list`, or prefer flat unindented output from `show`. `show` without `--format agent` is also safe — it word-wraps but never drops data.

---

## Commands

### Discover available fields for a run

```bash
.venv/bin/python tools/mlflow_query.py fields <RUN_NAME_OR_ID>
```

Returns all param and metric keys found across child runs, grouped as:
- **Params (pass to --fields)** — values to use in `--fields "..."`
- **Params (shown automatically)** — `question`, `question_id` — no need to request these
- **Metrics (shown automatically)** — metric scores, always included in `show` output

**Run this before `show` when unsure which `--fields` to request.**

---

### List recent evaluation runs

```bash
.venv/bin/python tools/mlflow_query.py list --format agent
```

Returns a markdown table with columns: Run Name, **Run ID**, Date, Status, Model, metric means, Failures.

Use the **Run Name** (e.g. `deepeval-2026-04-08-11-56-56`) as the argument to `show`.

---

### Show child runs for a parent run

```bash
.venv/bin/python tools/mlflow_query.py show <RUN_NAME_OR_ID> --format agent [--status passed|failed] [--fields FIELD1,FIELD2,...]
```

Returns one markdown section per question:
```
## Q-{id} | {status}
- question: ...
- {extra fields}: ...
- {metric_name}: 0.xxx
```

#### Standard fields for most tasks

```bash
--fields "actual output,expected output,failure"
```

#### Extended fields for failure root-cause analysis

```bash
--fields "actual output,expected output,failure,Grounding_GEval reason,Completeness_GEval reason,Reasoning_GEval reason"
```

#### RAG retrieval debugging

```bash
--fields "actual output,expected output,context,failure"
```

---

## Metrics

All metrics are on a 0–1 scale. Threshold for pass: **0.5**.

| Metric | What it measures |
|--------|-----------------|
| `Grounding_GEval` | Is the answer grounded in the retrieved context? |
| `Completeness_GEval` | Does the answer fully address the question? |
| `Reasoning_GEval` | Is the reasoning coherent and correct? |

---

## Common workflows

### 1. Overview of recent runs

```bash
.venv/bin/python tools/mlflow_query.py list --format agent
```

### 2. Discover what fields are available

```bash
.venv/bin/python tools/mlflow_query.py fields deepeval-2026-04-08-11-56-56
```

### 3. Drill into a specific run (all questions)

```bash
.venv/bin/python tools/mlflow_query.py show deepeval-2026-04-08-11-56-56 \
  --format agent \
  --fields "actual output,expected output,failure"
```

### 3. Investigate only failed questions

```bash
.venv/bin/python tools/mlflow_query.py show deepeval-2026-04-08-11-56-56 \
  --format agent --status failed \
  --fields "actual output,expected output,failure,Grounding_GEval reason,Completeness_GEval reason,Reasoning_GEval reason"
```

### 4. Compare two runs

Run `list` to get run names, then `show` each with the same `--fields`, and compare the per-question scores.
