#!/usr/bin/env python3
"""
MLflow Evaluation Query Tool

CLI for querying MLflow evaluation results:
  list  — Show recent parent evaluation runs with aggregated metrics
  show  — Drill into child runs for a specific parent run

Usage:
  mlflow_query.py list [-n COUNT] [--tracking-uri URI] [--experiment NAME]
  mlflow_query.py show <RUN_NAME_OR_ID> [--tracking-uri URI] [--experiment NAME]
                       [--status passed|failed] [--fields FIELD1,FIELD2,...]

Defaults are loaded from config/params.env (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _find_project_root() -> Path:
    """Walk up from __file__ until config/params.env is found."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config" / "params.env").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = _find_project_root()

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / "config" / "params.env")
except ImportError:
    pass  # python-dotenv unavailable — rely on environment

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("ERROR: mlflow is not installed. Run: pip install mlflow", file=sys.stderr)
    sys.exit(1)

logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

DEFAULT_TRACKING_URI = "sqlite:///./mlflow.db"
DEFAULT_EXPERIMENT_NAME = "RAG Chatbot"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(
    cli_uri: Optional[str], cli_experiment: Optional[str]
) -> tuple[str, str]:
    """Resolve tracking URI + experiment name (CLI > env > default)."""
    uri = cli_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    experiment = (
        cli_experiment or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    )

    # Resolve relative SQLite paths against project root so the script works
    # from any working directory.
    if uri.startswith("sqlite:///"):
        db_path = uri[len("sqlite:///") :]
        if db_path.startswith("./"):
            db_path = db_path[2:]
        if not os.path.isabs(db_path):
            db_path = str(PROJECT_ROOT / db_path)
        uri = f"sqlite:///{db_path}"

    return uri, experiment


# ---------------------------------------------------------------------------
# MLflow query helpers
# ---------------------------------------------------------------------------


def get_parent_runs(client: "MlflowClient", exp_id: str, max_results: int) -> List:
    """Return parent runs ordered by start_time DESC."""
    return client.search_runs(
        experiment_ids=[exp_id],
        filter_string="tags.run_type = 'parent'",
        order_by=["start_time DESC"],
        max_results=max_results,
    )


def get_child_runs(
    client: "MlflowClient",
    exp_id: str,
    parent_run_id: str,
    status_filter: Optional[str] = None,
) -> List:
    """Return child runs for parent_run_id, optionally filtered by status tag."""
    filter_parts = [f"tags.`mlflow.parentRunId` = '{parent_run_id}'"]
    if status_filter:
        filter_parts.append(f"tags.status = '{status_filter}'")
    return list(
        client.search_runs(
            experiment_ids=[exp_id],
            filter_string=" AND ".join(filter_parts),
            max_results=500,
        )
    )


def find_parent_run(client: "MlflowClient", exp_id: str, name_or_id: str):
    """Search by run_name first (Python-side match), then fall back to run ID."""
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="tags.run_type = 'parent'",
        order_by=["start_time DESC"],
        max_results=1000,
    )
    for run in runs:
        if run.info.run_name == name_or_id:
            return run

    # Fallback: treat argument as a raw run ID
    try:
        return client.get_run(name_or_id)
    except Exception as e:
        print(f"Exception getting parent run {e}")
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def natural_sort_key(run) -> int:
    """Extract the integer N from 'question-N' run names for correct ordering."""
    name = run.info.run_name or ""
    match = re.search(r"question-(\d+)", name)
    return int(match.group(1)) if match else 0


def _discover_parent_metric_cols(runs) -> List[str]:
    """Collect metric base names from parent run _mean metrics."""
    seen: List[str] = []
    for run in runs:
        for key in run.data.metrics:
            if key.endswith("_mean"):
                base = key[: -len("_mean")]
                if base not in seen:
                    seen.append(base)
    return sorted(seen)


def _discover_child_metric_cols(runs) -> List[str]:
    """Collect raw metric names from child runs."""
    seen: List[str] = []
    for run in runs:
        for key in run.data.metrics:
            if key not in seen:
                seen.append(key)
    return sorted(seen)


def _truncate(value: str, width: int) -> str:
    if len(value) > width:
        return value[: width - 3] + "..."
    return value


def _fmt_row(row: List[str], col_widths: List[int]) -> str:
    return (
        "| " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)) + " |"
    )


def format_parent_runs_table(runs) -> str:
    """Markdown table: Run Name, Date, Status, Model, metric means, Failures."""
    if not runs:
        return "_No parent runs found._"

    metric_cols = _discover_parent_metric_cols(runs)
    headers = (
        ["Run Name", "Date", "Status", "Model"]
        + [f"{m}_mean" for m in metric_cols]
        + ["Failures"]
    )

    rows: List[List[str]] = []
    for run in runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics

        run_name = run.info.run_name or run.info.run_id
        try:
            date = datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date = str(run.info.start_time or "")

        status = tags.get("status", "")
        model = params.get("app_model_name", "") or params.get("eval_model_name", "")
        failures = params.get("failure_count", "0")
        metric_values = [
            f"{metrics[f'{m}_mean']:.3f}" if f"{m}_mean" in metrics else ""
            for m in metric_cols
        ]
        rows.append([run_name, date, status, model] + metric_values + [failures])

    col_widths = [
        max(len(headers[i]), max(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join(
        [_fmt_row(headers, col_widths), sep] + [_fmt_row(r, col_widths) for r in rows]
    )


def format_child_runs_table(
    runs, parent_name: str, extra_fields: Optional[List[str]] = None
) -> str:
    """Wide table: Q-ID, Status, metric scores, Question (60 chars), extra fields (120 chars)."""
    extra_fields = extra_fields or []
    header = f"### Child runs for: {parent_name}"

    if not runs:
        return f"{header}\n\n_No child runs found._"

    sorted_runs = sorted(runs, key=natural_sort_key)
    metric_names = _discover_child_metric_cols(sorted_runs)

    headers = ["Q-ID", "Status"] + metric_names + ["Question"] + extra_fields
    rows: List[List[str]] = []

    for run in sorted_runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics

        q_id = str(params.get("question_id", run.info.run_name or run.info.run_id))
        status = tags.get("status", "")
        scores = [f"{metrics[m]:.3f}" if m in metrics else "" for m in metric_names]
        question = _truncate(str(params.get("question", "")), 60)
        extra_values = [_truncate(str(params.get(f, "")), 120) for f in extra_fields]

        rows.append([q_id, status] + scores + [question] + extra_values)

    col_widths = [
        max(len(headers[i]), max(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [header, "", _fmt_row(headers, col_widths), sep] + [
        _fmt_row(r, col_widths) for r in rows
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    """Word-wrap text to width, indenting continuation lines."""
    import textwrap

    lines = textwrap.wrap(text, width=width)
    if not lines:
        return indent + "—"
    return ("\n" + indent).join(lines)


def format_child_runs_records(
    runs, parent_name: str, extra_fields: Optional[List[str]] = None
) -> str:
    """Record-per-block layout: metrics on one line, text fields each on their own wrapped line."""
    extra_fields = extra_fields or []
    lines = [f"Child runs for: {parent_name}", ""]

    if not runs:
        lines.append("No child runs found.")
        return "\n".join(lines)

    sorted_runs = sorted(runs, key=natural_sort_key)
    metric_names = _discover_child_metric_cols(sorted_runs)
    divider = "─" * 80

    for run in sorted_runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics

        q_id = str(params.get("question_id", run.info.run_name or run.info.run_id))
        status = tags.get("status", "")
        scores = "  ".join(
            f"{m}: {metrics[m]:.3f}" for m in metric_names if m in metrics
        )

        status_label = f"[{status}]" if status else ""
        lines.append(f"{q_id}  {status_label}  {scores}".rstrip())

        question = str(params.get("question", ""))
        if question:
            lines.append(f"  Question:  {_wrap(question, 90, '             ')}")

        for field in extra_fields:
            value = str(params.get(field, ""))
            label = f"  {field.capitalize()}:"
            indent = " " * (len(label) + 2)
            lines.append(f"{label}  {_wrap(value, 90, indent)}")

        lines.append(divider)

    return "\n".join(lines)


def format_child_runs_agent(
    runs, parent_name: str, extra_fields: Optional[List[str]] = None
) -> str:
    """Agent-friendly format: markdown sections per question, no truncation, no decoration."""
    extra_fields = extra_fields or []
    lines = [f"# Child runs: {parent_name}", ""]

    if not runs:
        lines.append("No child runs found.")
        return "\n".join(lines)

    sorted_runs = sorted(runs, key=natural_sort_key)
    metric_names = _discover_child_metric_cols(sorted_runs)

    for run in sorted_runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics

        q_id = str(params.get("question_id", run.info.run_name or run.info.run_id))
        status = tags.get("status", "")

        lines.append(f"## Q-{q_id} | {status}")
        question = str(params.get("question", ""))
        if question:
            lines.append(f"- question: {question}")
        for field in extra_fields:
            value = str(params.get(field, ""))
            lines.append(f"- {field}: {value}")
        for m in metric_names:
            if m in metrics:
                lines.append(f"- {m}: {metrics[m]:.3f}")
        lines.append("")

    return "\n".join(lines)


def format_parent_runs_agent(runs) -> str:
    """Agent-friendly parent runs table: same as human format but includes run_id column."""
    if not runs:
        return "_No parent runs found._"

    metric_cols = _discover_parent_metric_cols(runs)
    headers = (
        ["Run Name", "Run ID", "Date", "Status", "Model"]
        + [f"{m}_mean" for m in metric_cols]
        + ["Failures"]
    )

    rows: List[List[str]] = []
    for run in runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics

        run_name = run.info.run_name or run.info.run_id
        run_id = run.info.run_id
        try:
            date = datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date = str(run.info.start_time or "")

        status = tags.get("status", "")
        model = params.get("app_model_name", "") or params.get("eval_model_name", "")
        failures = params.get("failure_count", "0")
        metric_values = [
            f"{metrics[f'{m}_mean']:.3f}" if f"{m}_mean" in metrics else ""
            for m in metric_cols
        ]
        rows.append(
            [run_name, run_id, date, status, model] + metric_values + [failures]
        )

    col_widths = [
        max(len(headers[i]), max(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join(
        [_fmt_row(headers, col_widths), sep] + [_fmt_row(r, col_widths) for r in rows]
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_list(args) -> None:
    uri, experiment = load_config(args.tracking_uri, args.experiment)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        print(f"ERROR: Experiment '{experiment}' not found at {uri}", file=sys.stderr)
        sys.exit(1)

    runs = get_parent_runs(client, exp.experiment_id, args.n)
    if args.output_format == "agent":
        print(format_parent_runs_agent(runs))
    else:
        print(format_parent_runs_table(runs))


def _resolve_show_args(args):
    """Shared setup for show / show_wide."""
    uri, experiment = load_config(args.tracking_uri, args.experiment)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        print(f"ERROR: Experiment '{experiment}' not found at {uri}", file=sys.stderr)
        sys.exit(1)

    parent = find_parent_run(client, exp.experiment_id, args.run)
    if parent is None:
        print(
            f"ERROR: Run '{args.run}' not found in experiment '{experiment}'",
            file=sys.stderr,
        )
        sys.exit(1)

    extra_fields: List[str] = []
    if args.fields:
        extra_fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    children = get_child_runs(
        client, exp.experiment_id, parent.info.run_id, args.status
    )
    parent_name = parent.info.run_name or parent.info.run_id
    return children, parent_name, extra_fields


def cmd_show(args) -> None:
    children, parent_name, extra_fields = _resolve_show_args(args)
    if args.output_format == "agent":
        print(format_child_runs_agent(children, parent_name, extra_fields))
    else:
        print(format_child_runs_records(children, parent_name, extra_fields))


def cmd_show_wide(args) -> None:
    children, parent_name, extra_fields = _resolve_show_args(args)
    if args.output_format == "agent":
        print(format_child_runs_agent(children, parent_name, extra_fields))
    else:
        print(format_child_runs_table(children, parent_name, extra_fields))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query MLflow evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list -n 5
  %(prog)s show deepeval-2026-04-08-11-56-56
  %(prog)s show deepeval-2026-04-08-11-56-56 --status failed
  %(prog)s show deepeval-2026-04-08-11-56-56 --fields "actual output,expected output,failure"
  %(prog)s show_wide deepeval-2026-04-08-11-56-56 --fields "actual output,expected output,failure"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_format_arg(p):
        p.add_argument(
            "--format",
            choices=["human", "agent"],
            default="human",
            dest="output_format",
            help="Output format: human (default, pretty) or agent (no truncation, minimal decoration)",
        )

    list_p = subparsers.add_parser("list", help="List recent parent evaluation runs")
    list_p.add_argument(
        "-n",
        type=int,
        default=10,
        metavar="COUNT",
        help="Number of runs to show (default: 10)",
    )
    list_p.add_argument("--tracking-uri", default=None, dest="tracking_uri")
    list_p.add_argument("--experiment", default=None)
    _add_format_arg(list_p)

    def _add_show_args(p):
        p.add_argument("run", metavar="RUN_NAME_OR_ID")
        p.add_argument("--tracking-uri", default=None, dest="tracking_uri")
        p.add_argument("--experiment", default=None)
        p.add_argument(
            "--status",
            choices=["passed", "failed"],
            default=None,
            help="Filter child runs by status tag",
        )
        p.add_argument(
            "--fields",
            default=None,
            help=(
                "Comma-separated extra MLflow param columns to include, e.g. "
                '"actual output,expected output,failure"'
            ),
        )
        _add_format_arg(p)

    _add_show_args(
        subparsers.add_parser(
            "show",
            help="Show child runs — readable record-per-block layout",
        )
    )
    _add_show_args(
        subparsers.add_parser(
            "show_wide",
            help="Show child runs — wide table layout",
        )
    )

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    else:
        cmd_show_wide(args)


if __name__ == "__main__":
    main()
