"""Unit tests for tools/mlflow_query.py — pure functions only, no MLflow I/O."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from tools.mlflow_query import (
    _truncate,
    format_child_runs_agent,
    format_parent_runs_agent,
    format_parent_runs_table,
    load_config,
    natural_sort_key,
    DEFAULT_EXPERIMENT_NAME,
    PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Helpers to build fake MLflow Run objects
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "abc123",
    run_name: str = "question-1",
    start_time: int = 1_700_000_000_000,
    params: dict | None = None,
    tags: dict | None = None,
    metrics: dict | None = None,
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = run_name
    run.info.start_time = start_time
    run.data.params = params or {}
    run.data.tags = tags or {}
    run.data.metrics = metrics or {}
    return run


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_cli_args_take_precedence(self):
        uri, exp = load_config("sqlite:///my.db", "MyExp")
        # load_config resolves relative sqlite paths to absolute
        assert uri.startswith("sqlite:///")
        assert uri.endswith("/my.db")
        assert exp == "MyExp"

    def test_defaults_when_no_args_no_env(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        uri, exp = load_config(None, None)
        assert exp == DEFAULT_EXPERIMENT_NAME
        # URI should be absolute after resolution
        assert uri.startswith("sqlite:///")
        assert not uri.startswith("sqlite:///./")

    def test_env_vars_used_when_no_cli(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "EnvExp")
        uri, exp = load_config(None, None)
        # relative sqlite paths are resolved to absolute
        assert uri.startswith("sqlite:///")
        assert uri.endswith("/env.db")
        assert exp == "EnvExp"

    def test_relative_sqlite_path_is_made_absolute(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        uri, _ = load_config(None, None)
        db_path = uri[len("sqlite:///") :]
        assert os.path.isabs(db_path), f"Expected absolute path, got: {db_path}"

    def test_dot_slash_prefix_stripped_and_made_absolute(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        uri, _ = load_config("sqlite:///./custom.db", None)
        assert uri == f"sqlite:///{PROJECT_ROOT}/custom.db"

    def test_already_absolute_sqlite_path_unchanged(self):
        uri, _ = load_config("sqlite:////absolute/path/my.db", None)
        assert uri == "sqlite:////absolute/path/my.db"

    def test_non_sqlite_uri_returned_as_is(self):
        uri, _ = load_config("http://localhost:5000", None)
        assert uri == "http://localhost:5000"


# ---------------------------------------------------------------------------
# natural_sort_key
# ---------------------------------------------------------------------------


class TestNaturalSortKey:
    def test_extracts_integer_from_question_name(self):
        run = _make_run(run_name="question-7")
        assert natural_sort_key(run) == 7

    def test_returns_zero_for_non_matching_name(self):
        run = _make_run(run_name="some-other-run")
        assert natural_sort_key(run) == 0

    def test_returns_zero_for_empty_name(self):
        run = _make_run(run_name="")
        assert natural_sort_key(run) == 0

    def test_returns_zero_when_run_name_is_none(self):
        run = _make_run(run_name="question-3")
        run.info.run_name = None  # override after creation
        assert natural_sort_key(run) == 0

    def test_multi_digit_number_parsed_correctly(self):
        run = _make_run(run_name="question-42")
        assert natural_sort_key(run) == 42

    def test_sorting_by_key_orders_correctly(self):
        runs = [_make_run(run_name=f"question-{n}") for n in [3, 1, 2]]
        sorted_runs = sorted(runs, key=natural_sort_key)
        names = [r.info.run_name for r in sorted_runs]
        assert names == ["question-1", "question-2", "question-3"]


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_string_exactly_at_width_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_string_one_over_width_truncated(self):
        result = _truncate("123456", 5)
        assert result == "12..."
        assert len(result) == 5

    def test_long_string_truncated_with_ellipsis(self):
        long = "a" * 100
        result = _truncate(long, 20)
        assert result.endswith("...")
        assert len(result) == 20

    def test_width_of_three_returns_just_ellipsis(self):
        result = _truncate("abcdef", 3)
        assert result == "..."

    def test_empty_string_unchanged(self):
        assert _truncate("", 10) == ""


# ---------------------------------------------------------------------------
# format_parent_runs_table
# ---------------------------------------------------------------------------


class TestFormatParentRunsTable:
    def test_empty_runs_returns_placeholder(self):
        result = format_parent_runs_table([])
        assert result == "_No parent runs found._"

    def test_single_run_produces_markdown_table(self):
        run = _make_run(
            run_name="eval-run-1",
            params={"app_model_name": "gpt-4", "failure_count": "2"},
            tags={"status": "passed"},
            metrics={"faithfulness_mean": 0.9, "relevance_mean": 0.8},
        )
        result = format_parent_runs_table([run])
        assert "eval-run-1" in result
        assert "gpt-4" in result
        assert "passed" in result
        assert "0.900" in result
        assert "0.800" in result
        assert "|" in result  # it's a Markdown table

    def test_header_row_present(self):
        run = _make_run(metrics={"score_mean": 1.0})
        result = format_parent_runs_table([run])
        assert "Run Name" in result
        assert "Date" in result
        assert "Status" in result
        assert "Model" in result
        assert "Failures" in result

    def test_separator_row_present(self):
        run = _make_run(metrics={"score_mean": 0.5})
        result = format_parent_runs_table([run])
        lines = result.splitlines()
        # Third line (index 1) should be the separator
        assert "---" in lines[1]

    def test_multiple_runs_all_appear(self):
        runs = [_make_run(run_id=f"id{i}", run_name=f"run-{i}") for i in range(3)]
        result = format_parent_runs_table(runs)
        for i in range(3):
            assert f"run-{i}" in result

    def test_metric_column_only_from_mean_metrics(self):
        run = _make_run(metrics={"faithfulness_mean": 0.7, "some_raw": 0.3})
        result = format_parent_runs_table([run])
        assert "faithfulness_mean" in result
        # raw metric without _mean suffix should not appear as a column header
        assert "some_raw" not in result

    def test_missing_metric_shows_blank(self):
        run1 = _make_run(run_name="run-a", metrics={"alpha_mean": 0.5})
        run2 = _make_run(run_name="run-b", metrics={"beta_mean": 0.6})
        result = format_parent_runs_table([run1, run2])
        # Both columns should exist; cells without data should be blank (empty string)
        assert "alpha_mean" in result
        assert "beta_mean" in result

    def test_fallback_to_eval_model_name_param(self):
        run = _make_run(params={"eval_model_name": "claude-3"})
        result = format_parent_runs_table([run])
        assert "claude-3" in result


# ---------------------------------------------------------------------------
# format_child_runs_agent
# ---------------------------------------------------------------------------


class TestFormatChildRunsAgent:
    def test_empty_runs_returns_header_and_message(self):
        result = format_child_runs_agent([], "my-run")
        assert "# Child runs: my-run" in result
        assert "No child runs found." in result

    def test_header_format(self):
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "What?"},
            tags={"status": "passed"},
        )
        result = format_child_runs_agent([run], "my-run")
        assert result.startswith("# Child runs: my-run")

    def test_question_section_heading(self):
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "What?"},
            tags={"status": "failed"},
        )
        result = format_child_runs_agent([run], "my-run")
        assert "## Q-1 | failed" in result

    def test_question_text_not_truncated(self):
        long_question = "A" * 200
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": long_question},
            tags={"status": "passed"},
        )
        result = format_child_runs_agent([run], "my-run")
        assert long_question in result
        assert "..." not in result

    def test_extra_fields_not_truncated(self):
        long_value = "B" * 300
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "Q?", "actual output": long_value},
            tags={"status": "passed"},
        )
        result = format_child_runs_agent(
            [run], "my-run", extra_fields=["actual output"]
        )
        assert long_value in result
        assert "..." not in result

    def test_metrics_formatted_as_list_items(self):
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "Q?"},
            tags={"status": "passed"},
            metrics={"faithfulness": 0.85, "relevance": 0.72},
        )
        result = format_child_runs_agent([run], "my-run")
        assert "- faithfulness: 0.850" in result
        assert "- relevance: 0.720" in result

    def test_extra_fields_appear_as_list_items(self):
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "Q?", "failure": "low score"},
            tags={"status": "failed"},
        )
        result = format_child_runs_agent([run], "my-run", extra_fields=["failure"])
        assert "- failure: low score" in result

    def test_runs_sorted_by_natural_key(self):
        runs = [
            _make_run(
                run_name=f"question-{n}",
                params={"question_id": str(n), "question": "Q?"},
                tags={"status": "passed"},
            )
            for n in [3, 1, 2]
        ]
        result = format_child_runs_agent(runs, "my-run")
        pos1 = result.index("## Q-1")
        pos2 = result.index("## Q-2")
        pos3 = result.index("## Q-3")
        assert pos1 < pos2 < pos3

    def test_no_decorative_dividers(self):
        run = _make_run(
            run_name="question-1",
            params={"question_id": "1", "question": "Q?"},
            tags={"status": "passed"},
        )
        result = format_child_runs_agent([run], "my-run")
        assert "─" not in result
        assert "---" not in result


# ---------------------------------------------------------------------------
# format_parent_runs_agent
# ---------------------------------------------------------------------------


class TestFormatParentRunsAgent:
    def test_empty_runs_returns_placeholder(self):
        result = format_parent_runs_agent([])
        assert result == "_No parent runs found._"

    def test_includes_run_id_column(self):
        run = _make_run(run_id="abc123", run_name="eval-run-1")
        result = format_parent_runs_agent([run])
        assert "Run ID" in result
        assert "abc123" in result

    def test_human_format_does_not_include_run_id_column(self):
        run = _make_run(run_id="abc123", run_name="eval-run-1")
        result = format_parent_runs_table([run])
        assert "Run ID" not in result

    def test_standard_columns_present(self):
        run = _make_run(
            run_name="eval-run-1",
            params={"app_model_name": "gpt-4", "failure_count": "0"},
            tags={"status": "passed"},
            metrics={"faithfulness_mean": 0.9},
        )
        result = format_parent_runs_agent([run])
        assert "Run Name" in result
        assert "Run ID" in result
        assert "Status" in result
        assert "Model" in result
        assert "Failures" in result
        assert "faithfulness_mean" in result
