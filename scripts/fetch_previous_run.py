#!/usr/bin/env python3
"""Fetch the two most recent parent eval runs from MLflow for comparison.

Outputs a JSON object with "current" and "previous" run data to stdout.
"""
import json
import os
import sys

import mlflow
from mlflow import MlflowClient


def run_to_dict(run):
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "start_time": run.info.start_time,
        "status": run.info.status,
        "metrics": run.data.metrics,
        "params": {
            k: v
            for k, v in run.data.params.items()
            if k in [
                "test_date",
                "app_model_name",
                "eval_model_name",
                "app_llm_provider",
                "rag_preprocessor",
            ]
        },
    }


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-evals")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(
            f"Warning: experiment '{experiment_name}' not found",
            file=sys.stderr,
        )
        print(json.dumps({"current": None, "previous": None}))
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'parent'",
        order_by=["start_time DESC"],
        max_results=2,
    )

    output = {
        "current": run_to_dict(runs[0]) if len(runs) >= 1 else None,
        "previous": run_to_dict(runs[1]) if len(runs) >= 2 else None,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
