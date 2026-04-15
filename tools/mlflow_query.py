import os
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RAG Chatbot")


def load_config(cli_uri=None, cli_experiment_id=None) -> tuple[str, str]:
    if cli_uri:
        tracking_uri = cli_uri
    elif MLFLOW_TRACKING_URI:
        tracking_uri = MLFLOW_TRACKING_URI
    else:
        raise ValueError("No MLFLOW_TRACKING_URI defined!")
    if cli_experiment_id:
        experiment_name = cli_experiment_id
    elif MLFLOW_EXPERIMENT_NAME:
        experiment_name = MLFLOW_EXPERIMENT_NAME
    else:
        raise ValueError("No MLFLOW_EXPERIMENT_NAME defined!")
    return tracking_uri, experiment_name


class MLFlowQuery:
    def __init__(self, mlflow_client):
        self.mlflow_client = mlflow_client

    def get_parent_runs(self, experiment_name, max_results=100) -> str:
        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        except Exception:
            raise
        experiment_id = experiment.experiment_id
        print(f"Experiment ID: {experiment.experiment_id}")
        parent_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.run_type = 'parent'",
            max_results=10,
        )
        # print(parent_runs[0])
        for run in parent_runs[:5]:
            print("#########")
            print(f"run_name: {run.info.run_name}")
            print(f"run_id: {run.info.run_id}")
            try:
                metrics = run.data.metrics
                print(
                    f"Completeness_GEval_mean: {metrics['Completeness_GEval_mean']} | "
                    f"Grounding_GEval_mean: {metrics['Grounding_GEval_mean']} | "
                    f"Reasoning_GEval_mean: {metrics['Reasoning_GEval_mean']}"
                )
            except Exception:
                print("Error getting metrics!")

    def get_child_runs(self, experiment_name, parent_run_id, status_filter=None):
        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        except Exception:
            raise
        experiment_id = experiment.experiment_id
        child_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )
        # print(child_runs[0])
        for run in child_runs[:5]:
            print("#########")
            print(f"run_name: {run.info.run_name}")
            print(f"question: {run.data.params['question'][:100]}")
            try:
                metrics = run.data.metrics
                print(
                    f"Completeness: {metrics['Completeness_GEval']} | "
                    f"Grounding: {metrics['Grounding_GEval']} | "
                    f"Reasoning: {metrics['Reasoning_GEval']}"
                )
            except Exception:
                print("Error getting metrics!")


def main() -> None:
    tracking_uri, experiment_name = load_config()
    mlflow_client = MlflowClient(tracking_uri)
    mlflow_query = MLFlowQuery(mlflow_client)
    mlflow_query.get_parent_runs(experiment_name, 2)
    mlflow_query.get_child_runs(experiment_name, "9fb3fc9ac8594594bd7c51e381c8a15a")


if __name__ == "__main__":
    main()

# Usage
# mlflow_query.py list [-n COUNT] [--tracking-uri URI] [--experiment NAME]
# mlflow_query.py show <RUN_NAME_OR_ID> [--tracking-uri URI] [--experiment NAME] [--status passed|failed]
# [--fields FIELD1,FIELD2,...]
