from collections import defaultdict
from datetime import datetime, timezone
from importlib import import_module
import json
import os
import mlflow
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
import pytest
from datasets import Dataset

from src.core.domain_expert_core import DomainExpertCore
from src.core.prompts import domain_expert_prompt
from src.core.rag_preprocessor import RAGPreprocessor
from tests.utils.deepeval_utils import DeepEvalLLMAdapter
from tests.utils.eval_dataset_loader import (
    load_golden_set_dataset,
    GoldenSetValidationError,
)

import logging
from src.env_loader import load_environment

logger = logging.getLogger(__name__)
load_environment()

EVAL_DB_DIR = os.getenv("EVAL_DB_DIR")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL_NAME = os.getenv("MODEL_NAME")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", MODEL_NAME)
EVAL_LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", LLM_PROVIDER).strip().lower()
EVAL_TOGETHER_API_KEY = os.getenv("EVAL_TOGETHER_API_KEY", TOGETHER_API_KEY)
EVAL_OLLAMA_BASE_URL = os.getenv("EVAL_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
EVAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "300"))
DEEPEVAL_MAX_CONCURRENT = int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "4"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-evals")
RAG_PREPROCESSOR = os.getenv("RAG_PREPROCESSOR", "legacy")
DOCLING_EXPORT_TYPE = os.getenv("DOCLING_EXPORT_TYPE", "")

GROUNDING_VERSION = "v1"
COMPLETENESS_VERSION = "v3"
REASONING_VERSION = "v1"
CORRECTNESS_VERSION = "v3"

_grounding_module = import_module(f"tests.evals.metrics.grounding.{GROUNDING_VERSION}")
GROUNDING_EVALUATION_STEPS = _grounding_module.EVALUATION_STEPS
GROUNDING_METADATA = _grounding_module.METADATA

_completeness_module = import_module(
    f"tests.evals.metrics.completeness.{COMPLETENESS_VERSION}"
)
COMPLETENESS_EVALUATION_STEPS = _completeness_module.EVALUATION_STEPS
COMPLETENESS_METADATA = _completeness_module.METADATA

_reasoning_module = import_module(f"tests.evals.metrics.reasoning.{REASONING_VERSION}")
REASONING_EVALUATION_STEPS = _reasoning_module.EVALUATION_STEPS
REASONING_METADATA = _reasoning_module.METADATA


class EvalResults:
    def __init__(self):
        self.metric_scores = defaultdict(list)
        self.failures = []


@pytest.fixture
def deepeval_metrics():
    deepeval_llm = DeepEvalLLMAdapter()

    grounding_and_correctness_metric = GEval(
        name="Grounding",
        evaluation_steps=GROUNDING_EVALUATION_STEPS,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.5,
        model=deepeval_llm,
    )

    completeness_metric = GEval(
        name="Completeness",
        evaluation_steps=COMPLETENESS_EVALUATION_STEPS,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.5,
        model=deepeval_llm,
    )

    reasoning_metric = GEval(
        name="Reasoning",
        evaluation_steps=REASONING_EVALUATION_STEPS,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.5,
        model=deepeval_llm,
    )

    return [grounding_and_correctness_metric, completeness_metric, reasoning_metric]


@pytest.fixture(scope="session")
def mlflow_parent_run(run_name):
    """Start parent run for entire test session"""
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    eval_results = EvalResults()

    with mlflow.start_run(run_name=run_name):
        # Log suite-level params
        mlflow.log_param("test_date", datetime.now().isoformat())
        mlflow.log_param("app_llm_provider", LLM_PROVIDER)
        mlflow.log_param("app_model_name", MODEL_NAME)
        mlflow.log_param("eval_llm_provider", EVAL_LLM_PROVIDER)
        mlflow.log_param("eval_model_name", EVAL_MODEL_NAME)
        mlflow.log_param("rag_preprocessor", RAG_PREPROCESSOR)
        mlflow.log_param("docling_export_type", DOCLING_EXPORT_TYPE)

        mlflow.log_param(
            "metrics_file_grounding",
            f"tests/evals/metrics/grounding/{GROUNDING_VERSION}.py",
        )
        mlflow.log_param(
            "metrics_file_correctness",
            f"tests/evals/metrics/correctness/{CORRECTNESS_VERSION}.py",
        )
        mlflow.log_dict(
            GROUNDING_EVALUATION_STEPS,
            f"tests/evals/metrics/evaluation_steps/grounding/{GROUNDING_VERSION}.py",
        )
        mlflow.log_dict(
            GROUNDING_METADATA,
            f"tests/evals/metrics/metadata/grounding/{GROUNDING_VERSION}.py",
        )
        mlflow.log_dict(
            COMPLETENESS_EVALUATION_STEPS,
            f"tests/evals/metrics/evaluation_steps/completeness/{COMPLETENESS_VERSION}.py",
        )
        mlflow.log_dict(
            COMPLETENESS_METADATA,
            f"tests/evals/metrics/metadata/completeness/{COMPLETENESS_VERSION}.py",
        )
        mlflow.log_dict(
            REASONING_EVALUATION_STEPS,
            f"tests/evals/metrics/evaluation_steps/reasoning/{REASONING_VERSION}.py",
        )
        mlflow.log_dict(
            REASONING_METADATA,
            f"tests/evals/metrics/metadata/reasoning/{REASONING_VERSION}.py",
        )
        mlflow.log_dict(
            {"template": domain_expert_prompt.template},
            "src/core/prompts/domain_expert_prompt.json",
        )
        mlflow.set_tag("run_type", "parent")

        yield run_name, eval_results  # Tests run here

        # Aggregate metrics after all tests complete
        for metric_name, scores in eval_results.metric_scores.items():
            if not scores:
                continue
            mlflow.log_metric(f"{metric_name}_mean", sum(scores) / len(scores))
            mlflow.log_metric(f"{metric_name}_min", min(scores))
            mlflow.log_metric(f"{metric_name}_max", max(scores))

        if eval_results.failures:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("failure_count", len(eval_results.failures))
            pytest.fail(
                f"{len(eval_results.failures)} test case(s) failed", pytrace=False
            )
        else:
            mlflow.set_tag("status", "passed")


@pytest.mark.slow
@pytest.mark.deepeval
@pytest.mark.skipif(
    EVAL_LLM_PROVIDER == "together" and not EVAL_TOGETHER_API_KEY,
    reason="EVAL_TOGETHER_API_KEY environment variable is required",
)
@pytest.mark.skipif(
    EVAL_LLM_PROVIDER == "ollama" and not EVAL_OLLAMA_BASE_URL,
    reason="EVAL_OLLAMA_BASE_URL environment variable is required",
)
@pytest.mark.skipif(
    not EVAL_LLM_PROVIDER
    or (EVAL_LLM_PROVIDER != "together" and EVAL_LLM_PROVIDER != "ollama"),
    reason="EVAL_LLM_PROVIDER environment variable must be together or ollama",
)
def test_deepeval_domain_expert(
    eval_test_vectordb, run_specific_question_id, deepeval_metrics, mlflow_parent_run
):
    __tracebackhide__ = True
    if not EVAL_DB_DIR:
        pytest.skip("EVAL_DB_DIR not set; see README for RAGAS setup.")

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(EVAL_DB_DIR, EMBED_MODEL)
    domain_expert = DomainExpertCore(vectordb)

    try:
        questions, ground_truths, question_ids = load_golden_set_dataset(
            run_specific_question_id=run_specific_question_id
        )
    except (FileNotFoundError, GoldenSetValidationError, json.JSONDecodeError) as exc:
        pytest.fail(f"Invalid RAGAS golden set: {exc}")

    answers = []
    contexts_list = []
    parent_run_name, eval_results = mlflow_parent_run

    for question in questions:
        answer = domain_expert.ask_question(question)
        answers.append(str(answer))

        docs = domain_expert.chain_manager.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)
        domain_expert.chain_manager.reset_chain_memory(domain_expert.qa_chain)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
            "question_id": question_ids,
        }
    )

    # Create test cases for all items in the dataset
    test_cases = [
        LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            expected_output=item["ground_truth"],
            context=item["contexts"],
            additional_metadata={"question_id": item["question_id"]},
        )
        for item in ds
    ]

    evaluation_result = evaluate(
        test_cases=test_cases,
        metrics=deepeval_metrics,
        display_config=DisplayConfig(show_indicator=True, print_results=False),
        async_config=AsyncConfig(
            max_concurrent=DEEPEVAL_MAX_CONCURRENT, throttle_value=1
        ),
        hyperparameters={"prompt": domain_expert_prompt.template},
    )

    for test_result in evaluation_result.test_results:
        question_id = (
            test_result.additional_metadata.get("question_id")
            if test_result.additional_metadata.get("question_id")
            else None
        )

        with mlflow.start_run(
            run_name=f"question-{question_id}",
            nested=True,
        ):
            mlflow.log_param("question_id", question_id)
            mlflow.log_param("question", test_result.input)
            mlflow.log_param("actual output", test_result.actual_output)
            mlflow.log_param("expected output", test_result.expected_output)
            mlflow.log_param("test result", test_result.success)
            mlflow.log_param("parent_run", parent_run_name)
            mlflow.log_param(
                "timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
            )

            mlflow.set_tag("run_type", "child")
            mlflow.set_tag("parent_run", parent_run_name)
            mlflow.set_tag("question", f"question-{question_id}")

            metrics_data = test_result.metrics_data or []
            for metric_data in metrics_data:
                sanitized_metric_name = sanitize_mlflow_name(metric_data.name)
                score = metric_data.score
                if score is not None:
                    eval_results.metric_scores[sanitized_metric_name].append(score)
                    mlflow.log_metric(sanitized_metric_name, score)
                    mlflow.log_param(f"{sanitized_metric_name} score", score)
                    mlflow.log_param(
                        f"{sanitized_metric_name} reason", metric_data.reason
                    )
                mlflow.log_dict(
                    {
                        "metric_name": metric_data.name,
                        "score": score,
                        "reason": metric_data.reason,
                        "success": metric_data.success,
                        "threshold": metric_data.threshold,
                    },
                    f"metrics/{sanitized_metric_name}_result.json",
                )

            mlflow.log_text(
                test_result.actual_output or "",
                "outputs/actual_output.txt",
            )
            mlflow.log_text(
                test_result.expected_output or "",
                "outputs/expected_output.txt",
            )

            if not test_result.success:
                eval_results.failures.append(test_result.name)
                mlflow.set_tag("status", "failed")
                mlflow.log_param("failure", "metric threshold not met")
                mlflow.log_param("context", test_result.context)
            else:
                mlflow.set_tag("status", "passed")
                mlflow.log_param("context", test_result.context)


def sanitize_mlflow_name(name: str) -> str:
    """Remove characters that MLflow doesn't allow"""
    return name.replace("[", "").replace("]", "").replace(" ", "_")
