import os
from langchain_together import Together
from langchain.llms.base import LLM
import logging
from src.env_loader import load_environment

logger = logging.getLogger(__name__)
load_environment()

# Default thresholds (can be overridden by environment variables)
RAGAS_ANSWER_RELEVANCY_THRESHOLD = float(
    os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", "0.4")
)
RAGAS_FAITHFULNESS_THRESHOLD = float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.4"))
RAGAS_PRECISION_THRESHOLD = float(os.getenv("RAGAS_PRECISION_THRESHOLD", "0.4"))
RAGAS_RECALL_THRESHOLD = float(os.getenv("RAGAS_RECALL_THRESHOLD", "0.4"))

RAGAS_ANSWER_RELEVANCY_MIN = float(os.getenv("RAGAS_ANSWER_RELEVANCY_MIN", "0.2"))
RAGAS_FAITHFULNESS_MIN = float(os.getenv("RAGAS_FAITHFULNESS_MIN", "0.2"))
RAGAS_PRECISION_MIN = float(os.getenv("RAGAS_PRECISION_MIN", "0.2"))
RAGAS_ANSWER_RECALL_MIN = float(os.getenv("RAGAS_ANSWER_RECALL_MIN", "0.2"))

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
RAGAS_MAX_TOKENS = int(os.getenv("RAGAS_MAX_TOKENS", "512"))


def print_ragas_results(results, dataset=None):
    """
    Pretty print RAGAS evaluation results.

    Args:
        results: RAGAS evaluation results object
        dataset: Optional dataset to show questions alongside results
    """
    results_df = results.to_pandas()

    logger.info("\n" + "=" * 80)
    logger.info("RAGAS EVALUATION RESULTS:")
    logger.info("=" * 80)

    if dataset:
        # Include questions in output
        results_df["question"] = [dataset[i]["question"] for i in range(len(dataset))]
        # Reorder columns to show question first
        cols = ["question"] + [col for col in results_df.columns if col != "question"]
        results_df = results_df[cols]

    logger.info(results_df.to_string())

    logger.info("\n" + "-" * 80)
    logger.info("SUMMARY STATISTICS:")
    logger.info("-" * 80)
    metrics_cols = [
        col
        for col in results_df.columns
        if col not in ["question", "answer", "contexts"]
    ]
    logger.info(results_df[metrics_cols].describe())
    logger.info("=" * 80 + "\n")


def assert_ragas_thresholds(
    results,
    answer_relevancy_threshold: float = RAGAS_ANSWER_RELEVANCY_THRESHOLD,
    faithfulness_threshold: float = RAGAS_FAITHFULNESS_THRESHOLD,
    precision_threshold: float = RAGAS_PRECISION_THRESHOLD,
    recall_threshold: float = RAGAS_RECALL_THRESHOLD,
    answer_relevancy_min: float = RAGAS_ANSWER_RELEVANCY_MIN,
    faithfulness_min: float = RAGAS_FAITHFULNESS_MIN,
    precision_min: float = RAGAS_PRECISION_MIN,
    recall_min: float = RAGAS_ANSWER_RECALL_MIN,
):
    """
    Assert RAGAS evaluation results meet quality thresholds.

    Args:
        results: RAGAS evaluation results object
        answer_relevancy_threshold: Minimum mean answer_relevancy score
        faithfulness_threshold: Minimum mean faithfulness score
        faithfulness_threshold: Minimum mean precision score
        min_answer_relevancy: Minimum individual answer_relevancy score (safety check)
        min_faithfulness: Minimum individual faithfulness score (safety check)
        min_precision: Minimum individual precision score (safety check)

    Raises:
        AssertionError: If any threshold is not met
    """
    results_df = results.to_pandas()

    # Check mean scores
    if "answer_relevancy" in results_df.columns:
        answer_rel_mean = results_df["answer_relevancy"].mean()
        assert answer_rel_mean >= answer_relevancy_threshold, (
            f"❌ Answer relevancy {answer_rel_mean:.3f} below threshold {answer_relevancy_threshold}\n"
            f"Per-question scores: {results_df['answer_relevancy'].tolist()}"
        )
    if "faithfulness" in results_df.columns:
        faithfulness_mean = results_df["faithfulness"].mean()
        assert faithfulness_mean >= faithfulness_threshold, (
            f"❌ Faithfulness {faithfulness_mean:.3f} below threshold {faithfulness_threshold}\n"
            f"Per-question scores: {results_df['faithfulness'].tolist()}"
        )

    if "context_precision" in results_df.columns:
        precision_mean = results_df["context_precision"].mean()
        assert precision_mean >= precision_threshold, (
            f"❌ Precision {precision_mean:.3f} below threshold {precision_threshold}\n"
            f"Per-question scores: {results_df['context_precision'].tolist()}"
        )
    if "context_recall" in results_df.columns:
        recall_mean = results_df["context_recall"].mean()
        assert recall_mean >= recall_threshold, (
            f"❌ Recall {recall_mean:.3f} below threshold {recall_threshold}\n"
            f"Per-question scores: {results_df['context_recall'].tolist()}"
        )

    # Safety checks for individual questions
    if "answer_relevancy" in results_df.columns:
        min_ans_rel = results_df["answer_relevancy"].min()
        assert (
            min_ans_rel >= answer_relevancy_min
        ), f"❌ At least one question has answer_relevancy {min_ans_rel:.3f} below minimum {answer_relevancy_min}"
    if "faithfulness" in results_df.columns:
        min_faith = results_df["faithfulness"].min()
        assert (
            min_faith >= faithfulness_min
        ), f"❌ At least one question has faithfulness {min_faith:.3f} below minimum {faithfulness_min}"

    if "context_precision" in results_df.columns:
        min_prec = results_df["context_precision"].min()
        assert (
            min_prec >= precision_min
        ), f"❌ At least one question has precision {min_prec:.3f} below minimum {precision_min}"

    if "context_recall" in results_df.columns:
        min_rec = results_df["context_recall"].min()
        assert (
            min_rec >= recall_min
        ), f"❌ At least one question has recall {min_rec:.3f} below minimum {recall_min}"

    print("✅ All RAGAS thresholds passed!")
    if "answer_relevancy" in results_df.columns:
        print(f"   Answer Relevancy: {answer_rel_mean:.3f} (min: {min_ans_rel:.3f})")
    if "faithfulness" in results_df.columns:
        print(f"   Faithfulness: {faithfulness_mean:.3f} (min: {min_faith:.3f})")
    if "context_precision" in results_df.columns:
        print(f"   Precision: {precision_mean:.3f} (min: {min_prec:.3f})")
    if "context_recall" in results_df.columns:
        print(f"   Recall: {recall_mean:.3f} (min: {min_rec:.3f})")


def get_ragas_llm() -> LLM:
    try:
        return Together(
            model=MODEL_NAME,
            together_api_key=TOGETHER_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=RAGAS_MAX_TOKENS,
        )
    except Exception as exception:
        raise Exception(f"❌ Error setting up LLM: {exception}") from exception
