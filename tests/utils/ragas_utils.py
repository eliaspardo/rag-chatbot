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
    Pretty print RAGAS evaluation results with clean formatting.

    Args:
        results: RAGAS evaluation results object
        dataset: Optional dataset to show questions alongside results
    """
    results_df = results.to_pandas()

    logger.info("=" * 80)
    logger.info("RAGAS EVALUATION RESULTS:")
    logger.info("=" * 80)

    # Define metric columns (exclude text-heavy columns)
    metric_cols = [
        col
        for col in results_df.columns
        if col not in ["question", "answer", "contexts", "ground_truth"]
    ]

    # Print per-question results with full details
    for idx, row in results_df.iterrows():
        logger.info(f"{'─' * 80}")
        logger.info(f"Question {idx + 1}:")

        # Show the question
        if dataset and idx < len(dataset):
            question = dataset[idx]["user_input"]
        elif "user_input" in row and row["user_input"] is not None:
            question = row["user_input"]
        else:
            question = "N/A"

        logger.info(f"Q: {question}")

        # Show the generated answer
        if "response" in row and row["response"] is not None:
            answer = row["response"]
            logger.info(f"A:{answer}")

        # Show the contexts (retrieved documents)
        if "retrieved_contexts" in row and row["retrieved_contexts"] is not None:
            contexts = row["retrieved_contexts"]
            logger.info(f"Contexts ({len(contexts)} retrieved):")
            for ctx_idx, context in enumerate(contexts, 1):
                context_single_line = " ".join(context.split())
                # Truncate very long contexts
                context_str = (
                    context_single_line[:150] + "..."
                    if len(context_single_line) > 150
                    else context_single_line
                )
                logger.info(f"    [{ctx_idx}] {context_str}")

        # Print metrics for this question
        logger.info("Metrics:")
        for col in metric_cols:
            if col in row and row[col] is not None:
                # Format based on type
                value = row[col]
                if isinstance(value, (int, float)):
                    logger.info(f"    {col}: {value:.4f}")
                else:
                    logger.info(f"    {col}: {value}")

    # Print summary statistics
    logger.info("=" * 80)
    logger.info("SUMMARY STATISTICS:")
    logger.info("=" * 80)

    if metric_cols:
        summary = results_df[metric_cols].describe()

        # Format the summary nicely
        logger.info(f"{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        logger.info("─" * 80)

        for col in metric_cols:
            if col in summary.columns:
                mean_val = summary.loc["mean", col]
                std_val = summary.loc["std", col]
                min_val = summary.loc["min", col]
                max_val = summary.loc["max", col]

                logger.info(
                    f"{col:<25} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}"
                )

    logger.info("\n" + "=" * 80 + "\n")


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
        precision_threshold: Minimum mean precision score
        recall_threshold: Minimum mean recall score
        answer_relevancy_min: Minimum individual answer_relevancy score (safety check)
        faithfulness_min: Minimum individual faithfulness score (safety check)
        precision_min: Minimum individual precision score (safety check)
        recall_min: Minimum individual recall score (safety check)

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

    print("\n✅ All RAGAS thresholds passed!")
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
