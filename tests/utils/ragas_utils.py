# tests/utils/ragas_utils.py
import os
from dotenv import load_dotenv

load_dotenv(override=False)

# Default thresholds (can be overridden by environment variables)
RAGAS_ANSWER_RELEVANCY_THRESHOLD = float(
    os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", "0.4")
)
RAGAS_FAITHFULNESS_THRESHOLD = float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.4"))
RAGAS_ANSWER_RELEVANCY_MIN = float(os.getenv("RAGAS_ANSWER_RELEVANCY_MIN", "0.2"))
RAGAS_FAITHFULNESS_MIN = float(os.getenv("RAGAS_FAITHFULNESS_MIN", "0.2"))


def print_ragas_results(results, dataset=None):
    """
    Pretty print RAGAS evaluation results.

    Args:
        results: RAGAS evaluation results object
        dataset: Optional dataset to show questions alongside results
    """
    results_df = results.to_pandas()

    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS:")
    print("=" * 80)

    if dataset:
        # Include questions in output
        results_df["question"] = [dataset[i]["question"] for i in range(len(dataset))]
        # Reorder columns to show question first
        cols = ["question"] + [col for col in results_df.columns if col != "question"]
        results_df = results_df[cols]

    print(results_df.to_string())

    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS:")
    print("-" * 80)
    metrics_cols = [
        col
        for col in results_df.columns
        if col not in ["question", "answer", "contexts"]
    ]
    print(results_df[metrics_cols].describe())
    print("=" * 80 + "\n")


def assert_ragas_thresholds(
    results,
    answer_relevancy_threshold: float = RAGAS_ANSWER_RELEVANCY_THRESHOLD,
    faithfulness_threshold: float = RAGAS_FAITHFULNESS_THRESHOLD,
    min_answer_relevancy: float = RAGAS_ANSWER_RELEVANCY_MIN,
    min_faithfulness: float = RAGAS_FAITHFULNESS_MIN,
):
    """
    Assert RAGAS evaluation results meet quality thresholds.

    Args:
        results: RAGAS evaluation results object
        answer_relevancy_threshold: Minimum mean answer_relevancy score
        faithfulness_threshold: Minimum mean faithfulness score
        min_answer_relevancy: Minimum individual answer_relevancy score (safety check)
        min_faithfulness: Minimum individual faithfulness score (safety check)

    Raises:
        AssertionError: If any threshold is not met
    """
    results_df = results.to_pandas()

    # Check mean scores
    answer_rel_mean = results_df["answer_relevancy"].mean()
    faithfulness_mean = results_df["faithfulness"].mean()

    assert answer_rel_mean >= answer_relevancy_threshold, (
        f"❌ Answer relevancy {answer_rel_mean:.3f} below threshold {answer_relevancy_threshold}\n"
        f"Per-question scores: {results_df['answer_relevancy'].tolist()}"
    )

    assert faithfulness_mean >= faithfulness_threshold, (
        f"❌ Faithfulness {faithfulness_mean:.3f} below threshold {faithfulness_threshold}\n"
        f"Per-question scores: {results_df['faithfulness'].tolist()}"
    )

    # Safety checks for individual questions
    min_ans_rel = results_df["answer_relevancy"].min()
    assert (
        min_ans_rel >= min_answer_relevancy
    ), f"❌ At least one question has answer_relevancy {min_ans_rel:.3f} below minimum {min_answer_relevancy}"

    min_faith = results_df["faithfulness"].min()
    assert (
        min_faith >= min_faithfulness
    ), f"❌ At least one question has faithfulness {min_faith:.3f} below minimum {min_faithfulness}"

    print(f"✅ All RAGAS thresholds passed!")
    print(f"   Answer Relevancy: {answer_rel_mean:.3f} (min: {min_ans_rel:.3f})")
    print(f"   Faithfulness: {faithfulness_mean:.3f} (min: {min_faith:.3f})")
