import os
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.domain_expert_core import setup_domain_expert_chain
from src.core.rag_preprocessor import RAGPreprocessor
from src.core.chain_manager import ChainManager
from tests.utils.ragas_utils import (
    print_ragas_results,
    save_ragas_results,
    assert_ragas_thresholds,
    get_ragas_llm,
)
from tests.utils.ragas_dataset_loader import (
    load_golden_set_dataset,
    GoldenSetValidationError,
)
import logging
import json

logger = logging.getLogger(__name__)


RAGAS_DB_DIR = os.getenv("RAGAS_DB_DIR")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL_NAME = os.getenv("MODEL_NAME")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


@pytest.mark.slow
@pytest.mark.ragas
@pytest.mark.skipif(not TOGETHER_API_KEY, reason="TOGETHER_API_KEY not set")
def test_ragas_domain_expert(ragas_test_vectordb):  # noqa: ARG001
    """
    For domain expert with Ragas, we check retrieval (context_precision), faithfulness and answer_relevancy.
    faithfulness -> Is the feedback grounded in the retrieved context (no hallucinations)?
    context_precision -> Are the relevant contexts ranked higher than irrelevant ones?
    answer_relevancy -> infer the QUESTIONS based on the LLM's response
    """
    if not RAGAS_DB_DIR:
        pytest.skip("RAGAS_DB_DIR not set; see README for RAGAS setup.")

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(RAGAS_DB_DIR, EMBED_MODEL)
    chain_manager = ChainManager(vectordb)
    qa_chain = setup_domain_expert_chain(chain_manager)

    try:
        questions, ground_truths = load_golden_set_dataset()
    except (FileNotFoundError, GoldenSetValidationError, json.JSONDecodeError) as exc:
        pytest.fail(f"Invalid RAGAS golden set: {exc}")

    answers = []
    contexts_list = []

    for question in questions:
        answer = chain_manager.ask_question(question, qa_chain)
        answers.append(str(answer))

        docs = chain_manager.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)
        chain_manager.reset_chain_memory(qa_chain)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    ragas_llm = get_ragas_llm()
    try:
        res = evaluate(
            ds,
            metrics=[answer_relevancy, faithfulness, context_precision],
            llm=ragas_llm,
            embeddings=embeddings,
        )
    except Exception as exception:
        logger.error(f"Evaluation error: {exception}")
        pytest.fail(f"RAGAS evaluation failed: {exception}")  # pragma: no cover

    print_ragas_results(res)
    save_paths = save_ragas_results(res)
    logger.info(f"Saved RAGAS results: {save_paths}")
    assert_ragas_thresholds(res)
