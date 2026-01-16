import json
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import pytest
from datasets import Dataset

from src.core.domain_expert_core import DomainExpertCore
from src.core.rag_preprocessor import RAGPreprocessor
from tests.utils.deepeval_utils import DeepEvalLLMAdapter
from tests.utils.ragas_dataset_loader import (
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
def test_correctness(ragas_test_vectordb):
    if not EVAL_DB_DIR:
        pytest.skip("EVAL_DB_DIR not set; see README for RAGAS setup.")

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(EVAL_DB_DIR, EMBED_MODEL)
    domain_expert = DomainExpertCore(vectordb)

    try:
        questions, ground_truths = load_golden_set_dataset()
    except (FileNotFoundError, GoldenSetValidationError, json.JSONDecodeError) as exc:
        pytest.fail(f"Invalid RAGAS golden set: {exc}")

    answers = []
    contexts_list = []

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
        }
    )

    # embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    deepeval_llm = DeepEvalLLMAdapter()

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=deepeval_llm,
    )

    print(f"{ds[0]['question']}")
    print(f"{ds[0]['answer']}")
    print(f"{ds[0]['ground_truth']}")
    test_case = LLMTestCase(
        input=ds[0]["question"],
        actual_output=ds[0]["answer"],
        expected_output=ds[0]["ground_truth"],
    )
    assert_test(test_case, [correctness_metric])
