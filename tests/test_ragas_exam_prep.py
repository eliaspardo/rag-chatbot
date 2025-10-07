import os
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_preprocessor import RAGPreprocessor
from src.chain_manager import ChainManager
from src.exam_prep import setup_exam_prep_chain
from src.prompts import exam_prep_question_prompt, exam_prep_answer_prompt
from tests.utils.ragas_utils import (
    print_ragas_results,
    assert_ragas_thresholds,
    get_ragas_llm,
)
import logging

logger = logging.getLogger(__name__)


ISTQB_DB_DIR = "tests/data/istqb_tm_faiss_db"
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

QUESTIONS = [
    # "What are the three types of business tools?",
    "Explain risk-based testing at a high level.",
    # "What are the different test metrics categories?",
]

GROUND_TRUTHS = [
    # "Comercial tools, open-source tools and custom tools.",
    "Risk-based testing is an approach that prioritizes testing efforts "
    "based on the risk of failure and its potential impact, "
    "focusing resources on the most critical areas first.",
    # "Project metrics, product metrics and process metrics.",
]


@pytest.mark.slow
@pytest.mark.skipif(not TOGETHER_API_KEY, reason="TOGETHER_API_KEY not set")
def test_ragas_exam_prep_answer():

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(ISTQB_DB_DIR, EMBED_MODEL)
    chain_manager = ChainManager(vectordb)
    llm = chain_manager.get_llm()
    answer_chain = setup_exam_prep_chain(
        chain_manager,
        llm,
        exam_prep_answer_prompt,
    )

    answers = []
    contexts_list = []

    for question in QUESTIONS:
        answer = chain_manager.ask_question(question, answer_chain)
        answers.append(str(answer))

        docs = chain_manager.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)

    ds = Dataset.from_dict(
        {
            "question": QUESTIONS,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": GROUND_TRUTHS,
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
    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    print_ragas_results(res)
    assert_ragas_thresholds(res)
