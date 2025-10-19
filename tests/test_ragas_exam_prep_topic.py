import os
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_preprocessor import RAGPreprocessor
from src.chain_manager import ChainManager
from src.exam_prep import setup_exam_prep_chain
from src.prompts import exam_prep_question_prompt
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

TOPICS = ["Types of business tools"]

GROUND_TRUTHS = ["What are the three types of business tools?"]


@pytest.mark.slow
@pytest.mark.skipif(not TOGETHER_API_KEY, reason="TOGETHER_API_KEY not set")
def test_ragas_exam_prep_topic():
    """
    For Exam Prep - Topic Chain with Ragas, we check retrieval (context_precision, context_recall) and faithfulness.
    faithfulness -> Is the feedback grounded in the retrieved context (no hallucinations)?
    context_precision -> Are the relevant contexts ranked higher than irrelevant ones?
    context_recall -> Did the retriever surface all relevant documents to the question/topic? Can all information in ground_truth be found somewhere in contexts?
    We're skipping answer_relevancy since the eval can't possibly infer the TOPICS based on the llm_questions
    """

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(ISTQB_DB_DIR, EMBED_MODEL)
    chain_manager = ChainManager(vectordb)
    llm = chain_manager.get_llm()
    topic_chain = setup_exam_prep_chain(
        chain_manager,
        llm,
        exam_prep_question_prompt,
    )

    llm_questions = []
    contexts_list = []

    for student_answer in TOPICS:
        llm_question = chain_manager.ask_question(student_answer, topic_chain)
        llm_questions.append(str(llm_question))

        docs = chain_manager.retriever.get_relevant_documents(student_answer)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)

    ds = Dataset.from_dict(
        {
            "question": TOPICS,
            "answer": llm_questions,
            "contexts": contexts_list,
            "ground_truth": GROUND_TRUTHS,
        }
    )
    print(*llm_questions)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    ragas_llm = get_ragas_llm()
    try:
        # We're skipping answer_relevancy as the input to the LLM is a request for a question.
        # so the eval can't infer what the input was.
        res = evaluate(
            ds,
            metrics=[faithfulness, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    print_ragas_results(res)
    assert_ragas_thresholds(res)
