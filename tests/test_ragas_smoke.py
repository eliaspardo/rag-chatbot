import os
import pytest
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_preprocessor import RAGPreprocessor
from src.chain_manager import ChainManager
from src.domain_expert import setup_domain_expert_chain
from src.prompts import domain_expert_prompt, condense_question_prompt
from tests.utils.ragas_utils import print_ragas_results, assert_ragas_thresholds
import logging

logging.basicConfig(level=logging.DEBUG)

ISTQB_DB_DIR = "tests/data/istqb_tm_faiss_db"
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

QUESTIONS = [
    "What are good practices for test tool introduction?",
    "Explain risk-based testing at a high level.",
    "What are the different test metrics categories?",
]


@pytest.mark.slow
@pytest.mark.skipif(not TOGETHER_API_KEY, reason="TOGETHER_API_KEY not set")
def test_ragas_domain_expert_smoke_minimal():

    rag_preprocessor = RAGPreprocessor()
    vectordb = rag_preprocessor.load_vector_store(ISTQB_DB_DIR, EMBED_MODEL)
    chain_manager = ChainManager(vectordb)
    llm = chain_manager.get_llm()
    qa_chain = setup_domain_expert_chain(
        chain_manager,
        llm,
        {"prompt": domain_expert_prompt},
        condense_question_prompt=condense_question_prompt,
    )

    answers = []
    contexts_list = []

    for question in QUESTIONS:
        answer = chain_manager.ask_question(question, qa_chain)
        answers.append(str(answer))

        docs = chain_manager.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)

    ds = Dataset.from_dict(
        {
            "question": QUESTIONS,
            "answer": answers,
            "contexts": contexts_list,
        }
    )

    # df = ds.to_pandas()
    # pd.set_option("display.max_colwidth", 100)  # Show more of each cell
    # print(df.to_string())

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    try:
        res = evaluate(
            ds,
            metrics=[answer_relevancy, faithfulness],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as e:
        print(f"Evaluation error: {e}")

    # print_ragas_results(res)
    assert_ragas_thresholds(res)
