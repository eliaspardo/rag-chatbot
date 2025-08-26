import os
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_together import Together
from langchain.llms.base import LLM
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


class ChainManager:
    model: str
    temperature: float
    max_tokens: int
    together_api_key: str
    retriever: VectorStoreRetriever

    def __init__(
        self,
        vectordb: FAISS,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        retrieval_k: int = RETRIEVAL_K,
    ):
        if vectordb is None:
            raise ValueError("vectordb cannot be None")
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY environment variable is required")
        self.model = MODEL_NAME
        self.together_api_key = TOGETHER_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retriever = vectordb.as_retriever(search_kwargs={"k": retrieval_k})

    # --- Initialize Together AI LLM ---
    def get_llm(self) -> LLM:
        try:
            return Together(
                model=self.model,
                together_api_key=self.together_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exception:
            raise Exception(f"❌ Error setting up LLM: {exception}") from exception

    # --- Get Conversational Chain based on a Document, resets memory ---
    def get_conversationalRetrievalChain(
        self,
        llm: LLM,
        prompt: dict,
        condense_question_prompt: PromptTemplate = None,
        verbose: bool = False,
    ) -> ConversationalRetrievalChain:
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
            kwargs = {
                "llm": llm,
                "retriever": self.retriever,
                "memory": memory,
                "combine_docs_chain_kwargs": prompt,
                "verbose": verbose,
            }

            if condense_question_prompt is not None:
                kwargs["condense_question_prompt"] = condense_question_prompt

            return ConversationalRetrievalChain.from_llm(**kwargs)

        except Exception as exception:
            raise Exception(f"❌ Error setting up Chain: {exception}") from exception

    # --- Run QA Chain ---
    def ask_question(self, question: str, qa_chain: Chain) -> str:
        try:
            response = qa_chain.invoke({"question": question})
            return str(response["answer"])
        except Exception as exception:
            raise Exception(f"❌ Error invoking LLM: {exception}") from exception
