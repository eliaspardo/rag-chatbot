"""Chain manager for building and running LangChain retrieval-augmented generation chains."""

import os
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_together import Together
from langchain_community.llms import Ollama
from langchain_core.language_models.llms import LLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.base import Chain
from langchain_core.vectorstores import VectorStoreRetriever
import re
import logging
from src.shared.env_loader import load_environment

logger = logging.getLogger(__name__)

load_environment()
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


class ChainManager:
    """Builds LLM instances and LangChain retrieval chains for the domain expert."""

    model: str
    temperature: float
    max_tokens: int
    together_api_key: str
    retriever: VectorStoreRetriever

    def __init__(
        self,
        vectordb: Chroma,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        retrieval_k: int = RETRIEVAL_K,
    ):
        if vectordb is None:
            raise ValueError("vectordb cannot be None")
        self.llm_provider = LLM_PROVIDER
        if not self.llm_provider or (
            self.llm_provider != "together" and self.llm_provider != "ollama"
        ):
            raise ValueError(
                "LLM_PROVIDER environment variable must be together or ollama"
            )
        if self.llm_provider == "together" and not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY environment variable is required")
        if self.llm_provider == "ollama" and not OLLAMA_BASE_URL:
            raise ValueError("OLLAMA_BASE_URL environment variable is required")
        self.model = MODEL_NAME
        self.together_api_key = TOGETHER_API_KEY
        self.base_url = OLLAMA_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retriever = vectordb.as_retriever(search_kwargs={"k": retrieval_k})

    def get_llm(self) -> LLM:
        """Instantiate and return the configured LLM (Together AI or Ollama)."""
        if self.llm_provider == "together":
            try:
                return Together(
                    model=self.model,
                    together_api_key=self.together_api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exception:
                raise Exception(
                    f"❌ Error setting up Together AI LLM: {exception}"
                ) from exception
        if self.llm_provider == "ollama":
            try:
                return Ollama(
                    model=self.model,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                )
            except Exception as exception:
                raise Exception(
                    f"❌ Error setting up Ollama LLM: {exception}"
                ) from exception
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm_provider}")

    def get_conversationalRetrievalChain(
        self,
        llm: LLM,
        prompt: dict,
        condense_question_prompt: PromptTemplate = None,
        verbose: bool = False,
    ) -> ConversationalRetrievalChain:
        """Build a ConversationalRetrievalChain with in-memory chat history."""
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

    def get_retrieval_qa_chain(
        self,
        llm: LLM,
        prompt: dict,
        verbose: bool = False,
    ) -> Chain:
        """Build a stateless RetrievalQA chain (no conversation memory)."""
        try:
            return RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self.retriever,
                chain_type_kwargs=prompt,
                verbose=verbose,
            )
        except Exception as exception:
            raise Exception(f"❌ Error setting up Chain: {exception}") from exception

    def reset_chain_memory(self, chain: Chain) -> None:
        """Clear the conversation memory of a chain that supports it."""
        if hasattr(chain, "memory") and chain.memory is not None:
            if hasattr(chain.memory, "clear"):
                chain.memory.clear()

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip trailing artifacts that some models emit after the answer."""
        # Remove <think>…</think> blocks (chain-of-thought reasoning)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove stray </think> and everything after it
        text = re.split(r"</think>", text, maxsplit=1)[0]
        # Remove anything from the first remaining XML-like tag onward
        text = re.split(r"<\w+", text, maxsplit=1)[0]
        # Remove trailing markdown section headers (### …)
        text = re.split(r"\n#{1,4}\s", text, maxsplit=1)[0]
        return text.strip()

    def ask_question(self, question: str, qa_chain: Chain) -> str:
        """Invoke the chain with a question and return the answer as a string."""
        try:
            if isinstance(qa_chain, RetrievalQA):
                response = qa_chain.invoke({"query": question})
                return self._clean_response(str(response["result"]))
            response = qa_chain.invoke({"question": question})
            return self._clean_response(str(response["answer"]))
        except Exception as exception:
            raise Exception(f"❌ Error invoking LLM: {exception}") from exception
