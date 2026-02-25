import os
import logging
from langchain_together import Together
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
from src.shared.env_loader import load_environment

logger = logging.getLogger(__name__)
load_environment()

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", MODEL_NAME)
EVAL_LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", LLM_PROVIDER).strip().lower()
EVAL_TOGETHER_API_KEY = os.getenv("EVAL_TOGETHER_API_KEY", TOGETHER_API_KEY)
EVAL_OLLAMA_BASE_URL = os.getenv("EVAL_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", str(TEMPERATURE)))
EVAL_MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "512"))


def build_provider_llm() -> LLM:
    if not EVAL_LLM_PROVIDER or (
        EVAL_LLM_PROVIDER != "together" and EVAL_LLM_PROVIDER != "ollama"
    ):
        raise ValueError(
            "EVAL_LLM_PROVIDER environment variable must be together or ollama"
        )
    if EVAL_LLM_PROVIDER == "together" and not EVAL_TOGETHER_API_KEY:
        raise ValueError("EVAL_TOGETHER_API_KEY environment variable is required")
    if EVAL_LLM_PROVIDER == "ollama" and not EVAL_OLLAMA_BASE_URL:
        raise ValueError("EVAL_OLLAMA_BASE_URL environment variable is required")
    if EVAL_LLM_PROVIDER == "together":
        try:
            return Together(
                model=EVAL_MODEL_NAME,
                together_api_key=EVAL_TOGETHER_API_KEY,
                temperature=EVAL_TEMPERATURE,
                max_tokens=EVAL_MAX_TOKENS,
            )
        except Exception as exception:
            raise Exception(
                f"❌ Error setting up Together AI LLM: {exception}"
            ) from exception
    if EVAL_LLM_PROVIDER == "ollama":
        try:
            return Ollama(
                model=EVAL_MODEL_NAME,
                base_url=EVAL_OLLAMA_BASE_URL,
                temperature=EVAL_TEMPERATURE,
                num_predict=EVAL_MAX_TOKENS,
            )
        except Exception as exception:
            raise Exception(
                f"❌ Error setting up Ollama LLM: {exception}"
            ) from exception
    else:
        raise Exception("❌ Error setting up LLM")
