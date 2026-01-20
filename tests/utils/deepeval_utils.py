from deepeval.models import DeepEvalBaseLLM
from tests.utils.evals_utils import build_provider_llm
import logging

logger = logging.getLogger(__name__)


class DeepEvalLLMAdapter(DeepEvalBaseLLM):
    def __init__(self):
        self.model = build_provider_llm()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        self.print_llm_exchange(prompt, result)
        return result.content if hasattr(result, "content") else str(result)

    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        self.print_llm_exchange(prompt, result)
        return result.content if hasattr(result, "content") else str(result)

    def get_model_name(self) -> str:
        return self.model.model

    def print_llm_exchange(self, prompt: str, result: str) -> None:
        logger.debug("\n--- Deepeval EVAL LLM prompt ---")
        logger.debug(prompt)
        logger.debug("\n--- Deepeval EVAL LLM output ---")
        logger.debug(result)
        logger.debug("\n-------------------------------\n")
