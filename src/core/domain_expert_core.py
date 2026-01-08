from langchain_community.vectorstores import FAISS
from src.core.chain_manager import ChainManager
from src.core.prompts import condense_question_prompt, domain_expert_prompt
from src.core.exceptions import DomainExpertSetupException
import logging

logger = logging.getLogger(__name__)


class DomainExpertCore:

    def __init__(self, vectordb: FAISS):
        try:
            self.chain_manager = ChainManager(vectordb)
        except ValueError as exception:
            logger.error(f"Error instantiating Chain Manager: {exception}")
            raise DomainExpertSetupException(
                "Error instantiating Chain Manager"
            ) from exception
        try:
            llm = self.chain_manager.get_llm()
        except Exception as exception:
            logger.error(f"Failed to get LLM: {exception}")
            raise DomainExpertSetupException("Error getting LLM") from exception

        try:
            self.qa_chain = self.chain_manager.get_conversationalRetrievalChain(
                llm,
                {"prompt": domain_expert_prompt},
                condense_question_prompt=condense_question_prompt,
            )

        except Exception as exception:
            logger.error(f"Failed to create QA chain: {exception}")
            raise DomainExpertSetupException("Error setting up QA chain") from exception

    def ask_question(self, question: str) -> str:
        try:
            answer = self.chain_manager.ask_question(question, self.qa_chain)
        except Exception as exception:
            logger.error(f"Error retrieving answer: {exception}")
            raise DomainExpertSetupException("Error retrieving answer") from exception
        return answer
