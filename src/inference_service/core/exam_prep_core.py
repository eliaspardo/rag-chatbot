from langchain_core.vectorstores import VectorStore
from src.inference_service.core.chain_manager import ChainManager
from src.shared.prompts import (
    exam_prep_get_question_prompt,
    exam_prep_get_feedback_prompt,
)
from src.shared.exceptions import ExamPrepQueryException, ExamPrepSetupException
import logging

logger = logging.getLogger(__name__)


class ExamPrepCore:
    def __init__(self, vectordb: VectorStore):
        try:
            self.chain_manager = ChainManager(vectordb)
        except ValueError as exception:
            logger.error(f"Error setting up chain: {exception}")
            raise ExamPrepSetupException(
                "Error instantiating Chain Manager"
            ) from exception

        try:
            llm = self.chain_manager.get_llm()
        except Exception as exception:
            logger.error(f"Error getting LLM: {exception}")
            raise ExamPrepSetupException("Error getting LLM") from exception

        try:
            self.get_question_chain = self.chain_manager.get_retrieval_qa_chain(
                llm, {"prompt": exam_prep_get_question_prompt}
            )
            self.get_feedback_chain = self.chain_manager.get_retrieval_qa_chain(
                llm, {"prompt": exam_prep_get_feedback_prompt}
            )
        except Exception as exception:
            logger.error(f"Error setting up chains: {exception}")
            raise ExamPrepSetupException("Error setting up chains") from exception

    def get_question(self, topic: str):
        try:
            llm_question = self.chain_manager.ask_question(
                topic, self.get_question_chain
            )
        except Exception as exception:
            logger.error(f"Error retrieving question: {exception}")
            raise ExamPrepQueryException("Error retrieving question") from exception
        return llm_question

    def get_feedback(self, llm_question: str, user_answer: str):
        llm_question_user_answer = llm_question + "\n" + user_answer
        try:
            llm_answer = self.chain_manager.ask_question(
                llm_question_user_answer, self.get_feedback_chain
            )
        except Exception as exception:
            logger.error(f"Error retrieving answer: {exception}")
            raise ExamPrepQueryException("Error retrieving answer") from exception
        return llm_answer
