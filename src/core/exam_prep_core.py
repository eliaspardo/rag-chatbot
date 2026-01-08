from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from src.core.chain_manager import ChainManager
from src.core.prompts import exam_prep_question_prompt, exam_prep_answer_prompt
from src.core.exceptions import ExamPrepQueryException, ExamPrepSetupException
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
            self.question_chain = setup_exam_prep_chain(
                self.chain_manager, llm, exam_prep_question_prompt
            )

            self.answer_chain = setup_exam_prep_chain(
                self.chain_manager, llm, exam_prep_answer_prompt
            )
        except Exception as exception:
            logger.error(f"Error setting up chains: {exception}")
            raise ExamPrepSetupException("Error setting up chains") from exception

    def get_question(self, topic: str):
        try:
            llm_question = self.chain_manager.ask_question(topic, self.question_chain)
        except Exception as exception:
            logger.error(f"Error retrieving question: {exception}")
            raise ExamPrepQueryException("Error retrieving question") from exception
        return llm_question

    def get_answer(self, llm_question_user_answer: str):
        try:
            llm_answer = self.chain_manager.ask_question(
                llm_question_user_answer, self.answer_chain
            )
        except Exception as exception:
            logger.error(f"Error retrieving answer: {exception}")
            raise ExamPrepQueryException("Error retrieving answer") from exception
        return llm_answer


def setup_exam_prep_chain(
    chain_manager: ChainManager,
    llm: LLM,
    prompt: PromptTemplate = None,
) -> ConversationalRetrievalChain:
    return chain_manager.get_conversationalRetrievalChain(
        llm,
        {"prompt": prompt},
    )
