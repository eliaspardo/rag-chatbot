from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from src.core.chain_manager import ChainManager
from src.core.prompts import condense_question_prompt, domain_expert_prompt
from src.core.exceptions import DomainExpertSetupException
import logging

logger = logging.getLogger(__name__)


def setup_domain_expert_chain(chain_manager: ChainManager) -> ConversationalRetrievalChain:

    try:
        llm = chain_manager.get_llm()
    except Exception as exception:
        logger.error(f"Failed to get LLM: {exception}")
        raise DomainExpertSetupException("Error getting LLM") from exception

    try:
        qa_chain = chain_manager.get_conversationalRetrievalChain(
            llm,
            {"prompt": domain_expert_prompt},
            condense_question_prompt=condense_question_prompt,
        )

    except Exception as exception:
        logger.error(f"Failed to create QA chain: {exception}")
        raise DomainExpertSetupException("Error setting up QA chain") from exception
    return qa_chain
