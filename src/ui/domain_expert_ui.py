from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from src.core.chain_manager import ChainManager
from src.core.constants import EXIT_WORDS, ChatbotMode, Error
from src.ui.console_ui import ConsoleUI
from src.core.exceptions import ExitApp
from src.core.domain_expert_core import setup_domain_expert_chain
import logging

logger = logging.getLogger(__name__)


def run_chat_loop(
    ui: ConsoleUI, chain_manager: ChainManager, qa_chain: ConversationalRetrievalChain
) -> None:
    ui.show_welcome_mode(ChatbotMode.DOMAIN_EXPERT)

    try:
        while True:
            question = ui.get_user_input("\n❓ Your question: ")

            if question.lower() in EXIT_WORDS:
                raise ExitApp()

            if question.lower() == "mode":
                ui.show_mode_switch()
                break

            if not question:
                ui.show_error(Error.NOT_A_QUESTION)
                continue

            ui.show_info_message("\n🤔 Thinking...")

            try:
                answer = chain_manager.ask_question(question, qa_chain)
                ui.show_answer(answer)
            except Exception as exception:
                logger.error(f"Error retrieving answer: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                continue

    except KeyboardInterrupt:
        raise ExitApp()


def domain_expert_ui(ui: ConsoleUI, vectordb: FAISS) -> None:
    try:
        chain_manager = ChainManager(vectordb)
    except ValueError as exception:
        logger.error(f"Error instantiating Chain Manager: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()
    ui.show_info_message("\n🧠 Setting up Domain Expert.")
    try:
        qa_chain = setup_domain_expert_chain(
            chain_manager
        )
    except Exception as exception:
        logger.error(f"Error setting up Domain Expert: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()

    run_chat_loop(ui, chain_manager, qa_chain)
    return