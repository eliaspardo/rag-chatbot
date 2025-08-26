from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from chain_manager import ChainManager
from prompts import condense_question_prompt, domain_expert_prompt
from constants import EXIT_WORDS, ChatbotMode, Error
from console_ui import ConsoleUI
from exceptions import ExitApp
import logging

logger = logging.getLogger(__name__)


# --- Run Chat Loop ---
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


def domain_expert(ui: ConsoleUI, vectordb: FAISS) -> None:
    try:
        chain_manager = ChainManager(vectordb)
    except ValueError as exception:
        logger.error(f"Error setting up chain: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()
    ui.show_info_message("\n🧠 Loading LLM.")
    try:
        llm = chain_manager.get_llm()
    except Exception as exception:
        logger.error(f"Error getting LLM: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()

    ui.show_info_message("\n⛓ Setting up Chain.")
    try:
        qa_chain = chain_manager.get_conversationalRetrievalChain(
            llm,
            {"prompt": domain_expert_prompt},
            condense_question_prompt=condense_question_prompt,
        )
    except Exception as exception:
        logger.error(f"Error setting up chain: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()

    run_chat_loop(ui, chain_manager, qa_chain)
    return
