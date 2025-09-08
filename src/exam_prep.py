from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from src.chain_manager import ChainManager
from src.prompts import exam_prep_question_prompt, exam_prep_answer_prompt
from src.constants import EXIT_WORDS, ChatbotMode, Error
from src.console_ui import ConsoleUI
from src.exceptions import ExitApp
import logging

logger = logging.getLogger(__name__)


# --- Run Chat Loop ---
def run_chat_loop(
    ui: ConsoleUI,
    chain_manager: ChainManager,
    question_chain: ConversationalRetrievalChain,
    answer_chain: ConversationalRetrievalChain,
) -> None:
    ui.show_welcome_mode(ChatbotMode.EXAM_PREP)

    try:
        while True:
            topic = ui.get_user_input(
                "\n🧠 Section / topic you want to be quizzed about: "
            )

            if topic.lower() in EXIT_WORDS:
                raise ExitApp()

            if topic.lower() == "mode":
                ui.show_mode_switch()
                break

            if not topic:
                ui.show_error(Error.NOT_A_TOPIC)
                continue

            ui.show_info_message("\n🤔 Thinking...")
            try:
                llm_question = chain_manager.ask_question(topic, question_chain)
                ui.show_llm_question(llm_question)
            except Exception as exception:
                logger.error(f"Error retrieving question: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                ui.show_info_message("Please try rephrasing your question.")
                continue
            user_answer = input("\n📝 Your answer: ").strip()

            if user_answer.lower() in EXIT_WORDS:
                raise ExitApp()

            if user_answer.lower() == "mode":
                ui.show_mode_switch()
                break

            if not user_answer:
                ui.show_error(Error.NO_USER_ANSWER)
                continue

            # Evaluate user's answer
            llm_question_user_answer = llm_question + "\n" + user_answer
            ui.show_info_message("\n🤔 Thinking...")
            try:
                llm_answer = chain_manager.ask_question(
                    llm_question_user_answer, answer_chain
                )
                ui.show_answer(llm_answer)
            except Exception as exception:
                logger.error(f"Error retrieving answer: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                ui.show_info_message("Please try rephrasing your answer.")
                continue

    except KeyboardInterrupt:
        raise ExitApp()


def exam_prep(ui: ConsoleUI, vectordb: FAISS) -> None:
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

    ui.show_info_message("\n⛓ Setting up Chains.")
    try:
        question_chain = chain_manager.get_conversationalRetrievalChain(
            llm, {"prompt": exam_prep_question_prompt}
        )

        answer_chain = chain_manager.get_conversationalRetrievalChain(
            llm, {"prompt": exam_prep_answer_prompt}
        )
    except Exception as exception:
        logger.error(f"Error setting up chains: {exception}")
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()

    run_chat_loop(ui, chain_manager, question_chain, answer_chain)
    return
