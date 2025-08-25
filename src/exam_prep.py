import sys
import textwrap
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from chain_manager import ChainManager
from prompts import exam_prep_question_prompt, exam_prep_answer_prompt
from constants import EXIT_WORDS, ChatbotMode, Error
from console_ui import ConsoleUI


# --- Run Chat Loop ---
def run_chat_loop(
    ui: ConsoleUI,
    chain_manager: ChainManager,
    question_chain: ConversationalRetrievalChain,
    answer_chain: ConversationalRetrievalChain,
):
    ui.show_welcome_mode(ChatbotMode.EXAM_PREP)

    try:
        while True:
            topic = ui.get_user_input(
                "\n🧠 Section / topic you want to be quizzed about: "
            )

            if topic.lower() in EXIT_WORDS:
                # TODO test whether this truly returns the goodbye message
                return "exit"

            if topic.lower() == "mode":
                ui.show_mode_switch()
                break

            if not topic:
                ui.show_error(Error.NOT_A_TOPIC)
                continue

            ui.show_llm_thinking()
            try:
                llm_question = chain_manager.ask_question(topic, question_chain)
                ui.show_llm_question(llm_question)
            except Exception as e:
                ui.show_error(Error.QUESTION_EXCEPTION, exception=e)

            user_answer = input("\n📝 Your answer: ").strip()

            if user_answer.lower() in EXIT_WORDS:
                # TODO test whether this truly returns the goodbye message
                return "exit"

            if user_answer.lower() == "mode":
                ui.show_mode_switch()
                break

            if not user_answer:
                ui.show_error(Error.NO_USER_ANSWER)
                continue

            # Evaluate user's answer
            llm_question_user_answer = llm_question + "\n" + user_answer
            ui.show_llm_thinking()
            try:
                llm_answer = chain_manager.ask_question(
                    llm_question_user_answer, answer_chain
                )
                ui.show_answer(llm_answer)
            except Exception as e:
                ui.show_error(Error.ANSWER_EXCEPTION, exception=e)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def exam_prep(ui: ConsoleUI, vectordb: FAISS):
    chain_manager = ChainManager(vectordb)
    print("🧠 Loading LLM.")
    llm = chain_manager.get_llm()

    print("⛓ Setting up Chains.")
    question_chain = chain_manager.get_conversationalRetrievalChain(
        llm, {"prompt": exam_prep_question_prompt}
    )

    answer_chain = chain_manager.get_conversationalRetrievalChain(
        llm, {"prompt": exam_prep_answer_prompt}
    )

    run_chat_loop(chain_manager, question_chain, answer_chain)

    return
