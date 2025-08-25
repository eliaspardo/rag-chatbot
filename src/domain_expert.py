from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from chain_manager import ChainManager
from prompts import condense_question_prompt, domain_expert_prompt
from constants import EXIT_WORDS, ChatbotMode, Error
from console_ui import ConsoleUI


# --- Run Chat Loop ---
def run_chat_loop(
    ui: ConsoleUI, chain_manager: ChainManager, qa_chain: ConversationalRetrievalChain
):
    ui.show_welcome_mode(ChatbotMode.DOMAIN_EXPERT)

    try:
        while True:
            question = ui.get_user_input("\n❓ Your question: ")

            if question.lower() in EXIT_WORDS:
                # TODO test whether this truly returns the goodbye message
                return "exit"

            if question.lower() == "mode":
                ui.show_mode_switch()
                break

            if not question:
                ui.show_error(Error.NOT_A_QUESTION)
                continue

            ui.show_llm_thinking()

            try:
                answer = chain_manager.ask_question(question, qa_chain)
                ui.show_answer(answer)
            except Exception as e:
                ui.show_error(Error.QUESTION_EXCEPTION, exception=e)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def domain_expert(ui: ConsoleUI, vectordb: FAISS):
    chain_manager = ChainManager(vectordb)
    print("🧠 Loading LLM.")
    llm = chain_manager.get_llm()

    print("⛓ Setting up Chain.")
    qa_chain = chain_manager.get_conversationalRetrievalChain(
        llm,
        {"prompt": domain_expert_prompt},
        condense_question_prompt=condense_question_prompt,
    )

    run_chat_loop(ui, chain_manager, qa_chain)

    return
