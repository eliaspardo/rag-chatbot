# Project: AI-Powered Chatbot
# Tools: Together AI (LLM), FAISS (Vector DB),
# Sentence Transformers (Embeddings), LangChain

import sys
import os
from src.console_ui import ConsoleUI
from src.domain_expert import domain_expert
from src.exam_prep import exam_prep
from src.constants import ChatbotMode, EXIT_WORDS, Error
from src.rag_preprocessor import RAGPreprocessor
from src.exceptions import ExitApp, FaissException, VectorStoreException
import logging
from langchain_community.vectorstores import FAISS
from src.env_loader import load_environment


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_environment()
DB_DIR = os.getenv("DB_DIR", "faiss_db")


def run_app(ui: ConsoleUI) -> None:

    rag_preprocessor = RAGPreprocessor()
    # Process PDF if not already embedded
    if not os.path.exists(DB_DIR):
        try:
            ui.show_info_message("\n🔍 Loading PDF...")
            texts = rag_preprocessor.load_pdf_text()
            ui.show_info_message("Splitting text to docs.")
            docs = rag_preprocessor.split_text_to_docs(texts)
        except Exception as exception:
            ui.show_error(Error.EXCEPTION, exception)
            raise ExitApp()

        if not docs:
            ui.show_error(Error.NO_DOCUMENTS)
            raise ExitApp()
        try:
            ui.show_info_message("Creating vector store.")
            rag_preprocessor.create_vector_store(docs)
        except FaissException as exception:
            ui.show_error(Error.EXCEPTION, exception)
            raise ExitApp()
        except VectorStoreException as exception:
            ui.show_error(Error.EXCEPTION, exception)
            raise ExitApp()
        ui.show_info_message("✅ Vector DB created and saved.")
    else:
        ui.show_info_message("📦 Using existing vector store.")

    # Load vector store, retriever and memory
    try:
        ui.show_info_message("📶 Loading vector store.")
        vectordb = rag_preprocessor.load_vector_store()
    except Exception as exception:
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()

    run_chat_loop(ui, vectordb)


def run_chat_loop(ui: ConsoleUI, vectordb: FAISS) -> None:
    while True:
        ui.show_operational_mode_selection()
        user_selection = ui.get_operational_mode_selection()

        if user_selection.lower() in EXIT_WORDS:
            raise ExitApp()

        if user_selection == "1":
            ui.show_entering_mode(ChatbotMode.DOMAIN_EXPERT)
            domain_expert(ui, vectordb)
            continue
        if user_selection == "2":
            ui.show_entering_mode(ChatbotMode.EXAM_PREP)
            exam_prep(ui, vectordb)
            continue
        else:
            ui.show_error(Error.INVALID_MODE)


def main() -> None:
    ui = ConsoleUI()
    ui.show_welcome()
    try:
        run_app(ui)
    except ExitApp:
        ui.show_exit_message()
        sys.exit(0)


if __name__ == "__main__":
    main()
