# Project: AI-Powered Chatbot
# Tools: Together AI (LLM), FAISS (Vector DB),
# Sentence Transformers (Embeddings), LangChain

import sys
import os
from dotenv import load_dotenv
from console_ui import ConsoleUI
from domain_expert import domain_expert
from exam_prep import exam_prep
from constants import ChatbotMode, EXIT_WORDS, Error
from rag_preprocessor import RAGPreprocessor
from exceptions import ExitApp, FaissException, VectorStoreException


# Load environment variables
load_dotenv()

# --- CONFIGURATION FROM .ENV ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DB_DIR = os.getenv("DB_DIR", "faiss_db")


def run_app(ui: ConsoleUI):

    rag_preprocessor = RAGPreprocessor()
    # Process PDF if not already embedded
    if not os.path.exists(DB_DIR):
        ui.show_info_message("\n🔍 Loading PDF...")
        texts = rag_preprocessor.load_pdf_text()

        ui.show_info_message("Splitting text to docs.")
        docs = rag_preprocessor.split_text_to_docs(texts)

        ui.show_info_message("Creating vector store.")
        if not docs:
            ui.show_error(Error.NO_DOCUMENTS)
            raise ExitApp()
        try:
            rag_preprocessor.create_vector_store(docs)
        except FaissException as e:
            ui.show_error(Error.FAISS_EXCEPTION, e)
        except VectorStoreException as e:
            ui.show_error(Error.VECTOR_EXCEPTION, e)
        ui.show_info_message("✅ Vector DB created and saved.")
    else:
        ui.show_info_message("📦 Using existing vector store.")

    # Load vector store, retriever and memory
    ui.show_info_message("📶 Loading vector store.")
    vectordb = rag_preprocessor.load_vector_store()

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


def main():
    ui = ConsoleUI()
    ui.show_welcome()
    try:
        result = run_app(ui)
    except ExitApp:
        ui.show_exit_message()
        sys.exit(0)
    if result == "exit":
        ui.show_exit_message()
        sys.exit(0)


if __name__ == "__main__":
    main()
