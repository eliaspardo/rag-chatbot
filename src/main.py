# Project: AI-Powered Chatbot
# Tools: Together AI (LLM), FAISS (Vector DB),
# Sentence Transformers (Embeddings), LangChain

import sys
from src.core.domain_expert_core import DomainExpertCore
from src.core.exam_prep_core import ExamPrepCore
from src.ui.domain_expert_ui import run_domain_expert_chat_loop
from src.ui.console_ui import ConsoleUI
from src.core.constants import ChatbotMode, EXIT_WORDS, Error
from src.core.rag_preprocessor import RAGPreprocessor
from src.core.exceptions import (
    ExitApp,
    FaissException,
    NoDocumentsException,
    VectorStoreException,
)
import logging
from langchain_community.vectorstores import FAISS
from src.env_loader import load_environment
from src.core.app_bootstrap import prepare_vector_store
from src.ui.exam_prep_ui import run_exam_prep_chat_loop


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_environment()


def run_app(ui: ConsoleUI) -> None:
    rag_preprocessor = RAGPreprocessor()
    try:
        vectordb = prepare_vector_store(
            rag_preprocessor=rag_preprocessor,
            progress_callback=ui.show_info_message,
        )
    except NoDocumentsException:
        ui.show_error(Error.NO_DOCUMENTS)
        raise ExitApp()
    except (FaissException, VectorStoreException) as exception:
        ui.show_error(Error.EXCEPTION, exception)
        raise ExitApp()
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
            try:
                domain_expert = DomainExpertCore(vectordb)
            except Exception as exception:
                logger.error(f"Error setting up Domain Expert: {exception}")
                ui.show_error(Error.EXCEPTION, exception)
                raise ExitApp()
            run_domain_expert_chat_loop(ui, domain_expert)
            continue
        if user_selection == "2":
            ui.show_entering_mode(ChatbotMode.EXAM_PREP)
            try:
                exam_prep = ExamPrepCore(vectordb)
            except Exception as exception:
                logger.error(f"Error setting up Exam Prep: {exception}")
                ui.show_error(Error.EXCEPTION, exception)
                raise ExitApp()
            run_exam_prep_chat_loop(ui, exam_prep)
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
