from src.core.constants import EXIT_WORDS, ChatbotMode, Error
from src.ui.console_ui import ConsoleUI
from src.core.exceptions import ExitApp
from src.core.domain_expert_core import DomainExpertCore
import logging

logger = logging.getLogger(__name__)


def domain_expert_ui(ui: ConsoleUI, domain_expert: DomainExpertCore) -> None:   
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
                answer = domain_expert.ask_question(question)
                ui.show_answer(answer)
            except Exception as exception:
                logger.error(f"Error retrieving answer: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                continue

    except KeyboardInterrupt:
        raise ExitApp()
