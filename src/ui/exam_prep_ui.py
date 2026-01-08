from src.core.constants import EXIT_WORDS, ChatbotMode, Error
from src.ui.console_ui import ConsoleUI
from src.core.exceptions import ExitApp
from src.core.exam_prep_core import ExamPrepCore
import logging

logger = logging.getLogger(__name__)


def run_exam_prep_chat_loop(ui: ConsoleUI, exam_prep: ExamPrepCore) -> None:
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
                llm_question = exam_prep.get_question(topic)
                ui.show_llm_question(llm_question)
            except Exception as exception:
                logger.error(f"Error retrieving question: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                ui.show_info_message("Please try rephrasing your question.")
                continue
            user_answer = ui.get_user_input("\n📝 Your answer: ")

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
                llm_answer = exam_prep.get_answer(llm_question_user_answer)
                ui.show_answer(llm_answer)
            except Exception as exception:
                logger.error(f"Error retrieving answer: {exception}")
                ui.show_error(Error.EXCEPTION, exception=exception)
                ui.show_info_message("Please try rephrasing your answer.")
                continue

    except KeyboardInterrupt:
        raise ExitApp()
