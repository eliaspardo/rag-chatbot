import pytest
from unittest.mock import Mock
from src.core.exam_prep_core import ExamPrepCore
from src.ui.console_ui import ConsoleUI
from src.ui.exam_prep_ui import run_exam_prep_chat_loop
from src.core.exceptions import ExitApp
from src.core.constants import Error, ChatbotMode, EXIT_WORDS


class TestExamPrepUi:
    @pytest.fixture
    def mock_console_ui(self):
        return Mock(spec=ConsoleUI)

    @pytest.fixture
    def mock_exam_prep_core(self):
        return Mock(spec=ExamPrepCore)

    def test_run_chat_loop_exit_words(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui.get_user_input.return_value = exit_word

            # Act
            with pytest.raises(ExitApp):
                run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(
                ChatbotMode.EXAM_PREP
            )

    def test_run_chat_loop_mode_switch(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        mock_console_ui.get_user_input.return_value = "mode"

        # Act
        run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_not_a_topic(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        mock_console_ui.get_user_input.side_effect = [""]

        # Act
        with pytest.raises(StopIteration):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NOT_A_TOPIC)

    def test_run_chat_loop_question_error(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "quit"]
        mock_exam_prep_core.get_question.side_effect = Exception(
            "Error getting question"
        )

        # Act
        with pytest.raises(ExitApp):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_error.assert_called_with(
            Error.EXCEPTION, exception=mock_exam_prep_core.get_question.side_effect
        )
        mock_console_ui.show_info_message.assert_any_call(
            "Please try rephrasing your question."
        )

    def test_run_chat_loop_exit_words_on_second_input(
        self, mock_console_ui, mock_exam_prep_core
    ):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui.get_user_input.side_effect = ["Sample Topic", exit_word]
            mock_exam_prep_core.get_question.return_value = "Sample Question"

            # Act
            with pytest.raises(ExitApp):
                run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(
                ChatbotMode.EXAM_PREP
            )

    def test_run_chat_loop_mode_switch_on_second_input(
        self, mock_console_ui, mock_exam_prep_core
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "mode"]
        mock_exam_prep_core.get_question.return_value = "Sample Question"

        # Act
        run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_no_user_answer(
        self, mock_console_ui, mock_exam_prep_core
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", ""]
        mock_exam_prep_core.get_question.return_value = "Sample Question"

        # Act
        with pytest.raises(StopIteration):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NO_USER_ANSWER)

    def test_run_chat_loop_answer_error(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        exception = Exception("Error getting answer")
        mock_console_ui.get_user_input.side_effect = [
            "Sample Topic",
            "Sample Answer",
            "quit",
        ]
        mock_exam_prep_core.get_question.return_value = "Sample Question"
        mock_exam_prep_core.get_answer.side_effect = exception

        # Act
        with pytest.raises(ExitApp):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_error.assert_called_with(
            Error.EXCEPTION, exception=exception
        )
        mock_console_ui.show_info_message.assert_any_call(
            "Please try rephrasing your answer."
        )

    def test_run_chat_loop_keyboard_exception(
        self, mock_console_ui, mock_exam_prep_core
    ):
        # Arrange
        exception = KeyboardInterrupt()
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", exception]
        mock_exam_prep_core.get_question.return_value = "Sample Question"

        # Act
        with pytest.raises(ExitApp):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

    def test_run_chat_loop_success(self, mock_console_ui, mock_exam_prep_core):
        # Arrange
        mock_console_ui.get_user_input.side_effect = [
            "Sample Topic",
            "Sample Answer",
            "quit",
        ]
        mock_exam_prep_core.get_question.return_value = "Sample Question"
        mock_exam_prep_core.get_answer.return_value = "Sample Answer"

        # Act
        with pytest.raises(ExitApp):
            run_exam_prep_chat_loop(mock_console_ui, mock_exam_prep_core)

        # Assert
        mock_console_ui.show_answer.assert_called_with(
            mock_exam_prep_core.get_answer.return_value
        )
