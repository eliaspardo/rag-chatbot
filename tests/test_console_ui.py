import pytest
from src.console_ui import ConsoleUI, Error
from constants import ChatbotMode, DEFAULT_CONSOLE_WIDTH
from unittest.mock import patch


class TestConsoleUI:

    @pytest.fixture
    def ui(self):
        return ConsoleUI()

    # @patch("os.get_terminal_size().columns", return_value=80)
    def test_show_welcome(self, ui, capsys):
        ui.show_welcome()
        captured = capsys.readouterr()
        assert "Starting RAG Chatbot..." in captured.out
        assert "=" * DEFAULT_CONSOLE_WIDTH in captured.out

    def test_show_operational_mode_selection(self, ui, capsys):
        ui.show_operational_mode_selection()
        captured = capsys.readouterr()
        assert "Select an Operational Mode:" in captured.out
        assert "Type 'quit', 'exit', or 'no' to stop." in captured.out

    @patch("builtins.input", return_value="  test input  ")
    def test_get_operational_mode_selection_with_whitespaces(self, mock_input, ui):
        returned = ui.get_operational_mode_selection()
        assert returned == "test input"

    def test_show_entering_mode(self, ui, capsys):
        ui.show_entering_mode(ChatbotMode.DOMAIN_EXPERT)
        captured = capsys.readouterr()
        assert "Entering Domain Expert Chatbot mode..." in captured.out

    def test_show_welcome_mode_domain_expert(self, ui, capsys):
        ui.show_welcome_mode(ChatbotMode.DOMAIN_EXPERT)
        captured = capsys.readouterr()
        assert "Domain Expert Mode Ready!" in captured.out
        assert "Ask me anything about your document." in captured.out

    def test_show_welcome_mode_exam_prep(self, ui, capsys):
        ui.show_welcome_mode(ChatbotMode.EXAM_PREP)
        captured = capsys.readouterr()
        assert "Exam Prep Mode Ready!" in captured.out
        assert "What section or topic you want me to quiz you on?" in captured.out

    def test_show_mode_switch(self, ui, capsys):
        ui.show_mode_switch()
        captured = capsys.readouterr()
        assert "Returning to Operational Mode selection" in captured.out

    @patch("builtins.input", return_value="  test input  ")
    def test_get_input_with_whitespaces(self, mock_input, ui):
        user_input = ui.get_user_input("  test input  ")
        assert "test input" == user_input

    def test_show_info_message(self, ui, capsys):
        ui.show_info_message("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_show_answer(self, ui, capsys):
        ui.show_answer("Test answer")
        captured = capsys.readouterr()
        assert "Answer" in captured.out
        assert "=" * DEFAULT_CONSOLE_WIDTH in captured.out
        assert "Test answer" in captured.out

    def test_show_llm_question(self, ui, capsys):
        ui.show_llm_question("Test question")
        captured = capsys.readouterr()
        assert "Question:" in captured.out
        assert "=" * DEFAULT_CONSOLE_WIDTH in captured.out
        assert "Test question" in captured.out

    def test_show_error_no_documents(self, ui, capsys):
        ui.show_error(Error.NO_DOCUMENTS)
        captured = capsys.readouterr()
        assert ("No documents found after splitting — aborting.") in captured.out

    def test_show_error_invalid_mode(self, ui, capsys):
        ui.show_error(Error.INVALID_MODE)
        captured = capsys.readouterr()
        assert ("Please select a valid Operational Mode!") in captured.out

    def test_show_error_not_a_question(self, ui, capsys):
        ui.show_error(Error.NOT_A_QUESTION)
        captured = capsys.readouterr()
        assert ("Please enter a question.") in captured.out

    def test_show_error_not_a_topic(self, ui, capsys):
        ui.show_error(Error.NOT_A_TOPIC)
        captured = capsys.readouterr()
        assert ("Please enter a section / topic.") in captured.out

    def test_show_error_no_user_answer(self, ui, capsys):
        ui.show_error(Error.NO_USER_ANSWER)
        captured = capsys.readouterr()
        assert ("Let's start all over again.") in captured.out

    def test_show_error_exception(self, ui, capsys):
        ui.show_error(Error.EXCEPTION, Exception("Test Exception!"))
        captured = capsys.readouterr()
        assert ("Test Exception!") in captured.out

    def test_show_exit_message(self, ui, capsys):
        ui.show_exit_message()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out
