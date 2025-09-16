import pytest
from unittest.mock import Mock, patch
from src.exceptions import ExitApp, FaissException, VectorStoreException
from src.constants import Error, EXIT_WORDS, ChatbotMode
from src.console_ui import ConsoleUI
from src.main import main, run_app, run_chat_loop
from src.rag_preprocessor import RAGPreprocessor


class TestMain:

    @pytest.fixture
    def mock_console_ui_instance(self):
        return Mock(spec=ConsoleUI)

    @pytest.fixture
    def mock_vectordb_instance(self):
        return Mock()

    @patch("src.main.run_app")
    @patch("src.main.ConsoleUI")
    def test_main_shows_welcome(self, mock_console_ui, mock_run_app):
        # Arrange
        mock_run_app.side_effect = None
        # Act
        main()

        # Assert
        mock_console_ui.return_value.show_welcome.assert_called_once()

    @patch("sys.exit")
    @patch("src.main.run_app")
    @patch("src.main.ConsoleUI")
    def test_main_exitApp_error(self, mock_console_ui, mock_run_app, mock_sys_exit):
        # Arrange
        mock_run_app.side_effect = ExitApp()
        mock_sys_exit.side_effect = None
        # Act
        main()

        # Assert
        mock_console_ui.return_value.show_exit_message.assert_called_once()

    @patch("src.main.run_chat_loop")
    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=True)
    def test_run_app_path_exists_uses_existing_vector_store_success(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_run_chat_loop,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.load_vector_store.return_value = None
        mock_run_chat_loop.return_value = None

        # Act
        run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_info_message.assert_any_call(
            "📦 Using existing vector store."
        )
        mock_console_ui_instance.show_info_message.assert_any_call(
            ("📶 Loading vector store.")
        )
        mock_run_chat_loop.assert_called_once()

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=True)
    def test_run_app_path_exists_error_loading_store(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_vector_store.side_effect = Exception(
            "Error loading vector store"
        )

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_with(
            Error.EXCEPTION,
            mock_rag_preprocessor.return_value.load_vector_store.side_effect,
        )

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=False)
    def test_run_app_path_not_exists_error_loading_pdf_text(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_pdf_text.side_effect = Exception(
            "Error loading pdf"
        )

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_with(
            Error.EXCEPTION,
            mock_rag_preprocessor.return_value.load_pdf_text.side_effect,
        )

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=False)
    def test_run_app_path_not_exists_error_splitting_text(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.split_text_to_docs.side_effect = Exception(
            "Error splitting text"
        )

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_with(
            Error.EXCEPTION,
            mock_rag_preprocessor.return_value.split_text_to_docs.side_effect,
        )

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=False)
    def test_run_app_path_not_exists_error_no_docs(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_pdf_text.return_value = []
        mock_rag_preprocessor.return_value.split_text_to_docs.return_value = []

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_with(Error.NO_DOCUMENTS)

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=False)
    def test_run_app_path_not_exists_error_create_vector_store(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_pdf_text.return_value = []
        mock_rag_preprocessor.return_value.split_text_to_docs.return_value = [
            "Sample Text"
        ]
        exceptions = [FaissException(), VectorStoreException()]
        for exception in exceptions:
            mock_rag_preprocessor.return_value.create_vector_store.side_effect = (
                exception
            )

            # Act
            with pytest.raises(ExitApp):
                run_app(mock_console_ui_instance)

            # Assert
            mock_console_ui_instance.show_info_message.assert_called_with(
                "Creating vector store."
            )
            mock_console_ui_instance.show_error.assert_called_with(
                Error.EXCEPTION,
                mock_rag_preprocessor.return_value.create_vector_store.side_effect,
            )

    @patch("src.main.run_chat_loop")
    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=False)
    def test_run_app_path_not_exists_success(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_run_chat_loop,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_pdf_text.return_value = []
        mock_rag_preprocessor.return_value.split_text_to_docs.return_value = [
            "Sample Text"
        ]
        mock_rag_preprocessor.return_value.create_vector_store.return_value = Mock()
        mock_rag_preprocessor.return_value.load_vector_store.return_value = Mock()
        mock_run_chat_loop.return_value = None

        # Act
        run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_info_message.assert_any_call(
            "✅ Vector DB created and saved."
        )

    def test_run_chat_loop_exit_words(
        self, mock_console_ui_instance, mock_vectordb_instance
    ):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui_instance.get_operational_mode_selection.return_value = (
                exit_word
            )

            # Act
            with pytest.raises(ExitApp):
                run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

            # Assert
            mock_console_ui_instance.show_operational_mode_selection.assert_called()

    @patch("src.main.domain_expert")
    @patch("src.main.exam_prep")
    def test_run_chat_loop_select_modes(
        self,
        mock_exam_prep,
        mock_domain_expert,
        mock_console_ui_instance,
        mock_vectordb_instance,
    ):
        # Arrange
        modes = [ChatbotMode.DOMAIN_EXPERT, ChatbotMode.EXAM_PREP]
        mock_exam_prep.return_value = None
        mock_domain_expert.return_value = None
        mock_console_ui_instance.get_operational_mode_selection.side_effect = ["1", "2"]
        with pytest.raises(StopIteration):

            for mode in modes:

                # Act
                run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

                # Assert
                mock_console_ui_instance.show_entering_mode.assert_called_with(mode)
            mock_exam_prep.assert_called_once()
            mock_domain_expert.assert_called_once()

    def test_run_chat_loop_invalid_input(
        self,
        mock_console_ui_instance,
        mock_vectordb_instance,
    ):
        # Arrange
        mock_console_ui_instance.get_operational_mode_selection.side_effect = [
            "Invalid input"
        ]
        with pytest.raises(StopIteration):
            # Act
            run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

            # Assert
            mock_console_ui_instance.show_error.assert_called_with(Error.INVALID_MODE)
