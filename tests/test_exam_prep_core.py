import pytest
from unittest.mock import Mock, patch, call
from langchain_core.vectorstores import VectorStore
from src.core.exam_prep_core import ExamPrepCore
from src.core.chain_manager import ChainManager
from src.core.exceptions import ExamPrepQueryException, ExamPrepSetupException
from src.core.prompts import exam_prep_question_prompt, exam_prep_answer_prompt


class TestExamPrepCore:
    @pytest.fixture
    def mock_vectordb(self):
        return Mock(spec=VectorStore)

    @pytest.fixture
    def mock_chain_manager(self):
        return Mock(spec=ChainManager)

    def _build_core(self, mock_chain_manager_class, mock_chain_manager, mock_vectordb):
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_llm = Mock()
        mock_chain_manager.get_llm.return_value = mock_llm
        question_chain = Mock()
        answer_chain = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.side_effect = [
            question_chain,
            answer_chain,
        ]
        core = ExamPrepCore(mock_vectordb)
        return core, question_chain, answer_chain

    @patch("src.core.exam_prep_core.ChainManager")
    def test_exam_prep_init_success(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Act
        core, question_chain, answer_chain = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )

        # Assert
        mock_chain_manager_class.assert_called_once_with(mock_vectordb)
        mock_chain_manager.get_llm.assert_called_once()
        mock_chain_manager.get_conversationalRetrievalChain.assert_has_calls(
            [
                call(
                    mock_chain_manager.get_llm.return_value,
                    {"prompt": exam_prep_question_prompt},
                ),
                call(
                    mock_chain_manager.get_llm.return_value,
                    {"prompt": exam_prep_answer_prompt},
                ),
            ]
        )
        assert core.chain_manager == mock_chain_manager
        assert core.question_chain == question_chain
        assert core.answer_chain == answer_chain

    @patch("src.core.exam_prep_core.ChainManager")
    def test_exam_prep_init_chain_manager_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
    ):
        # Arrange
        mock_chain_manager_class.side_effect = ValueError(
            "Error instantiating Chain Manager"
        )

        # Act
        with pytest.raises(ExamPrepSetupException):
            ExamPrepCore(mock_vectordb)

    @patch("src.core.exam_prep_core.ChainManager")
    def test_exam_prep_get_llm_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.side_effect = Exception("Error getting LLM")

        # Act
        with pytest.raises(ExamPrepSetupException):
            ExamPrepCore(mock_vectordb)

    @patch("src.core.exam_prep_core.ChainManager")
    def test_exam_prep_get_conversational_chain_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.return_value = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.side_effect = Exception(
            "Error getting ConversationalRetrievalChain"
        )

        # Act
        with pytest.raises(ExamPrepSetupException):
            ExamPrepCore(mock_vectordb)

    @patch("src.core.exam_prep_core.ChainManager")
    def test_get_question_success(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core, _, _ = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.return_value = "Sample question"

        # Act
        question = core.get_question("Sample topic")

        # Assert
        assert question == "Sample question"
        mock_chain_manager.ask_question.assert_called_once_with(
            "Sample topic", core.question_chain
        )

    @patch("src.core.exam_prep_core.ChainManager")
    def test_get_question_failure(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core, _, _ = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.side_effect = Exception(
            "Error retrieving question"
        )

        # Act
        with pytest.raises(ExamPrepQueryException):
            core.get_question("Sample topic")

    @patch("src.core.exam_prep_core.ChainManager")
    def test_get_answer_success(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core, _, _ = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.return_value = "Sample answer"

        # Act
        answer = core.get_feedback("Sample question", "Sample answer")

        # Assert
        assert answer == "Sample answer"
        mock_chain_manager.ask_question.assert_called_once_with(
            "Sample question\nSample answer", core.answer_chain
        )

    @patch("src.core.exam_prep_core.ChainManager")
    def test_get_answer_failure(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core, _, _ = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.side_effect = Exception(
            "Error retrieving answer"
        )

        # Act
        with pytest.raises(ExamPrepQueryException):
            core.get_feedback("Sample question", "Sample answer")
