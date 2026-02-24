import pytest
from src.core.chain_manager import ChainManager
from langchain_community.vectorstores import Chroma
from unittest.mock import Mock, patch
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from src.core.prompts import domain_expert_prompt, domain_expert_condense_prompt


class TestChainManager:
    @pytest.fixture
    def mock_vectordb(self):
        """Create a mock Chroma vector database."""
        mock_db = Mock(spec=Chroma)
        mock_retriever = Mock()
        mock_db.as_retriever.return_value = mock_retriever
        return mock_db

    @pytest.fixture
    @patch("src.core.chain_manager.LLM_PROVIDER", "together")
    @patch("src.core.chain_manager.TOGETHER_API_KEY", "test-api-key")
    def chain_manager(self, mock_vectordb):
        """Create a ChainManager instance with mocked dependencies."""
        return ChainManager(mock_vectordb)

    def test_init_together_with_valid_inputs(self, mock_vectordb):
        chain_manager = ChainManager(
            mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1
        )
        assert chain_manager.temperature == 0.7
        assert chain_manager.max_tokens == 256
        mock_vectordb.as_retriever.assert_called_once()

    @patch("src.core.chain_manager.LLM_PROVIDER", "ollama")
    @patch("src.core.chain_manager.OLLAMA_BASE_URL", "http://localhost:11434")
    def test_init_ollama_with_valid_inputs(self, mock_vectordb):
        chain_manager = ChainManager(
            mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1
        )
        assert chain_manager.temperature == 0.7
        assert chain_manager.max_tokens == 256
        mock_vectordb.as_retriever.assert_called_once()

    def test_init_without_vectordb(self):
        with pytest.raises(ValueError, match="vectordb cannot be None"):
            ChainManager(None)

    @patch("src.core.chain_manager.LLM_PROVIDER", "together")
    @patch("src.core.chain_manager.TOGETHER_API_KEY", None)
    def test_init_missing_together_api_key(self, mock_vectordb):
        with pytest.raises(
            ValueError, match="TOGETHER_API_KEY environment variable is required"
        ):
            ChainManager(mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1)

    @patch("src.core.chain_manager.LLM_PROVIDER", "ollama")
    @patch("src.core.chain_manager.TOGETHER_API_KEY", None)
    @patch("src.core.chain_manager.OLLAMA_BASE_URL", "http://localhost:11434")
    def test_init_missing_together_api_key_using_ollama(self, mock_vectordb):
        ChainManager(mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1)

    @patch("src.core.chain_manager.LLM_PROVIDER", "ollama")
    @patch("src.core.chain_manager.OLLAMA_BASE_URL", None)
    def test_init_missing_ollama_base_url(self, mock_vectordb):
        with pytest.raises(
            ValueError, match="OLLAMA_BASE_URL environment variable is required"
        ):
            ChainManager(mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1)

    @patch("src.core.chain_manager.LLM_PROVIDER", "invalid")
    def test_init_invalid_llm_provider(self, mock_vectordb):
        with pytest.raises(
            ValueError,
            match=("LLM_PROVIDER environment variable must be together or ollama"),
        ):
            ChainManager(mock_vectordb, temperature=0.7, max_tokens=256, retrieval_k=1)

    @patch("src.core.chain_manager.Together")
    def test_get_llm_success_together(self, mock_together, chain_manager):
        # Arrange
        mock_llm = Mock(spec=LLM)
        mock_together.return_value = mock_llm

        # Act
        result = chain_manager.get_llm()

        # Assert
        assert result == mock_llm
        mock_together.assert_called_once_with(
            model=chain_manager.model,
            together_api_key=chain_manager.together_api_key,
            temperature=chain_manager.temperature,
            max_tokens=chain_manager.max_tokens,
        )

    @patch("src.core.chain_manager.LLM_PROVIDER", "ollama")
    @patch("src.core.chain_manager.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.core.chain_manager.Ollama")
    def test_get_llm_success_ollama(self, mock_ollama, mock_vectordb):
        chain_manager = ChainManager(mock_vectordb)
        # Arrange
        mock_llm = Mock(spec=LLM)
        mock_ollama.return_value = mock_llm

        # Act
        result = chain_manager.get_llm()

        # Assert
        assert result == mock_llm
        mock_ollama.assert_called_once_with(
            model=chain_manager.model,
            base_url=chain_manager.base_url,
            temperature=chain_manager.temperature,
            num_predict=chain_manager.max_tokens,
        )

    @patch("src.core.chain_manager.Together")
    def test_get_llm_failure(self, mock_together, chain_manager):
        mock_together.side_effect = Exception("API connection failed")

        with pytest.raises(Exception, match="Error setting up Together AI LLM"):
            chain_manager.get_llm()

    @patch("src.core.chain_manager.ConversationBufferMemory")
    def test_get_conversationalRetrievalChain_exception_ConversationBufferMemory(
        self, mock_conversation_buffer_memory, chain_manager
    ):
        # Arrange
        mock_conversation_buffer_memory.side_effect = Exception(
            "Error in ConversationBufferMemory"
        )
        mock_llm = Mock(spec=LLM)

        # Act
        with pytest.raises(Exception, match="Error in ConversationBufferMemory"):
            chain_manager.get_conversationalRetrievalChain(
                mock_llm, {"sample_dict": "sample"}
            )

    def test_get_conversationalRetrievalChain_exception_mock_llm_and_vectordb(
        self, chain_manager
    ):
        with pytest.raises(
            Exception,
            match="Input should be a valid dictionary or instance of BaseRetriever",
        ):
            chain_manager.get_conversationalRetrievalChain(
                chain_manager.get_llm(),
                {"prompt": domain_expert_prompt},
                condense_question_prompt=domain_expert_condense_prompt,
            )

    def test_ask_question_success(self, chain_manager):
        # Arrange
        question = "This is the question"
        mock_chain = Mock()
        mock_response = {"answer": "This is the answer"}
        mock_chain.invoke.return_value = mock_response

        # Act
        answer = chain_manager.ask_question(question, mock_chain)

        # Assert
        assert answer == "This is the answer"

    def test_ask_question_retrieval_qa(self, chain_manager):
        # Arrange
        question = "This is the question"
        mock_chain = Mock(spec=RetrievalQA)
        mock_chain.invoke.return_value = {"result": "This is the answer"}

        # Act
        answer = chain_manager.ask_question(question, mock_chain)

        # Assert
        assert answer == "This is the answer"
        mock_chain.invoke.assert_called_once_with({"query": question})

    def test_ask_question_failure(self, chain_manager):
        # Arrange
        question = "This is the question"
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Exception getting answer")

        # Act
        with pytest.raises(Exception, match="Error invoking LLM:"):
            chain_manager.ask_question(question, mock_chain)
