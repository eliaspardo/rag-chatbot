import pytest
from unittest.mock import Mock, patch
from src.core.rag_preprocessor import RAGPreprocessor
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
import shutil


TEST_PDF = "tests/data/pdf-test.pdf"
STRING_LIST = ["test", "string", "for", "testing"]
EMPTY_LIST = []
STRING_LIST_EMPTY_CHUNKS = ["test", "    ", "for", "   testing   "]
PAGE_CONTENT = "Test content"
TEST_DB_DIR = "tests/faiss_db"
TEST_PREEXISTING_DB_DIR = "tests/data/test_faiss_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TestRagPreprocessor:
    @pytest.fixture
    def rag_preprocessor(self):
        return RAGPreprocessor()

    def test_load_pdf_text_no_file(self, rag_preprocessor):
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text("no_file")

    @patch("src.core.rag_preprocessor.fitz.open")
    def test_load_pdf_text_default_no_file(self, mock_fitz_open, rag_preprocessor):
        mock_fitz_open.side_effect = FileNotFoundError("No such file or directory")
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text()

    def test_load_pdf_text_successful(self, rag_preprocessor):
        texts = rag_preprocessor.load_pdf_text(TEST_PDF)
        full_text = "".join(texts)
        assert "Congratulations" in full_text

    def test_split_text_to_docs_success(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(STRING_LIST)
        assert len(documents) > 0
        for document in documents:
            print("document")
            assert isinstance(document, Document)

    def test_split_text_to_docs_empty_list(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(EMPTY_LIST)
        assert len(documents) == 0

    def test_split_text_to_docs_empty_chunks(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(STRING_LIST_EMPTY_CHUNKS)
        assert all(document.page_content.strip() for document in documents)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    def test_create_vector_store_deletes_existing_storage(
        self, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing by mocking and generate exception so folder is not created
        mock_huggingFaceEmbeddings.side_effect = Exception("Error creating embeddings")
        # Create folder
        os.mkdir(TEST_DB_DIR)
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        # Ignore errors when creating embedding as it was mocked
        with pytest.raises(Exception, match="Error creating embeddings"):
            rag_preprocessor.create_vector_store(docs=documents, db_dir=TEST_DB_DIR)

        # Assert
        # We expect the folder not to exist as it was cleaned up
        assert not os.path.isdir(TEST_DB_DIR)

        # Clean up folder, just in case
        if os.path.exists(TEST_DB_DIR):
            shutil.rmtree(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    def test_create_vector_store_throws_exception_mocked_HuggingFaceEmbeddings(
        self, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        mock_huggingFaceEmbeddings.side_effect = Exception("Error creating embeddings")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        with pytest.raises(Exception, match="Error creating embeddings"):
            rag_preprocessor.create_vector_store(docs=documents, db_dir=TEST_DB_DIR)

        # Assert
        assert not os.path.isdir(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.FAISS")
    def test_create_vector_store_throws_value_error_mocked_FAISS_from_documents(
        self, mock_faiss, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing
        mock_huggingFaceEmbeddings.return_value = None

        mock_faiss.from_documents.side_effect = ValueError("Wrong Documents")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        with pytest.raises(Exception, match="Wrong Documents"):
            rag_preprocessor.create_vector_store(docs=documents, db_dir=TEST_DB_DIR)

        # Assert
        assert not os.path.isdir(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.FAISS")
    def test_create_vector_store_throws_runtime_error_mocked_FAISS_from_documents(
        self, mock_faiss, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing
        mock_huggingFaceEmbeddings.return_value = None

        mock_faiss.from_documents.side_effect = RuntimeError("Runtime error")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        with pytest.raises(Exception, match="Runtime error"):
            rag_preprocessor.create_vector_store(docs=documents, db_dir=TEST_DB_DIR)

        # Assert
        assert not os.path.isdir(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.FAISS")
    def test_create_vector_store_throws_exception_mocked_FAISS_save_local(
        self, mock_faiss, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing
        mock_huggingFaceEmbeddings.return_value = None
        mock_vectordb_instance = Mock(spec=FAISS)
        mock_faiss.from_documents.return_value = mock_vectordb_instance

        mock_vectordb_instance.save_local.side_effect = Exception("Error writing")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        with pytest.raises(Exception, match="Error writing"):
            rag_preprocessor.create_vector_store(docs=documents, db_dir=TEST_DB_DIR)

        # Assert
        assert not os.path.isdir(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.FAISS")
    def test_create_vector_store_success(
        self, mock_faiss, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing
        mock_embeddings = Mock()
        mock_huggingFaceEmbeddings.return_value = mock_embeddings
        mock_vectordb_instance = Mock(spec=FAISS)
        mock_faiss.from_documents.return_value = mock_vectordb_instance
        mock_vectordb_instance.save_local.return_value = None
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        vectordb = rag_preprocessor.create_vector_store(
            docs=documents, db_dir=TEST_DB_DIR
        )

        # Assert
        mock_huggingFaceEmbeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_faiss.from_documents.assert_called_once_with(documents, mock_embeddings)
        mock_vectordb_instance.save_local.assert_called_once_with(TEST_DB_DIR)
        assert vectordb is mock_vectordb_instance

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    def test_load_vector_store_does_not_exist(
        self, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        # Speed up testing
        mock_huggingFaceEmbeddings.return_value = None

        # Act
        with pytest.raises(Exception, match="No such file or directory"):
            rag_preprocessor.load_vector_store(TEST_DB_DIR)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.FAISS")
    def test_load_vector_store_success(
        self, mock_faiss, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        mock_huggingFaceEmbeddings.return_value = Mock()
        mock_faiss.load_local.return_value = Mock(spec=FAISS)

        # Act
        rag_preprocessor.load_vector_store(TEST_PREEXISTING_DB_DIR)

        # Assert
        mock_huggingFaceEmbeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_faiss.load_local.assert_called_once_with(
            TEST_PREEXISTING_DB_DIR,
            mock_huggingFaceEmbeddings.return_value,
            allow_dangerous_deserialization=True,
        )
