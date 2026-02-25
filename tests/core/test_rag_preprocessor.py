import pytest
from unittest.mock import Mock, patch
from src.core import rag_preprocessor as rag_preprocessor_module
from src.core.rag_preprocessor import (
    DoclingRAGPreprocessor,
    LegacyRAGPreprocessor,
    RAGPreprocessor,
)
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

TEST_PDF = "tests/data/pdf-test.pdf"
STRING_LIST = ["test", "string", "for", "testing"]
EMPTY_LIST = []
STRING_LIST_EMPTY_CHUNKS = ["test", "    ", "for", "   testing   "]
PAGE_CONTENT = "Test content"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "rag_documents"


class TestRagPreprocessor:
    @pytest.fixture
    def mock_chroma_client(self):
        with patch("src.core.rag_preprocessor.chromadb.HttpClient") as mock:
            yield mock.return_value

    @pytest.fixture
    def rag_preprocessor(self, mock_chroma_client):
        return LegacyRAGPreprocessor()

    def test_base_load_pdf_text_not_implemented(self, mock_chroma_client):
        with pytest.raises(NotImplementedError):
            RAGPreprocessor().load_pdf_text(TEST_PDF)

    def test_base_split_text_to_docs_not_implemented(self, mock_chroma_client):
        with pytest.raises(NotImplementedError):
            RAGPreprocessor().split_text_to_docs([Document(page_content="test")])

    def test_load_pdf_text_no_file(self, rag_preprocessor):
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text("no_file")

    @patch("src.core.rag_preprocessor.fitz.open")
    def test_load_pdf_text_default_no_file(self, mock_fitz_open, rag_preprocessor):
        mock_fitz_open.side_effect = FileNotFoundError("No such file or directory")
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text(TEST_PDF)

    def test_load_pdf_text_successful(self, rag_preprocessor):
        docs = rag_preprocessor.load_pdf_text(TEST_PDF)
        full_text = "".join(doc.page_content for doc in docs)
        assert "Congratulations" in full_text

    def test_split_text_to_docs_success(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(
            [Document(page_content=item) for item in STRING_LIST]
        )
        assert len(documents) > 0
        for document in documents:
            print("document")
            assert isinstance(document, Document)

    def test_split_text_to_docs_empty_list(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(EMPTY_LIST)
        assert len(documents) == 0

    def test_split_text_to_docs_empty_chunks(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(
            [Document(page_content=item) for item in STRING_LIST_EMPTY_CHUNKS]
        )
        assert all(document.page_content.strip() for document in documents)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    def test_create_vector_store_throws_exception_mocked_HuggingFaceEmbeddings(
        self, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        mock_huggingFaceEmbeddings.side_effect = Exception("Error creating embeddings")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Error creating embeddings"):
            rag_preprocessor.create_vector_store(docs=documents)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.Chroma")
    def test_create_vector_store_throws_value_error_mocked_Chroma_from_documents(
        self, mock_chroma, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        mock_huggingFaceEmbeddings.return_value = None
        mock_chroma.from_documents.side_effect = ValueError("Wrong Documents")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Wrong Documents"):
            rag_preprocessor.create_vector_store(docs=documents)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.Chroma")
    def test_create_vector_store_throws_runtime_error_mocked_Chroma_from_documents(
        self, mock_chroma, mock_huggingFaceEmbeddings, rag_preprocessor
    ):
        # Arrange
        mock_huggingFaceEmbeddings.return_value = None
        mock_chroma.from_documents.side_effect = RuntimeError("Runtime error")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Runtime error"):
            rag_preprocessor.create_vector_store(docs=documents)

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.Chroma")
    def test_create_vector_store_success(
        self,
        mock_chroma,
        mock_huggingFaceEmbeddings,
        rag_preprocessor,
        mock_chroma_client,
    ):
        # Arrange
        mock_embeddings = Mock()
        mock_huggingFaceEmbeddings.return_value = mock_embeddings
        mock_vectordb_instance = Mock(spec=Chroma)
        mock_chroma.from_documents.return_value = mock_vectordb_instance
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        vectordb = rag_preprocessor.create_vector_store(docs=documents)

        # Assert
        mock_huggingFaceEmbeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_chroma.from_documents.assert_called_once_with(
            documents,
            mock_embeddings,
            client=mock_chroma_client,
            collection_name=CHROMA_COLLECTION,
        )
        assert vectordb is mock_vectordb_instance

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.Chroma")
    def test_load_vector_store_does_not_exist(
        self,
        mock_chroma,
        mock_huggingFaceEmbeddings,
        rag_preprocessor,
        mock_chroma_client,
    ):
        # Arrange
        mock_embeddings = Mock()
        mock_huggingFaceEmbeddings.return_value = mock_embeddings
        mock_vectordb = Mock(spec=Chroma)
        mock_chroma.return_value = mock_vectordb

        # Act
        result = rag_preprocessor.load_vector_store()

        # Assert - ChromaDB HTTP client handles empty collection gracefully
        mock_chroma.assert_called_once_with(
            embedding_function=mock_embeddings,
            client=mock_chroma_client,
            collection_name=CHROMA_COLLECTION,
        )
        assert result is mock_vectordb

    @patch("src.core.rag_preprocessor.HuggingFaceEmbeddings")
    @patch("src.core.rag_preprocessor.Chroma")
    def test_load_vector_store_success(
        self,
        mock_chroma,
        mock_huggingFaceEmbeddings,
        rag_preprocessor,
        mock_chroma_client,
    ):
        # Arrange
        mock_embeddings = Mock()
        mock_huggingFaceEmbeddings.return_value = mock_embeddings
        mock_chroma.return_value = Mock(spec=Chroma)

        # Act
        rag_preprocessor.load_vector_store()

        # Assert
        mock_huggingFaceEmbeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_chroma.assert_called_once_with(
            embedding_function=mock_embeddings,
            client=mock_chroma_client,
            collection_name=CHROMA_COLLECTION,
        )

    def test_get_rag_preprocessor_legacy(self, monkeypatch, mock_chroma_client):
        monkeypatch.setattr(rag_preprocessor_module, "RAG_PREPROCESSOR", "legacy")
        preprocessor = rag_preprocessor_module.get_rag_preprocessor()
        assert isinstance(preprocessor, LegacyRAGPreprocessor)

    def test_get_rag_preprocessor_docling(self, monkeypatch, mock_chroma_client):
        monkeypatch.setattr(rag_preprocessor_module, "RAG_PREPROCESSOR", "docling")
        preprocessor = rag_preprocessor_module.get_rag_preprocessor()
        assert isinstance(preprocessor, DoclingRAGPreprocessor)

    def test_get_rag_preprocessor_default(
        self, monkeypatch, caplog, mock_chroma_client
    ):
        monkeypatch.setattr(rag_preprocessor_module, "RAG_PREPROCESSOR", "unknown")
        preprocessor = rag_preprocessor_module.get_rag_preprocessor()
        assert isinstance(preprocessor, LegacyRAGPreprocessor)
        assert "Defaulting to legacy" in caplog.text

    def test_docling_split_by_numbered_headings(self, mock_chroma_client):
        preprocessor = DoclingRAGPreprocessor()
        text = "1 Intro\nFirst line\n1.1 Details\nSecond line"
        sections = preprocessor.split_by_numbered_headings(text)
        assert len(sections) == 2
        assert sections[0].metadata["section"] == "1 Intro"
        assert "First line" in sections[0].page_content

    def test_docling_split_with_fallback(self, mock_chroma_client):
        preprocessor = DoclingRAGPreprocessor()
        docs = [Document(page_content="1 Intro\nFirst line\n1.1 Details\nSecond line")]
        splits = preprocessor.split_with_fallback(docs)
        assert splits
        for doc in splits:
            assert isinstance(doc, Document)
