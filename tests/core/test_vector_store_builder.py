import pytest
from unittest.mock import Mock, patch
from src.ingestion_service import vector_store_builder as vector_store_builder_module
from src.ingestion_service.vector_store_builder import (
    DoclingVectorStoreBuilder,
    LegacyVectorStoreBuilder,
    VectorStoreBuilder,
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


class TestVectorStoreBuilder:
    @pytest.fixture
    def mock_chroma_client(self):
        with patch(
            "src.ingestion_service.vector_store_builder.chromadb.HttpClient"
        ) as mock:
            yield mock.return_value

    @pytest.fixture
    def vector_store_builder(self, mock_chroma_client):
        return LegacyVectorStoreBuilder()

    def test_base_load_pdf_text_not_implemented(self, mock_chroma_client):
        with pytest.raises(NotImplementedError):
            VectorStoreBuilder().load_pdf_text(TEST_PDF)

    def test_base_split_text_to_docs_not_implemented(self, mock_chroma_client):
        with pytest.raises(NotImplementedError):
            VectorStoreBuilder().split_text_to_docs([Document(page_content="test")])

    def test_load_pdf_text_no_file(self, vector_store_builder):
        with pytest.raises(Exception, match="Error reading PDF file"):
            vector_store_builder.load_pdf_text("no_file")

    @patch("src.ingestion_service.vector_store_builder.fitz.open")
    def test_load_pdf_text_default_no_file(self, mock_fitz_open, vector_store_builder):
        mock_fitz_open.side_effect = FileNotFoundError("No such file or directory")
        with pytest.raises(Exception, match="Error reading PDF file"):
            vector_store_builder.load_pdf_text(TEST_PDF)

    def test_load_pdf_text_successful(self, vector_store_builder):
        docs = vector_store_builder.load_pdf_text(TEST_PDF)
        full_text = "".join(doc.page_content for doc in docs)
        assert "Congratulations" in full_text

    def test_split_text_to_docs_success(self, vector_store_builder):
        documents = vector_store_builder.split_text_to_docs(
            [Document(page_content=item) for item in STRING_LIST]
        )
        assert len(documents) > 0
        for document in documents:
            print("document")
            assert isinstance(document, Document)

    def test_split_text_to_docs_empty_list(self, vector_store_builder):
        documents = vector_store_builder.split_text_to_docs(EMPTY_LIST)
        assert len(documents) == 0

    def test_split_text_to_docs_empty_chunks(self, vector_store_builder):
        documents = vector_store_builder.split_text_to_docs(
            [Document(page_content=item) for item in STRING_LIST_EMPTY_CHUNKS]
        )
        assert all(document.page_content.strip() for document in documents)

    @patch("src.ingestion_service.vector_store_builder.HuggingFaceEmbeddings")
    def test_create_vector_store_throws_exception_mocked_HuggingFaceEmbeddings(
        self, mock_huggingFaceEmbeddings, vector_store_builder
    ):
        # Arrange
        mock_huggingFaceEmbeddings.side_effect = Exception("Error creating embeddings")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Error creating embeddings"):
            vector_store_builder.create_vector_store(docs=documents)

    @patch("src.ingestion_service.vector_store_builder.HuggingFaceEmbeddings")
    @patch("src.ingestion_service.vector_store_builder.Chroma")
    def test_create_vector_store_throws_value_error_mocked_Chroma_from_documents(
        self, mock_chroma, mock_huggingFaceEmbeddings, vector_store_builder
    ):
        # Arrange
        mock_huggingFaceEmbeddings.return_value = None
        mock_chroma.from_documents.side_effect = ValueError("Wrong Documents")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Wrong Documents"):
            vector_store_builder.create_vector_store(docs=documents)

    @patch("src.ingestion_service.vector_store_builder.HuggingFaceEmbeddings")
    @patch("src.ingestion_service.vector_store_builder.Chroma")
    def test_create_vector_store_throws_runtime_error_mocked_Chroma_from_documents(
        self, mock_chroma, mock_huggingFaceEmbeddings, vector_store_builder
    ):
        # Arrange
        mock_huggingFaceEmbeddings.return_value = None
        mock_chroma.from_documents.side_effect = RuntimeError("Runtime error")
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act & Assert
        with pytest.raises(Exception, match="Runtime error"):
            vector_store_builder.create_vector_store(docs=documents)

    @patch("src.ingestion_service.vector_store_builder.HuggingFaceEmbeddings")
    @patch("src.ingestion_service.vector_store_builder.Chroma")
    def test_create_vector_store_success(
        self,
        mock_chroma,
        mock_huggingFaceEmbeddings,
        vector_store_builder,
        mock_chroma_client,
    ):
        # Arrange
        mock_embeddings = Mock()
        mock_huggingFaceEmbeddings.return_value = mock_embeddings
        mock_vectordb_instance = Mock(spec=Chroma)
        mock_chroma.from_documents.return_value = mock_vectordb_instance
        documents = [Document(page_content=PAGE_CONTENT)]

        # Act
        vectordb = vector_store_builder.create_vector_store(docs=documents)

        # Assert
        mock_huggingFaceEmbeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_chroma.from_documents.assert_called_once_with(
            documents,
            mock_embeddings,
            client=mock_chroma_client,
            collection_name=CHROMA_COLLECTION,
        )
        assert vectordb is mock_vectordb_instance

    def test_get_vector_store_builder_legacy(self, monkeypatch, mock_chroma_client):
        monkeypatch.setattr(vector_store_builder_module, "RAG_PREPROCESSOR", "legacy")
        builder = vector_store_builder_module.get_vector_store_builder()
        assert isinstance(builder, LegacyVectorStoreBuilder)

    def test_get_vector_store_builder_docling(self, monkeypatch, mock_chroma_client):
        monkeypatch.setattr(vector_store_builder_module, "RAG_PREPROCESSOR", "docling")
        builder = vector_store_builder_module.get_vector_store_builder()
        assert isinstance(builder, DoclingVectorStoreBuilder)

    def test_get_vector_store_builder_default(
        self, monkeypatch, caplog, mock_chroma_client
    ):
        monkeypatch.setattr(vector_store_builder_module, "RAG_PREPROCESSOR", "unknown")
        builder = vector_store_builder_module.get_vector_store_builder()
        assert isinstance(builder, LegacyVectorStoreBuilder)
        assert "Defaulting to legacy" in caplog.text

    def test_docling_split_by_numbered_headings(self, mock_chroma_client):
        builder = DoclingVectorStoreBuilder()
        text = "1 Intro\nFirst line\n1.1 Details\nSecond line"
        sections = builder.split_by_numbered_headings(text)
        assert len(sections) == 2
        assert sections[0].metadata["section"] == "1 Intro"
        assert "First line" in sections[0].page_content

    def test_docling_split_with_fallback(self, mock_chroma_client):
        builder = DoclingVectorStoreBuilder()
        docs = [Document(page_content="1 Intro\nFirst line\n1.1 Details\nSecond line")]
        splits = builder.split_with_fallback(docs)
        assert splits
        for doc in splits:
            assert isinstance(doc, Document)
