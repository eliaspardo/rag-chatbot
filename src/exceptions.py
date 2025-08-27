class ExitApp(Exception):
    """Raised when the user wants to exit the application."""

    pass


class VectorStoreException(Exception):
    """Base exception for vector store errors."""


class FaissException(VectorStoreException):
    """Raised when FAISS DB creation fails."""
