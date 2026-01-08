class ExitApp(Exception):
    """Raised when the user wants to exit the application."""

    pass


class DomainExpertSetupException(Exception):
    """Raised when there's an issue setting up the domain expert."""

    pass


class DomainExpertQueryException(Exception):
    """Raised when there's an issue retrieving an answer in domain expert."""

    pass


class VectorStoreException(Exception):
    """Base exception for vector store errors."""


class FaissException(VectorStoreException):
    """Raised when FAISS DB creation fails."""


class NoDocumentsException(Exception):
    """Raised when preprocessing produces zero documents."""
