class ExitApp(Exception):
    """Raised when the user wants to exit the application."""

    pass


class ConfigurationException(Exception):
    """Raised when there's an issue with the configuration."""

    pass


class ServerSetupException(Exception):
    """Raised when there's an issue starting up the server."""

    pass


class DomainExpertSetupException(Exception):
    """Raised when there's an issue setting up the Domain Expert."""

    pass


class DomainExpertQueryException(Exception):
    """Raised when there's an issue retrieving an answer in Domain Expert."""

    pass


class ExamPrepSetupException(Exception):
    """Raised when there's an issue setting up the Exam Prep."""

    pass


class ExamPrepQueryException(Exception):
    """Raised when there's an issue retrieving an answer in Exam Prep."""

    pass


class VectorStoreException(Exception):
    """Base exception for vector store errors."""


class ChromaException(VectorStoreException):
    """Raised when Chroma DB creation fails."""


class NoDocumentsException(Exception):
    """Raised when preprocessing produces zero documents."""
