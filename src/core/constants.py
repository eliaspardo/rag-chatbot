from enum import Enum


class ChatbotMode(Enum):
    DOMAIN_EXPERT = "Domain Expert"
    EXAM_PREP = "Exam Prep"


class Error(Enum):
    NO_DOCUMENTS = "No documents provided"
    INVALID_MODE = "Invalid Operational Mode"
    NOT_A_QUESTION = "Not a question"
    NOT_A_TOPIC = "Not a topic"
    NO_USER_ANSWER = "User did not input answer"
    EXCEPTION = "Exception"


EXIT_WORDS = {"quit", "exit", "no", "stop"}

DEFAULT_CONSOLE_WIDTH = 80
