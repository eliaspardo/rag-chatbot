from enum import Enum


class ChatbotMode(Enum):
    DOMAIN_EXPERT = "Domain Expert"
    EXAM_PREP = "Exam Prep"


class Error(Enum):
    INVALID_MODE = "Invalid Operational Mode"
    NOT_A_QUESTION = "Not a question"
    NOT_A_TOPIC = "Not a topic"
    NO_USER_ANSWER = "User did not input answer"
    QUESTION_EXCEPTION = "Exception when asking question to LLM"
    ANSWER_EXCEPTION = "Exception from LLM when answering question"


EXIT_WORDS = {"quit", "exit", "no", "stop"}
