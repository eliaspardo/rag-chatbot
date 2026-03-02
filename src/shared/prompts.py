# flake8: noqa
import os
from langchain.prompts import PromptTemplate
from src.shared.env_loader import load_environment

load_environment()

CHATBOT_ROLE = os.getenv("CHATBOT_ROLE", "expert tutor")
USE_CASE = os.getenv("USE_CASE", "learn from the provided materials")

# Condense into a single question taking into account history
domain_expert_condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Rephrase the Follow Up Input as a standalone question. Output ONLY the question, then STOP.

Last conversation exchange:
{chat_history}

Follow Up Input: {question}
Standalone question (output the question only):""",
)

domain_expert_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""### INSTRUCTIONS ###
    You are a {CHATBOT_ROLE} helping a student {USE_CASE}.

IMPORTANT INSTRUCTIONS:
- Use the CONTEXT provided to answer the CURRENT question only.
- Keep responses concise and focused.

### CONTEXT ###
CONTEXT: {{context}}

### QUESTION ###
QUESTION: {{question}}

### ANSWER ###""",
)

exam_prep_get_question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""You are a {CHATBOT_ROLE} helping a STUDENT {USE_CASE} by providing questions about specific subjects, sections, or topics.

IMPORTANT INSTRUCTIONS:
- When STUDENT asks you to "ask me a question" or "quiz me" about a topic from the CONTEXT, respond with ONLY a clear, specific question about that topic.
- Do not repeat previous questions, always ask something new.
- Keep responses concise and focused.

Based on the CONTEXT provided, respond appropriately:

CONTEXT: {{context}}

TOPIC: {{question}}
RESPONSE:""",
)

exam_prep_get_feedback_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""You are a {CHATBOT_ROLE} helping a STUDENT {USE_CASE}.

IMPORTANT INSTRUCTIONS:
- You will receive a QUESTION AND STUDENTS ANSWER: evaluate the student's answer based on the CONTEXT and provide feedback.
- Provide student's feedback summary in a separate line at the end of the response.
- Keep responses concise and focused.

Based on the CONTEXT provided, respond appropriately:

CONTEXT: {{context}}

QUESTION AND STUDENTS ANSWER: {{question}}
RESPONSE:""",
)
