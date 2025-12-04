import os
from langchain.prompts import PromptTemplate
from src.env_loader import load_environment

load_environment()

CHATBOT_ROLE = os.getenv("CHATBOT_ROLE", "expert tutor")
USE_CASE = os.getenv("USE_CASE", "learn from the provided materials")

# Condense into a single question taking into account history
condense_question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Create a standalone question from the Follow Up Inpu that incorporates relevant context from the conversation history.

Last conversation exchange:
{chat_history}

Follow Up Input: {question}
Standalone question:""",
)

domain_expert_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""You are a {CHATBOT_ROLE} helping a student {USE_CASE}. 

IMPORTANT INSTRUCTIONS:
- Use the CONTEXT provided to answer the question.
- Keep responses concise and focused.

Based on the CONTEXT provided, respond appropriately:

CONTEXT: {{context}}

QUESTION: {{question}}
ANSWER:""",
)

exam_prep_question_prompt = PromptTemplate(
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

exam_prep_answer_prompt = PromptTemplate(
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
