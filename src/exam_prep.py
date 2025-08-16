from langchain.prompts import PromptTemplate

# Condense into a single question taking into account history
condense_question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You need to create a standalone question from the Follow Up Input.

Rules:
- If asking for a quiz/question ("ask me", "quiz me", "test me"), convert to: "What information should I know about [topic]?"
- If it's an answer to a previous question, return the topic from that previous question
- If it's a new question, make it standalone
- Focus on the TOPIC/SUBJECT for retrieval, not the specific request type


Last conversation exchange:
{chat_history}

Follow Up Input: {question}
Standalone question:""",
)

# System prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an ISTQB testing expert helping a STUDENT learn the ISTQB Test Manager syllabus and prepare for its exam. 

IMPORTANT INSTRUCTIONS:
- When STUDENT asks you to "ask me a question" or "quiz me" about a topic from the CONTEXT, respond with ONLY a clear, specific question about that topic.
- When STUDENT provides an answer to a question you previously asked, evaluate their answer based on the CONTEXT and provide feedback.
- When STUDENT asks a factual question answer based on the CONTEXT provided
- Do not repeat previous questions, always ask something new.
- Keep responses concise and focused.

Based on the CONTEXT provided, respond appropriately:

CONTEXT: {context}

STUDENT'S LAST INPUT: {question}
RESPONSE:""",
)


def exam_prep(vectordb):
    return
