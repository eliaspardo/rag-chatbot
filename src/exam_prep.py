import sys
import os
import textwrap
import traceback

from langchain_together import Together
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from prompts import exam_prep_question_prompt, exam_prep_answer_prompt

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION FROM .ENV ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))


# --- Initialize Together AI LLM ---
def get_llm() -> LLM:
    return Together(
        model=MODEL_NAME,
        temperature=0.3,
        max_tokens=512,
        together_api_key=TOGETHER_API_KEY,
    )


# --- Run Chat Loop ---
def run_chat_loop(question_chain, answer_chain):
    print("\n" + "=" * 50)
    print("\n🤖 RAG Chatbot in Exam Prep Mode Ready!")
    print("Provide a topic and I'll give you a question about it.")
    print("\n⚙ Type 'mode' to return to Operational Mode selection menu.")
    print("=" * 50)

    try:
        while True:
            print("What section or topic do you want me to quiz you about?")
            topic = input("\n🧠 Section / topic you want to be quizzed about: ").strip()

            if topic.lower() in ["quit", "exit", "no", "stop"]:
                print("\n👋 Goodbye!")
                sys.exit(0)

            if topic.lower() == "mode":
                print("\n🔄 Returning to Operational Mode selection...")
                break

            if not topic:
                print("\n❌ Please enter a section / topic.")
                continue

            print("\n🤔 Thinking...")
            try:
                llm_question = ask_question(topic, question_chain)
                print("\n❓ Your question:")
                print("=" * 50)
                print(textwrap.fill(llm_question, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try rephrasing your question.")

            user_answer = input("\n❓ Your answer: ").strip()

            if user_answer.lower() in ["quit", "exit", "no", "stop"]:
                print("\n👋 Goodbye!")
                sys.exit(0)

            if user_answer.lower() == "mode":
                print("\n🔄 Returning to Operational Mode selection...")
                break

            if not user_answer:
                print("\n❌ Let's start all over again.")
                continue

            # Evaluate user's answer
            llm_question_user_answer = llm_question + "\n" + user_answer
            print("\n🤔 Thinking...")
            try:
                llm_answer = ask_question(llm_question_user_answer, answer_chain)
                print("\n💡 Answer:")
                print("=" * 50)
                print(textwrap.fill(llm_answer, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing answer: {e}")
                print("Please try rephrasing your answer.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# --- Run QA Chain ---
def ask_question(question: str, qa_chain: RetrievalQA):
    try:
        # Show memory state
        # chat_history = qa_chain.memory.chat_memory.messages
        # print(f"💭 Chat history length: {len(chat_history)}")

        response = qa_chain.invoke({"question": question})

        return str(response["answer"])
    except Exception as e:
        print(f"❌ Error invoking LLM: {e}")
        traceback.print_exc()
        return "An error occurred while processing your question."


def exam_prep(vectordb):
    print("🐕 Loading retriever.")
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    print("💾 Setting up memory.")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    print("🧠 Loading LLM.")
    llm = get_llm()

    print("⛓ Setting up Chain.")
    question_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": exam_prep_question_prompt},
        # verbose=True,
    )

    answer_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": exam_prep_answer_prompt},
        # verbose=True,
    )

    run_chat_loop(question_chain, answer_chain)

    return
