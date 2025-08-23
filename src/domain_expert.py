import sys
import os
import textwrap
import traceback

from langchain_together import Together
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from prompts import condense_question_prompt, domain_expert_prompt

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
def run_chat_loop(qa_chain):
    print("\n" + "=" * 50)
    print("\n🤖 RAG Chatbot in Domain Expert Mode Ready!")
    print("=" * 50)
    print("Ask me anything about your document.")
    print("\n⚙ Type 'mode' to return to Operational Mode selection menu.")
    print("=" * 50)

    try:
        while True:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ["quit", "exit", "no", "stop"]:
                print("\n👋 Goodbye!")
                sys.exit(0)

            if question.lower() == "mode":
                print("\n🔄 Returning to Operational Mode selection...")
                break

            if not question:
                print("\n❌ Please enter a question.")
                continue

            print("\n🤔 Thinking...")
            try:
                answer = ask_question(question, qa_chain)
                print("\n💡 Answer:")
                print("=" * 50)
                print(textwrap.fill(answer, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try rephrasing your question.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# --- Run QA Chain ---
def ask_question(question: str, qa_chain: RetrievalQA):
    try:
        response = qa_chain.invoke({"question": question})

        return str(response["answer"])
    except Exception as e:
        print(f"❌ Error invoking LLM: {e}")
        traceback.print_exc()
        return "An error occurred while processing your question."


def domain_expert(vectordb):
    print("🐕 Loading retriever.")
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    print("💾 Setting up memory.")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    print("🧠 Loading LLM.")
    llm = get_llm()

    print("⛓ Setting up Chain.")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": domain_expert_prompt},
        verbose=True,  # Uncomment for debugging
    )

    run_chat_loop(qa_chain)

    return
