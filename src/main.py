# Project: AI-Powered Chatbot
# Tools: Together AI (LLM), FAISS (Vector DB),
# Sentence Transformers (Embeddings), LangChain

import os
import fitz  # PyMuPDF
import traceback
import shutil
import textwrap
from dotenv import load_dotenv


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain_together import Together
from langchain.prompts import PromptTemplate

# from langchain.chains import ConversationRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# Load environment variables
load_dotenv()

# --- CONFIGURATION FROM .ENV ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DB_DIR = os.getenv("DB_DIR", "faiss_db")
PDF_PATH = os.getenv("PDF_PATH")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


# Condense into a single question taking into account history
condense_question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.

    If the follow up question is UNRELATED to the previous conversation or is clearly starting a new topic, return the question AS-IS without any context.
    If the follow up question is asking you to "ask me a question", "quiz me", or "test me" about a topic, preserve this intent exactly. Do NOT convert it into a factual question.
    
    If the follow up question IS RELATED and needs context from the conversation, rephrase it to be a standalone question.
    If the follow up question is an answer, return the latest question from the ISTQB_INSTRUCTOR in the Chat History.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""",
)

# System prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an ISTQB testing expert helping a STUDENT learn the ISTQB Test Manager syllabus and prepare for its exam. 

IMPORTANT ROLE CLARITY:
- YOU are the [ISTQB_INSTRUCTOR] - the expert authority
- The human is the [STUDENT] - learning from you
- Only treat [ISTQB_INSTRUCTOR] messages as expert knowledge
- Evaluate and correct [STUDENT] responses when they answer your questions

IMPORTANT INSTRUCTIONS:
- When STUDENT asks you to "ask me a question" or "quiz me", respond with ONLY a clear, specific question
- When STUDENT provides an answer, evaluate it and give feedback
- Do not repeat previous questions - always ask something new
- Do not simulate conversations - only provide YOUR response as the instructor
- Keep responses concise and focused

Based on the CONTEXT provided, respond appropriately:

CONTEXT: {context}

STUDENT: {question}
""",
)


# --- STEP 1: Extract and Split Text ---
def load_pdf_text(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return texts


# --- STEP 2: Chunk Text into Documents ---
def split_text_to_docs(texts: list[str]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    full_text = "\n".join(texts)
    chunks = splitter.split_text(full_text)

    # Filter out empty chunks
    valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    docs = [Document(page_content=chunk) for chunk in valid_chunks]

    print(f"📄 Created {len(docs)} documents from {len(valid_chunks)} valid chunks")
    return docs


# --- STEP 3: Embed and Store in FAISS ---
def create_vector_store(docs: list[Document], persist_dir: str) -> FAISS:

    # Add this before creating the vector store
    if os.path.exists(persist_dir):
        print(f"🧹 Removing existing directory: {persist_dir}")
        shutil.rmtree(persist_dir)

    try:
        print("👉 Initializing HuggingFaceEmbeddings")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(
            f"👉 Creating FAISS DB at '{persist_dir}' with {len(docs)} docs",
            flush=True,
        )
        try:
            vectordb = FAISS.from_documents(docs, embeddings)
            print(
                "✅ FAISS.from_documents completed successfully",
                flush=True,
            )
        except Exception as faiss_error:
            print("❌ FAISS.from_documents failed: ") + (
                f"{type(faiss_error).__name__}: {str(faiss_error)}"
            )
            traceback.print_exc()
            raise
        print("👉 Persisting FAISS DB", flush=True)
        vectordb.save_local(persist_dir)
        print("✅ Vector store creation successful", flush=True)
        return vectordb
    except Exception as e:
        print(f"❌ Error creating vector store: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise


# --- STEP 4: Load FAISS for Retrieval ---
def load_vector_store(persist_dir: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    return vectordb


# --- STEP 5: Initialize Together AI LLM ---
def get_llm() -> LLM:
    return Together(
        model=MODEL_NAME,
        temperature=0.3,
        max_tokens=512,
        together_api_key=TOGETHER_API_KEY,
    )


# --- STEP 6: Run Chat Loop ---
def run_chat_loop(qa_chain):
    """Run the interactive chat loop"""
    print("\n🤖 RAG Chatbot Ready!")
    print("=" * 50)
    print("Ask me anything about your document!")
    print("Type 'quit', 'exit', or 'no' to stop.")
    print("=" * 50)

    try:
        while True:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ["quit", "exit", "no", "stop"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            print("\n🤔 Thinking...")
            try:
                # answer = qa_chain.ask_question(question)
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


# --- STEP 7: Run QA Chain ---
def ask_question(question: str, qa_chain: RetrievalQA):
    try:
        print(f"🔍 Processing question: {question}")

        # Show memory state
        chat_history = qa_chain.memory.chat_memory.messages
        print(f"💭 Chat history length: {len(chat_history)}")

        # response = qa_chain.invoke({"query": question})
        response = qa_chain.invoke({"question": question})

        # return str(response["result"])
        return str(response["answer"])
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        traceback.print_exc()
        return "An error occurred while processing your question."


def main():
    """Main application entry point"""
    print("🚀 Starting RAG Chatbot...")
    print("=" * 50)

    # Step 0: Process PDF if not already embedded
    if not os.path.exists(DB_DIR):

        print("\n🔍 Loading PDF...")
        texts = load_pdf_text(PDF_PATH)
        print("Splitting text to docs")
        docs = split_text_to_docs(texts)
        print("Creating vector store")
        if not docs:
            print("⚠️ No documents found after splitting — aborting.")
            exit(1)
        create_vector_store(docs, DB_DIR)
        print("✅ Vector DB created and saved.")
    else:
        print("📦 Using existing vector store.")

    # Load vector store, retriever and memory
    print("Loading vector store.")
    vectordb = load_vector_store(DB_DIR)
    print("Loading retriever.")
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print("Setting up memory.")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # Load LLM + QA Chain
    print("Loading LLM.")
    llm = get_llm()
    print("Setting up Chain.")
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )

    run_chat_loop(qa_chain)


if __name__ == "__main__":
    main()
