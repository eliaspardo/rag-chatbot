# Project: AI-Powered Chatbot
# Tools: Together AI (LLM), FAISS (Vector DB),
# Sentence Transformers (Embeddings), LangChain

import sys
import os
import fitz  # PyMuPDF
import traceback
import shutil
from dotenv import load_dotenv


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from domain_expert import domain_expert
from exam_prep import exam_prep


# Load environment variables
load_dotenv()

# --- CONFIGURATION FROM .ENV ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DB_DIR = os.getenv("DB_DIR", "faiss_db")
PDF_PATH = os.getenv("PDF_PATH")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


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


def main():
    """Main application entry point"""
    print("🚀 Starting RAG Chatbot...")
    print("=" * 50)

    # Step 0: Process PDF if not already embedded
    if not os.path.exists(DB_DIR):

        print("\n🔍 Loading PDF...")
        texts = load_pdf_text(PDF_PATH)
        print("Splitting text to docs.")
        docs = split_text_to_docs(texts)
        print("Creating vector store.")
        if not docs:
            print("⚠️ No documents found after splitting — aborting.")
            exit(1)
        create_vector_store(docs, DB_DIR)
        print("✅ Vector DB created and saved.")
    else:
        print("📦 Using existing vector store.")

    # Load vector store, retriever and memory
    print("📶 Loading vector store.")
    vectordb = load_vector_store(DB_DIR)

    while True:
        print("=" * 50)
        print("\n Select an Operational Mode:")
        print("1) 🎓 Domain Expert Chatbot - Ask questions about the context imported.")
        print("2) 📝 Exam Prep Chatbot - Get a question from a particular topic.")
        print("\n Type 'quit', 'exit', or 'no' to stop.")
        print("=" * 50)

        user_selection = input("\n☰ Your selection:").strip()

        if user_selection.lower() in ["quit", "exit", "no", "stop"]:
            print("\n👋 Goodbye!")
            sys.exit(0)

        if user_selection == "1":
            print("\n ⎆ Entering Domain Expert Chatbot mode...")
            domain_expert(vectordb)
            continue
        if user_selection == "2":
            print("\n ⎆ Entering Exam Prep Chatbot mode...")
            exam_prep(vectordb)
            continue
        else:
            print("Please select a valid Operational Mode!")


if __name__ == "__main__":
    main()
