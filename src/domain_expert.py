import sys
import textwrap
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from chain_manager import ChainManager
from prompts import condense_question_prompt, domain_expert_prompt


# --- Run Chat Loop ---
def run_chat_loop(chain_manager: ChainManager, qa_chain: ConversationalRetrievalChain):
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
                answer = chain_manager.ask_question(question, qa_chain)
                print("\n💡 Answer:")
                print("=" * 50)
                print(textwrap.fill(answer, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try rephrasing your question.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def domain_expert(vectordb):
    chain_manager = ChainManager(vectordb)
    print("🧠 Loading LLM.")
    llm = chain_manager.get_llm()

    print("⛓ Setting up Chain.")
    qa_chain = chain_manager.get_conversationalRetrievalChain(
        llm,
        {"prompt": domain_expert_prompt},
        condense_question_prompt=condense_question_prompt,
    )

    run_chat_loop(chain_manager, qa_chain)

    return
