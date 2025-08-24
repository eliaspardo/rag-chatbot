import sys
import textwrap
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from chain_manager import ChainManager
from prompts import exam_prep_question_prompt, exam_prep_answer_prompt


# --- Run Chat Loop ---
def run_chat_loop(
    chain_manager: ChainManager,
    question_chain: ConversationalRetrievalChain,
    answer_chain: ConversationalRetrievalChain,
):
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
                llm_question = chain_manager.ask_question(topic, question_chain)
                print("\n❓ Question:")
                print("=" * 50)
                print(textwrap.fill(llm_question, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try rephrasing your question.")

            user_answer = input("\n📝 Your answer: ").strip()

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
                llm_answer = chain_manager.ask_question(
                    llm_question_user_answer, answer_chain
                )
                print("\n💡 Answer:")
                print("=" * 50)
                print(textwrap.fill(llm_answer, width=80))
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error processing answer: {e}")
                print("Please try rephrasing your answer.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def exam_prep(vectordb):
    chain_manager = ChainManager(vectordb)
    print("🧠 Loading LLM.")
    llm = chain_manager.get_llm()

    print("⛓ Setting up Chains.")
    question_chain = chain_manager.get_conversationalRetrievalChain(
        llm, {"prompt": exam_prep_question_prompt}
    )

    answer_chain = chain_manager.get_conversationalRetrievalChain(
        llm, {"prompt": exam_prep_answer_prompt}
    )

    run_chat_loop(chain_manager, question_chain, answer_chain)

    return
