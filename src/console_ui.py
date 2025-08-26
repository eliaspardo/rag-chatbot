import textwrap
from constants import ChatbotMode, Error


class ConsoleUI:
    def show_welcome(self) -> None:
        print("🚀 Starting RAG Chatbot...")
        print("=" * 50)

    def show_operational_mode_selection(self) -> None:
        print("=" * 50)
        print("\n Select an Operational Mode:")
        print("1) 🎓 Domain Expert Chatbot - Ask questions about the context imported.")
        print("2) 📝 Exam Prep Chatbot - Get a question from a particular topic.")
        print("\n Type 'quit', 'exit', or 'no' to stop.")
        print("=" * 50)

    def get_operational_mode_selection(self) -> str:
        return input("\n☰ Your selection:").strip()

    def show_entering_mode(self, mode: ChatbotMode) -> None:
        print(f"\n ⎆ Entering {mode.value} Chatbot mode...")

    def show_welcome_mode(self, mode: ChatbotMode) -> None:
        print("\n" + "=" * 50)
        print(f"\n🤖 RAG Chatbot in {mode.value} Mode Ready!")
        print("\n⚙ Type 'mode' to return to Operational Mode selection menu.")
        print("=" * 50)
        if mode == ChatbotMode.DOMAIN_EXPERT:
            print("\nAsk me anything about your document.")
        if mode == ChatbotMode.EXAM_PREP:
            print("\nWhat section or topic you want me to quiz you on?")
        print("=" * 50)

    def show_mode_switch(self) -> None:
        print("\n🔄 Returning to Operational Mode selection...")

    def get_user_input(self, prompt: str) -> str:
        return input(prompt).strip()

    def show_info_message(self, message: str) -> None:
        print(message)

    def show_answer(self, answer: str) -> None:
        print("\n💡 Answer:")
        print("=" * 50)
        print(textwrap.fill(answer, width=80))
        print("=" * 50)

    def show_llm_question(self, llm_question: str) -> None:
        print("\n❓ Question:")
        print("=" * 50)
        print(textwrap.fill(llm_question, width=80))
        print("=" * 50)

    def show_error(self, error: Error, exception: Exception = None) -> None:
        match error:
            case Error.NO_DOCUMENTS:
                print("⚠️ No documents found after splitting — aborting.")
            case Error.INVALID_MODE:
                print("Please select a valid Operational Mode!")
            case Error.NOT_A_QUESTION:
                print("\n❌ Please enter a question.")
            case Error.NOT_A_TOPIC:
                print("\n❌ Please enter a section / topic.")
            case Error.NO_USER_ANSWER:
                print("\n❌ Let's start all over again.")
            case Error.EXCEPTION:
                print(f"❌ {exception}")

    def show_exit_message(self) -> None:
        print("\n👋 Goodbye!")
