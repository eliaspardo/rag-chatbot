"""Streamlit chat application for the RAG Chatbot UI service."""

import os
from pathlib import Path
from typing import List

import streamlit as st

from src.ui_service.inference_service_client import (
    InferenceServiceClient,
    NoDocumentsIngestedError,
)

INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")

# Icon paths
SCRIPT_DIR = Path(__file__).parent
ROBOT_ICON_PATH = SCRIPT_DIR / "robot_icon.svg"


def _init_session_state() -> None:
    """Initialize Streamlit session state variables for chat history and session tracking."""
    if "domain_history" not in st.session_state:
        st.session_state.domain_history = []
    if "domain_session_id" not in st.session_state:
        st.session_state.domain_session_id = None
    if "domain_system_messages" not in st.session_state:
        st.session_state.domain_system_messages = []


def _render_system_messages(messages: List[str]) -> None:
    """Display each system message as a Streamlit warning banner."""
    for message in messages:
        st.warning(message)


AVATAR_USER = "🧑"  # Person for user
AVATAR_ASSISTANT = str(ROBOT_ICON_PATH)  # Green robot for assistant


@st.cache_resource
def _get_client() -> InferenceServiceClient:
    """Return a cached InferenceServiceClient instance."""
    return InferenceServiceClient(INFERENCE_SERVICE_URL)


def _render_domain_expert() -> None:
    """Render the domain expert chat interface with message history and input box."""
    client = _get_client()
    _render_system_messages(st.session_state.domain_system_messages)
    for message in st.session_state.domain_history:
        avatar = AVATAR_USER if message["role"] == "user" else AVATAR_ASSISTANT
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.domain_history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            response = client.ask_question(prompt, st.session_state.domain_session_id)
        except NoDocumentsIngestedError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    st.session_state.domain_session_id = response.session_id
    if response.system_message:
        st.session_state.domain_system_messages.append(response.system_message)
    st.session_state.domain_history.append(
        {"role": "assistant", "content": response.answer}
    )
    st.rerun()


def _apply_custom_css() -> None:
    """Inject custom CSS to replace the default red focus highlights with teal."""
    st.markdown(
        """
        <style>
        /* Focus states - use teal instead of red */
        *:focus {
            outline-color: #3eb489 !important;
            box-shadow: 0 0 0 2px #3eb489 !important;
        }

        /* Chat input focus */
        .stChatInput textarea:focus {
            border-color: #3eb489 !important;
            box-shadow: 0 0 0 2px rgba(62, 180, 137, 0.3) !important;
        }

        /* Text input focus */
        .stTextInput input:focus {
            border-color: #3eb489 !important;
            box-shadow: 0 0 0 2px rgba(62, 180, 137, 0.3) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chat_page():
    """Render the chat page."""
    _apply_custom_css()
    _init_session_state()
    st.title("RAG Chatbot")
    _render_domain_expert()


def main() -> None:
    """Configure Streamlit page settings and run the multi-page navigation."""
    st.set_page_config(
        page_title="Chat", page_icon=str(ROBOT_ICON_PATH), layout="centered"
    )

    # Configure navigation with custom page names
    chat = st.Page(chat_page, title="Chat", icon="💬", default=True)
    system = st.Page("pages/System.py", title="System Status", icon="⚙️")

    pg = st.navigation([chat, system])
    pg.run()


if __name__ == "__main__":
    main()
