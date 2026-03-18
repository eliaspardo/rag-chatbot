import os
from pathlib import Path
from typing import List

import streamlit as st

from src.ui_service.inference_service_client import InferenceServiceClient

INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")

# Icon paths
SCRIPT_DIR = Path(__file__).parent
ROBOT_ICON_PATH = SCRIPT_DIR / "robot_icon.svg"


def _init_session_state() -> None:
    if "domain_history" not in st.session_state:
        st.session_state.domain_history = []
    if "domain_session_id" not in st.session_state:
        st.session_state.domain_session_id = None
    if "domain_system_messages" not in st.session_state:
        st.session_state.domain_system_messages = []


def _render_system_messages(messages: List[str]) -> None:
    for message in messages:
        st.warning(message)


AVATAR_USER = "🧑"  # Person for user
AVATAR_ASSISTANT = str(ROBOT_ICON_PATH)  # Green robot for assistant


def _get_status_icon(status: str) -> str:
    status_lower = status.lower()
    if "completed" in status_lower:
        return "✅"
    if "pending" in status_lower:
        return "⏳"
    return "❌"


@st.fragment(run_every=30)
def _render_health_status(client: InferenceServiceClient) -> None:
    with st.expander("Inference Service Health", expanded=False):
        if st.button("Refresh"):
            st.rerun(scope="fragment")

        health = client.get_health()

        if not health.is_healthy:
            st.error(health.error_message or "Inference service unavailable")
            return

        st.success("Connected")
        st.metric("Documents in vector store", health.vector_store_count)

        if not health.documents:
            st.info("No documents loaded yet")
            return

        for doc in health.documents:
            icon = _get_status_icon(doc.status)
            st.write(f"{icon} **{doc.doc_name}** — {doc.status}")


def _render_domain_expert(client: InferenceServiceClient) -> None:
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
            response = client.ask_question(
                prompt, st.session_state.domain_session_id
            )
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


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot", page_icon=str(ROBOT_ICON_PATH), layout="centered"
    )
    _apply_custom_css()
    _init_session_state()

    client = InferenceServiceClient(INFERENCE_SERVICE_URL)

    st.title("RAG Chatbot")
    _render_health_status(client)
    _render_domain_expert(client)


if __name__ == "__main__":
    main()
