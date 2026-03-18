import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30

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


def _post_json(url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
    except ValueError:
        st.error("Invalid response from API.")
    return None


def _render_system_messages(messages: List[str]) -> None:
    for message in messages:
        st.warning(message)


AVATAR_USER = "🧑"  # Person for user
AVATAR_ASSISTANT = str(ROBOT_ICON_PATH)  # Green robot for assistant


def _render_domain_expert() -> None:
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
        data = _post_json(
            f"{DEFAULT_API_BASE_URL}/chat/domain-expert/",
            {
                "question": prompt,
                "session_id": st.session_state.domain_session_id,
            },
        )

    if not data:
        return

    st.session_state.domain_session_id = data["session_id"]
    system_message = data.get("system_message")
    if system_message:
        st.session_state.domain_system_messages.append(system_message)
    st.session_state.domain_history.append(
        {"role": "assistant", "content": data["answer"]}
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

    st.title("RAG Chatbot")
    _render_domain_expert()


if __name__ == "__main__":
    main()
