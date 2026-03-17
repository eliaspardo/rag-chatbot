import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30


def _init_session_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL
    if "api_base_url_last" not in st.session_state:
        st.session_state.api_base_url_last = st.session_state.api_base_url

    if "domain_history" not in st.session_state:
        st.session_state.domain_history = []
    if "domain_session_id" not in st.session_state:
        st.session_state.domain_session_id = None
    if "domain_system_messages" not in st.session_state:
        st.session_state.domain_system_messages = []


def _reset_state() -> None:
    st.session_state.domain_history = []
    st.session_state.domain_session_id = None
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


def _render_domain_expert() -> None:
    _render_system_messages(st.session_state.domain_system_messages)
    for message in st.session_state.domain_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.domain_history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        data = _post_json(
            f"{st.session_state.api_base_url}/chat/domain-expert/",
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


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="R", layout="centered")
    _init_session_state()

    st.sidebar.header("Settings")
    api_base_url = st.sidebar.text_input("API base URL", key="api_base_url")
    if api_base_url != st.session_state.api_base_url_last:
        st.session_state.api_base_url_last = api_base_url
        _reset_state()

    st.title("RAG Chatbot")
    st.subheader("Domain Expert")
    _render_domain_expert()


if __name__ == "__main__":
    main()
