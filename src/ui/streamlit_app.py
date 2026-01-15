import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30

MODE_DOMAIN_EXPERT = "Domain Expert"
MODE_EXAM_PREP = "Exam Prep"
MODES = [MODE_DOMAIN_EXPERT, MODE_EXAM_PREP]


def _init_session_state() -> None:
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = MODE_DOMAIN_EXPERT
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

    if "exam_history" not in st.session_state:
        st.session_state.exam_history = []
    if "exam_stage" not in st.session_state:
        st.session_state.exam_stage = "topic"
    if "exam_question" not in st.session_state:
        st.session_state.exam_question = ""


def _reset_mode_state() -> None:
    st.session_state.domain_history = []
    st.session_state.domain_session_id = None
    st.session_state.domain_system_messages = []

    st.session_state.exam_history = []
    st.session_state.exam_stage = "topic"
    st.session_state.exam_question = ""


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


def _render_exam_prep_history() -> None:
    if not st.session_state.exam_history:
        return
    with st.expander("Previous questions"):
        for entry in st.session_state.exam_history:
            st.markdown(f"**Topic:** {entry['topic']}")
            st.markdown(f"**Question:** {entry['question']}")
            if entry.get("answer"):
                st.markdown(f"**Your answer:** {entry['answer']}")
            if entry.get("feedback"):
                st.markdown(f"**Feedback:** {entry['feedback']}")
            st.divider()


def _render_exam_prep() -> None:
    _render_exam_prep_history()

    if st.session_state.exam_stage == "topic":
        with st.form("exam_topic_form"):
            topic = st.text_input("Section / topic you want to be quizzed about")
            submitted = st.form_submit_button("Get question")
        if not submitted:
            return
        if not topic.strip():
            st.error("Please enter a topic.")
            return
        with st.spinner("Generating a question..."):
            data = _post_json(
                f"{st.session_state.api_base_url}/chat/exam-prep/get_question/",
                {"user_topic": topic},
            )
        if not data:
            return
        question = data["llm_question"]
        st.session_state.exam_question = question
        st.session_state.exam_history.append(
            {"topic": topic, "question": question, "answer": "", "feedback": ""}
        )
        st.session_state.exam_stage = "answer"
        st.rerun()

    if st.session_state.exam_stage == "answer":
        if not st.session_state.exam_question:
            st.session_state.exam_stage = "topic"
            st.rerun()
        st.markdown(f"**Question:** {st.session_state.exam_question}")
        with st.form("exam_answer_form"):
            answer = st.text_area("Your answer")
            submitted = st.form_submit_button("Get feedback")
        if not submitted:
            return
        if not answer.strip():
            st.error("Please enter your answer.")
            return
        with st.spinner("Reviewing your answer..."):
            data = _post_json(
                f"{st.session_state.api_base_url}/chat/exam-prep/get_feedback/",
                {
                    "llm_question": st.session_state.exam_question,
                    "user_answer": answer,
                },
            )
        if not data:
            return
        feedback = data["feedback"]
        st.session_state.exam_history[-1]["answer"] = answer
        st.session_state.exam_history[-1]["feedback"] = feedback
        st.session_state.exam_stage = "done"
        st.rerun()

    if st.session_state.exam_stage == "done":
        last_entry = st.session_state.exam_history[-1]
        st.markdown(f"**Question:** {last_entry['question']}")
        st.markdown(f"**Your answer:** {last_entry['answer']}")
        st.markdown(f"**Feedback:** {last_entry['feedback']}")
        if st.button("Ask another question"):
            st.session_state.exam_stage = "topic"
            st.session_state.exam_question = ""
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="R", layout="centered")
    _init_session_state()

    st.sidebar.header("Settings")
    api_base_url = st.sidebar.text_input("API base URL", key="api_base_url")
    if api_base_url != st.session_state.api_base_url_last:
        st.session_state.api_base_url_last = api_base_url
        _reset_mode_state()

    mode = st.sidebar.radio(
        "Mode",
        MODES,
        index=MODES.index(st.session_state.active_mode),
    )
    if mode != st.session_state.active_mode:
        st.session_state.active_mode = mode
        _reset_mode_state()

    st.title("RAG Chatbot")
    if st.session_state.active_mode == MODE_DOMAIN_EXPERT:
        st.subheader("Domain Expert")
        _render_domain_expert()
    else:
        st.subheader("Exam Prep")
        _render_exam_prep()


if __name__ == "__main__":
    main()
