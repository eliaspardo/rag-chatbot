import os

import streamlit as st

from src.ui_service.inference_service_client import InferenceServiceClient

INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")


def _get_status_icon(status: str) -> str:
    """Return icon based on document processing status."""
    status_lower = status.lower()
    if "completed" in status_lower:
        return "✅"
    if "pending" in status_lower:
        return "⏳"
    return "❌"


def main():
    st.set_page_config(page_title="System Status", page_icon="⚙️", layout="centered")

    # Back to chat navigation
    st.page_link("streamlit_app.py", label="← Back to Chat", icon="💬")

    st.title("System Status")

    # Health Status Section
    st.header("Inference Service Health")

    client = InferenceServiceClient(INFERENCE_SERVICE_URL)

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Refresh"):
            st.rerun()

    health = client.get_health()

    if not health.is_healthy:
        st.error(health.error_message or "Inference service unavailable")
    else:
        st.success("Connected")
        st.metric("Documents in vector store", health.vector_store_count)

        if health.documents:
            st.subheader("Loaded Documents")
            for doc in health.documents:
                icon = _get_status_icon(doc.status)
                st.write(f"{icon} **{doc.doc_name}** — {doc.status}")
        else:
            st.info("No documents loaded yet")

    # Future: Document Ingestion UI
    st.divider()
    st.header("Document Ingestion")
    st.info("📄 Document upload and ingestion UI will be added here")


if __name__ == "__main__":
    main()
