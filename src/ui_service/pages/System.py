"""Streamlit page displaying the inference service system status and loaded documents."""

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


st.title("System Status")

# Health Status Section
st.header("Inference Service Health")

# Custom CSS for vertical alignment
st.markdown(
    """
    <style>
    div[data-testid="column"] {
        display: flex;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def _get_client() -> InferenceServiceClient:
    """Return a cached InferenceServiceClient instance."""
    return InferenceServiceClient(INFERENCE_SERVICE_URL)


client = _get_client()

health = client.get_health()

col1, col2 = st.columns([4, 1])
with col1:
    if not health.is_healthy:
        st.error(health.error_message or "Inference service unavailable")
    else:
        st.markdown(
            '<p style="color: green; margin: 0; padding: 8px 0;">✓ Connected</p>',
            unsafe_allow_html=True,
        )
with col2:
    if st.button("Refresh"):
        st.rerun()

if health.is_healthy:
    st.metric("Documents in vector store", health.vector_store_count)

    if health.documents:
        st.subheader("Loaded Documents")
        for doc in health.documents:
            icon = _get_status_icon(doc.status)
            st.write(f"{icon} **{doc.doc_name}** — {doc.status}")
    else:
        st.info("No documents loaded yet")

# Future: Document Ingestion UI
# st.divider()
# st.header("Document Ingestion")
# st.info("📄 Document upload and ingestion UI will be added here")
