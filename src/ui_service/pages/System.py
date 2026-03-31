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
    # Custom HTML metric with data-testid, styled to match Streamlit's metric component
    st.markdown(
        f"""
        <div style="
            background-color: transparent;
            padding: 1rem 0;
        ">
            <div style="
                font-size: 0.875rem;
                font-weight: 400;
                color: var(--text-color, rgba(250, 250, 250, 0.6));
                margin-bottom: 0.25rem;
            ">Documents in vector store</div>
            <div data-testid="documents_in_vector_store_count"
                style="
                font-size: 2.5rem;
                font-weight: 600;
                line-height: 1.2;
                color: var(--text-color, rgb(250, 250, 250));
            ">{health.vector_store_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if health.documents:
        st.subheader("Loaded Documents")
        # Build HTML for documents list with data-testid
        docs_html = '<div data-testid="loaded_documents_list">'
        for doc in health.documents:
            icon = _get_status_icon(doc.status)
            docs_html += f"<p>{icon} <strong>{doc.doc_name}</strong> — {doc.status}</p>"
        docs_html += "</div>"
        st.markdown(docs_html, unsafe_allow_html=True)
    else:
        st.info("No documents loaded yet")

# Future: Document Ingestion UI
# st.divider()
# st.header("Document Ingestion")
# st.info("📄 Document upload and ingestion UI will be added here")
