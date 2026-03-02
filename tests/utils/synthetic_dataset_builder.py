import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.shared.env_loader import load_environment
from tests.utils.evals_utils import build_provider_llm

logger = logging.getLogger(__name__)

load_environment()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)


def _extract_page_number(metadata: dict) -> int | None:
    """Normalize the loader's page metadata to 1-based page numbers."""
    if not metadata:
        return None
    raw_page = metadata.get("page")
    if raw_page is None:
        raw_page = metadata.get("page_number")
    if raw_page is None:
        return None
    try:
        raw_int = int(raw_page)
    except (TypeError, ValueError):
        return None
    if raw_int < 0:
        return None
    return raw_int + 1  # PyMuPDFLoader is 0-based; force 1-based for consistency


def load_and_filter_pages(
    pdf_path: str, start_page: int, end_page: int
) -> List[Document]:
    if start_page < 1 or end_page < start_page:
        raise ValueError("start_page must be >= 1 and end_page must be >= start_page")

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    logger.info("Loaded %s pages from %s", len(docs), pdf_path)

    filtered: List[Document] = []
    for doc in docs:
        page_number = _extract_page_number(doc.metadata)
        if page_number is None:
            continue
        if start_page <= page_number <= end_page:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["page_number"] = page_number
            doc.metadata["start_page"] = page_number
            doc.metadata["end_page"] = page_number
            filtered.append(doc)

    filtered.sort(key=lambda d: d.metadata.get("page_number", 0))
    logger.info("Kept %s pages in range [%s, %s]", len(filtered), start_page, end_page)
    return filtered


def chunk_documents_with_page_ranges(
    docs: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = splitter.split_documents(list(docs))

    for chunk in chunked_docs:
        start_page = chunk.metadata.get("start_page") or _extract_page_number(
            chunk.metadata
        )
        end_page = chunk.metadata.get("end_page") or start_page

        if isinstance(start_page, list):
            start_page = min(start_page)
        if isinstance(end_page, list):
            end_page = max(end_page)

        chunk.metadata["start_page"] = start_page
        chunk.metadata["end_page"] = end_page

    logger.info("Produced %s chunks", len(chunked_docs))
    return chunked_docs


def _normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question).strip()
    collapsed = re.sub(r"[\s]*[\?\.!:,;]+$", "", collapsed)
    return collapsed.lower()


def dedupe_questions(questions: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for question in questions:
        normalized = _normalize_question(question)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(question.strip())
    return unique


def _parse_question_list(raw_text: str) -> List[str]:
    cleaned = raw_text.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list of questions")

    questions: List[str] = [
        q.strip() for q in parsed if isinstance(q, str) and q.strip()
    ]
    return questions


def generate_questions_for_chunk(
    chunk: Document,
    llm,
    questions_per_chunk: int,
    max_retries: int = 1,
) -> List[str]:
    base_prompt = (
        "You are creating evaluation questions for a QA system. "
        "Only use the provided chunk content. "
        "Generate {n} diverse questions that can be answered using the chunk alone. "
        "If the chunk lacks enough information, respond with an empty JSON list []. "
        'Return JSON only in this exact format: ["question1", "question2", ...]. '
        "Do not add any commentary or keys.\n\n"
        "Chunk (pages {start_page}-{end_page}):\n{chunk}\n"
    ).format(
        n=questions_per_chunk,
        start_page=chunk.metadata.get("start_page", "?"),
        end_page=chunk.metadata.get("end_page", "?"),
        chunk=chunk.page_content,
    )

    prompt = base_prompt
    for attempt in range(max_retries + 1):
        response = llm.invoke(prompt)
        raw_text = str(response)
        try:
            questions = _parse_question_list(raw_text)
            if not questions_per_chunk:
                return questions
            return questions[:questions_per_chunk]
        except Exception:
            if attempt >= max_retries:
                logger.warning(
                    "Failed to parse questions after %s attempts", attempt + 1
                )
                break
            prompt = (
                "The previous output was not valid JSON. "
                "Respond again with ONLY a JSON list of strings and no commentary.\n\n"
                f"{base_prompt}"
            )
    return []


def build_retriever(chunks: Sequence[Document], top_k: int):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(list(chunks), embeddings)
    return vectordb.as_retriever(search_kwargs={"k": top_k})


def answer_question_with_context(question: str, retriever, llm, top_k: int) -> str:
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "INSUFFICIENT_CONTEXT"

    contexts = []
    for doc in docs[:top_k]:
        start_page = doc.metadata.get("start_page") or _extract_page_number(
            doc.metadata
        )
        end_page = doc.metadata.get("end_page") or start_page
        header = f"[pages {start_page}-{end_page}] " if start_page and end_page else ""
        contexts.append(f"{header}{doc.page_content}")

    context_block = "\n\n".join(contexts)
    prompt = (
        "You are answering a question using ONLY the provided context snippets. "
        'If the context is insufficient, reply exactly with "INSUFFICIENT_CONTEXT". '
        "Do not add citations or commentary.\n\n"
        f"Context snippets (top {len(contexts)}):\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = llm.invoke(prompt)
    answer = str(response).strip()
    return answer or "INSUFFICIENT_CONTEXT"


def build_synthetic_dataset_from_pdf(
    pdf_path: str,
    start_page: int,
    end_page: int,
    questions_per_chunk: int = 4,
    max_questions: int = 60,
    chunk_size: int = 6000,
    chunk_overlap: int = 800,
    top_k: int = 3,
    seed: int | None = None,
    return_debug: bool = False,
) -> list[dict]:
    """
    Generate a synthetic QA dataset (question + ground_truth) from a PDF slice.

    Returns:
        list of {"question": str, "ground_truth": str}
    """
    if seed is not None:
        random.seed(seed)

    pages = load_and_filter_pages(pdf_path, start_page, end_page)
    if not pages:
        logger.warning(
            "No pages found in the specified range; returning empty dataset."
        )
        return [] if not return_debug else ([], {"pages": 0, "chunks": 0})

    chunks = chunk_documents_with_page_ranges(pages, chunk_size, chunk_overlap)
    if not chunks:
        logger.warning("No chunks produced; returning empty dataset.")
        return [] if not return_debug else ([], {"pages": len(pages), "chunks": 0})

    llm = build_provider_llm()

    raw_questions: List[str] = []
    for chunk in chunks:
        questions = generate_questions_for_chunk(
            chunk, llm, questions_per_chunk=questions_per_chunk
        )
        raw_questions.extend(questions)
    logger.info("Generated %s raw questions", len(raw_questions))

    unique_questions = dedupe_questions(raw_questions)
    logger.info("Questions after dedupe: %s", len(unique_questions))

    if max_questions and len(unique_questions) > max_questions:
        unique_questions = unique_questions[:max_questions]
        logger.info("Trimmed to max_questions=%s", max_questions)

    retriever = build_retriever(chunks, top_k=top_k)

    qa_pairs: List[dict] = []
    for question in unique_questions:
        answer = answer_question_with_context(question, retriever, llm, top_k=top_k)
        if answer and answer != "INSUFFICIENT_CONTEXT":
            qa_pairs.append({"question": question, "ground_truth": answer})

    logger.info("Final QA pairs: %s", len(qa_pairs))
    if return_debug:
        debug_info = {
            "pages_loaded": len(pages),
            "chunks_produced": len(chunks),
            "raw_questions": len(raw_questions),
            "unique_questions": len(unique_questions),
            "qa_pairs": len(qa_pairs),
        }
        return qa_pairs, debug_info
    return qa_pairs


def example_usage() -> None:
    """Minimal example to build and persist a dataset to JSON."""
    pdf_path = "data/your_pdf.pdf"
    output_path = Path("tests/artifacts/synthetic_dataset.json")

    dataset = build_synthetic_dataset_from_pdf(
        pdf_path=pdf_path,
        start_page=18,
        end_page=70,
        questions_per_chunk=3,
        max_questions=1,
        chunk_size=2000,
        chunk_overlap=200,
        top_k=3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s QA pairs to %s", len(dataset), output_path)


if __name__ == "__main__":
    example_usage()
