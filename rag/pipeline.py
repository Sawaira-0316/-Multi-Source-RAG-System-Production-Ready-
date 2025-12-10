from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from data_ingestion.universal_loader import load_corpus_from_paths
from data_ingestion.web_loader import load_web_pages
from embeddings.embedder import get_embedder
from vector_store.faiss_index import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
)
from llm.generator import generate_answer


INDEX_DIR = str(Path("artifacts") / "faiss_index")


def ingest_data(
    paths: List[str],
    urls: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load documents from:
    - Local files / folders
    - Web URLs
    """
    docs: List[Document] = []

    if paths:
        docs.extend(load_corpus_from_paths(paths))

    if urls:
        docs.extend(load_web_pages(urls))

    return docs


def build_and_save_index(
    paths: List[str],
    urls: Optional[List[str]] = None,
) -> int:
    """
    Ingest everything, build FAISS index, save to disk.

    Returns:
        int: Number of documents indexed.
    """
    docs = ingest_data(paths, urls)
    if not docs:
        raise ValueError("No documents found to index. Check your paths/URLs.")

    embedder = get_embedder()
    vectorstore = build_faiss_index(docs, embedder)
    save_faiss_index(vectorstore, INDEX_DIR)
    return len(docs)


def answer_question(question: str) -> Tuple[str, List[Document]]:
    """
    Load the FAISS index, retrieve relevant documents, and generate an answer.

    Returns:
        (answer: str, source_docs: List[Document])
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    embedder = get_embedder()
    vectorstore = load_faiss_index(INDEX_DIR, embedder)

    # ✅ Directly use similarity_search (NO retriever object, NO get_relevant_documents)
    docs = vectorstore.similarity_search(question, k=6)

    answer, used_docs = generate_answer(question, docs)
    return answer, used_docs
