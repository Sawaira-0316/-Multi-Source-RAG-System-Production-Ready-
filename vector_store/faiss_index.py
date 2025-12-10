# vector_store/faiss_index.py
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_faiss_index(
    docs: List[Document],
    embedder: OpenAIEmbeddings,
) -> FAISS:
    if not docs:
        raise ValueError("No documents provided to build the FAISS index.")
    return FAISS.from_documents(docs, embedder)


def save_faiss_index(
    vectorstore: FAISS,
    index_dir: str,
) -> None:
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_faiss_index(
    index_dir: str,
    embedder: OpenAIEmbeddings,
) -> FAISS:
    path = Path(index_dir)
    if not path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    return FAISS.load_local(
        str(path),
        embedder,
        allow_dangerous_deserialization=True,
    )
