# data_ingestion/pdf_loader.py
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document



def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load one or more PDF files and return them as LangChain Document objects.

    WHY:
    - We want a unified format (Document) so later steps (embeddings, vector store)
      don't care if data came from PDF, web, or CSV.
    """
    docs: List[Document] = []

    for path_str in pdf_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[PDF LOADER] File not found: {path}")
            continue

        loader = PyPDFLoader(str(path))
        pdf_docs = loader.load()
        docs.extend(pdf_docs)

    return docs
