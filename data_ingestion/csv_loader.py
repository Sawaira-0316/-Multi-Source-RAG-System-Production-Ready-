from pathlib import Path
from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document


def load_csvs(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    docs: List[Document] = []
    for csv_path in folder.glob("*.csv"):
        loader = CSVLoader(file_path=str(csv_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs
