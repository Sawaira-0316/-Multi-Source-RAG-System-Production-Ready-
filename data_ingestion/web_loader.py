# data_ingestion/web_loader.py

from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def load_web_pages(urls: List[str]) -> List[Document]:
    docs: List[Document] = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": url, "type": "web"},
                )
            )
        except Exception:
            # skip failed URLs
            continue

    return docs
