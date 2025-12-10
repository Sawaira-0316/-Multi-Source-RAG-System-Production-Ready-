# embeddings/embedder.py
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # loads OPENAI_API_KEY from .env


def get_embedder(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Return an OpenAIEmbeddings instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

    return OpenAIEmbeddings(model=model, api_key=api_key)
