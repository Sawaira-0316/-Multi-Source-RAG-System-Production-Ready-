# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import MultiSourceRAGPipeline

app = FastAPI(title="Multi-Source RAG API")

# Create one global pipeline instance
pipeline = MultiSourceRAGPipeline()

# Build the index ONCE when server starts
# In real app: load paths from config / env
pdfs = ["./data/example.pdf"]
urls = ["https://example.com"]
csv_path = "./data/example.csv"
csv_cols = ["title", "description"]

pipeline.build_index(pdf_paths=pdfs, urls=urls, csv_path=csv_path, csv_text_columns=csv_cols)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_rag(req: QueryRequest):
    """
    Accepts a user query and returns an RAG answer.

    WHY:
    - This lets your UI, mobile app, or other services call the RAG system via HTTP.
    """
    result = pipeline.answer(req.query)
    return result
