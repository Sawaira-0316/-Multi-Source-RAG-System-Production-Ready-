# llm/generator.py
import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()


class AnswerGenerator:
    """
    Uses an LLM (ChatOpenAI) to synthesize an answer from retrieved docs.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    def build_prompt(self, query: str, docs: List[Document]) -> str:
        context = ""
        for i, doc in enumerate(docs, start=1):
            context += f"\n[DOC {i}]\n{doc.page_content[:1500]}\n"

        prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you don't know.

Question:
{query}

Context:
{context}

Include short references like [DOC 1], [DOC 2] where relevant.
"""
        return prompt

    def generate_answer(self, query: str, docs: List[Document]) -> str:
        prompt = self.build_prompt(query, docs)
        response = self.llm.invoke(prompt)
        return response.content


def generate_answer(query: str, docs: List[Document]) -> Tuple[str, List[Document]]:
    """
    Function-level API used by rag.pipeline.
    """
    generator = AnswerGenerator()
    answer = generator.generate_answer(query, docs)
    return answer, docs
