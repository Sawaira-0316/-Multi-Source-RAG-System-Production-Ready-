# E:\Multi-Source RAG System\llm\generator.py

from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


def _format_docs(docs: List[Document], max_chars: int = 6000) -> str:
    """
    Join retrieved docs into a single context string, truncated to max_chars.
    """
    chunks = []
    total = 0

    for i, d in enumerate(docs, start=1):
        text = d.page_content or ""
        snippet = f"[Source {i}]\n{text.strip()}\n\n"
        if total + len(snippet) > max_chars:
            break
        chunks.append(snippet)
        total += len(snippet)

    return "".join(chunks).strip()


def generate_answer(question: str, docs: List[Document]) -> Tuple[str, List[Document]]:
    """
    Use an LLM (ChatOpenAI) to generate an answer based on retrieved documents.

    Returns:
        answer: str - the model's answer
        used_docs: List[Document] - the same docs passed in (for UI display)
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Prepare context from retrieved documents
    context = _format_docs(docs)

    if not context:
        # No docs found; still let the LLM try to answer, but warn in the prompt
        context = "No relevant documents were retrieved. Answer based only on general knowledge."

    system_prompt = """
You are an assistant that answers questions using the provided context.
Always base your answer primarily on the context. If the context does not
contain the answer, say that it is not available in the provided documents.

Be concise, clear, and do not fabricate details that are not supported by the context.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # you can change to another OpenAI chat model if you like
        temperature=0.2,
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = result.content if hasattr(result, "content") else str(result)
    return answer, docs
