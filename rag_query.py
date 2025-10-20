# rag_query.py
import os
from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from rag_common import active_version  # <-- new helper

DB_DIR = "store"

SYSTEM = (
    "You answer only from the provided context. "
    "If the answer is not in the context, say 'I don't know'. Be concise."
)

def _ensure_store_exists(version: str) -> str:
    """Return store/<version> or raise a clear error if missing."""
    path = os.path.join(DB_DIR, version)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Vector store not found: {path}\n"
            f"→ Build one with: python ragvc.py build {version}"
        )
    return path

def build_rag(k: int = 6, score_threshold: float = 0.2, version: str | None = None):
    """
    Build the runtime RAG components:
      - LLM (deterministic)
      - Retriever (with similarity score threshold)
      - Prompt
    Optional: override version (else use active).
    """
    ver = version or active_version()
    persist_dir = _ensure_store_exists(ver)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vs.as_retriever(
        search_type="similarity",  # ✅ simpler mode — no threshold
        search_kwargs={"k": k},  # ✅ top-k only
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Question: {q}\n\nContext:\n{ctx}")
    ])
    return llm, retriever, prompt

def ask(question: str, *, show_ctx: bool = False) -> str:
    """Return just the answer text. Set show_ctx=True to print retrieved chunks."""
    llm, retriever, prompt = build_rag()
    docs = retriever.invoke(question)

    if not docs:
        return "I don't know."

    if show_ctx:
        print("\n--- Retrieved chunks ---")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] source={d.metadata.get('source','?')}\n{d.page_content[:250]}...\n")

    ctx = "\n---\n".join(d.page_content for d in docs)
    msg = prompt.format_messages(q=question, ctx=ctx)
    return llm.invoke(msg).content

def ask_with_citations(question: str) -> Tuple[str, List[str]]:
    """
    Return (answer_text, citations) where citations is a list of 'source' strings.
    Useful for debugging and UI.
    """
    llm, retriever, prompt = build_rag()
    docs = retriever.invoke(question)
    if not docs:
        return "I don't know.", []

    ctx = "\n---\n".join(d.page_content for d in docs)
    msg = prompt.format_messages(q=question, ctx=ctx)
    answer = llm.invoke(msg).content
    cites = [d.metadata.get("source", "?") for d in docs]
    return answer, cites

if __name__ == "__main__":
    for q in [
        "What is RAG?",
        "How to run Playwright tests with pytest?",
        "How to enable headed mode in Playwright?",
        "What does temperature=0 do in OpenAI models?",
    ]:
        ans, cites = ask_with_citations(q)
        print(f"\nQ: {q}\nA: {ans}\nCitations: {cites}")
