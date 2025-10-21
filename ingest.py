# ingest.py
# Purpose: load docs from /data → chunk → embed → persist to Chroma (/store)

# Built-in: work with folders/files
import os

# OpenAI embeddings (turn text into vectors)
from langchain_openai import OpenAIEmbeddings

# Loaders: PDFs and plain text/Markdown
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader

# Split long docs into overlapping chunks (better retrieval)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Modern Chroma import (no deprecation warning)
from langchain_chroma import Chroma

# Folder with your raw documents
DATA_DIR = "data"

# Folder where Chroma DB will be saved
DB_DIR = "store"

# Load all supported docs from DATA_DIR
def load_docs():
    # Start an empty list to collect Document objects
    docs = []

    # Guard: ensure /data exists
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    # If there are PDFs, load them all
    if any(p.lower().endswith(".pdf") for p in os.listdir(DATA_DIR)):
        docs += PyPDFDirectoryLoader(DATA_DIR).load()

    # Load .txt and .md files one by one
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith((".txt", ".md")):
            path = os.path.join(DATA_DIR, name)
            docs += TextLoader(path, encoding="utf-8").load()

    # Guard: empty corpus check
    if not docs:
        raise ValueError(
            f"No documents found in {DATA_DIR}. "
            f"Add .txt/.md (and/or .pdf) files and run again."
        )

    # Return the list of Document objects
    return docs

# Main pipeline: chunk → embed → persist
def main():
    # 1) Load raw documents
    docs = load_docs()

    # 2) Create a splitter (good defaults for Q&A)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # ~500 chars per chunk
        chunk_overlap=100,   # keep context continuity
    )

    chunks = splitter.split_documents(docs)

    for d in chunks:
        src = d.metadata.get("source") or d.metadata.get("file_path")
        d.metadata["source"] = src or "data"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    unique_sources = {c.metadata.get("source") for c in chunks}
    print(f"✅ Ingested {len(chunks)} chunks from {len(unique_sources)} files into {DB_DIR}")
    print("   Sources:", ", ".join(sorted(s for s in unique_sources if s)))

if __name__ == "__main__":
    main()
