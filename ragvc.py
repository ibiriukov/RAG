# ragvc.py — Tiny RAG Embedding Version Control
# -----------------------------------------------------------
# This script lets you:
#   1. Build different versions of embeddings (emb_v1, emb_v2, etc.)
#   2. Switch between them
#   3. Check which version is active
#   4. Optionally evaluate them with RAGAS metrics
# -----------------------------------------------------------

# 🧰 Built-in Python modules
import os, json, time, glob, hashlib, argparse

# 🧠 LangChain + OpenAI tools
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 📄 Document loaders for PDFs and text files
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader

# ✂️ Text splitter — breaks long text into overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 💾 Vector database — stores and retrieves embeddings
from langchain_chroma import Chroma

# 🧪 Optional: RAGAS evaluation (used only if installed)
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, answer_correctness
    RAGAS_OK = True
except Exception:
    RAGAS_OK = False


# -----------------------------------------------------------
# 🔧 Configuration — Folder names
# -----------------------------------------------------------
DATA_DIR = "data"    # where your .txt/.md/.pdf files live
STORE_DIR = "store"  # where all embedding versions are stored


# -----------------------------------------------------------
# 🧭 Helper 1 — Find which version of embeddings is active
# -----------------------------------------------------------
def active_version(default="emb_v1"):
    """
    Find the current active version.
    Priority order:
      1. Environment variable EMB_VERSION
      2. store/current.txt file
      3. Default fallback ("emb_v1")
    """
    env = os.getenv("EMB_VERSION")  # check environment variable
    if env:
        return env
    try:
        # read store/current.txt if it exists
        return open(os.path.join(STORE_DIR, "current.txt")).read().strip()
    except FileNotFoundError:
        # if not found, use default
        return default


# -----------------------------------------------------------
# 🔒 Helper 2 — Generate a unique hash of your /data folder
# -----------------------------------------------------------
def corpus_hash(path=DATA_DIR):
    """
    Create a short fingerprint (hash) of all files in /data.
    Helps detect if files changed between versions.
    """
    h = hashlib.sha256()
    # go through all files in /data recursively
    for p in sorted(glob.glob(f"{path}/**/*", recursive=True)):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read())  # update the hash with file content
    return h.hexdigest()[:16]  # return first 16 characters


# -----------------------------------------------------------
# 📚 Helper 3 — Load documents from /data
# -----------------------------------------------------------
def load_docs():
    """
    Load all supported documents (PDF, TXT, MD) from /data
    and return them as LangChain Document objects.
    """
    docs = []

    # If there are PDFs, load them all
    if glob.glob(f"{DATA_DIR}/*.pdf"):
        docs += PyPDFDirectoryLoader(DATA_DIR).load()

    # Load all .txt and .md files
    for p in glob.glob(f"{DATA_DIR}/*"):
        if p.endswith((".txt", ".md")):
            docs += TextLoader(p, encoding="utf-8").load()

    return docs


# -----------------------------------------------------------
# 🏗️  MAIN FUNCTION — Build a new embedding version
# -----------------------------------------------------------
def build(version: str):
    """
    Create a new embedding database (e.g., emb_v2)
    from all documents inside /data.
    """
    # 1️⃣ Ensure output folders exist
    os.makedirs(STORE_DIR, exist_ok=True)
    db_dir = os.path.join(STORE_DIR, version)
    os.makedirs(db_dir, exist_ok=True)

    # 2️⃣ Load all source documents
    docs = load_docs()
    if not docs:
        raise SystemExit("No documents in ./data")

    # 3️⃣ Split documents into smaller chunks for embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # each chunk ~800 characters
        chunk_overlap=120 # small overlap to preserve context
    )
    chunks = splitter.split_documents(docs)

    # 4️⃣ Create embeddings using OpenAI's text-embedding-3-small model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5️⃣ Build a Chroma vector store from those chunks and save it
    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )

    # 6️⃣ Save metadata info for tracking
    meta = {
        "version": version,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_hash": corpus_hash(),
        "splitter": {"type": "RecursiveCharacter", "chunk_size": 800, "overlap": 120},
        "embedding": {"model": "text-embedding-3-small", "dim": 1536},
        "notes": "built via ragvc.py",
    }
    with open(os.path.join(db_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 7️⃣ Done! Print summary
    print(f"✅ Built {version} with {len(chunks)} chunks → {db_dir}")


# -----------------------------------------------------------
# 🔄  Switch active version
# -----------------------------------------------------------
def use(version: str):
    """
    Mark a version as 'active' so rag_query.py uses it by default.
    """
    os.makedirs(STORE_DIR, exist_ok=True)

    # Check that the version actually exists
    if not os.path.isdir(os.path.join(STORE_DIR, version)):
        raise SystemExit(f"{version} not found in {STORE_DIR}/")

    # Write to store/current.txt
    with open(os.path.join(STORE_DIR, "current.txt"), "w") as f:
        f.write(version)

    print(f"✅ Active version set to {version}")


# -----------------------------------------------------------
# 📋  Show current and available versions
# -----------------------------------------------------------
def status():
    """
    Print the currently active version and all available ones.
    """
    avail = [d for d in os.listdir(STORE_DIR)
             if os.path.isdir(os.path.join(STORE_DIR, d))]
    curr = active_version()
    print("Active:", curr)
    print("Available:", ", ".join(sorted(avail)) or "(none)")


# -----------------------------------------------------------
# 🧪  Mini evaluation with RAGAS (optional)
# -----------------------------------------------------------
def eval_small(version: str):
    """
    Quick self-test using the RAGAS framework.
    Compares retrieved answers to small reference samples.
    """
    # 1️⃣ Skip if RAGAS not installed
    if not RAGAS_OK:
        print("RAGAS not installed; skipping eval. (pip install ragas datasets)")
        return

    # 2️⃣ Define a couple of mini test questions + references
    QUESTIONS = [
        "What is RAG?",
        "What does temperature=0 do in OpenAI models?",
    ]
    REFS = [
        "RAG stands for Retrieval-Augmented Generation and grounds answers in retrieved context.",
        "temperature=0 makes output deterministic (low randomness).",
    ]

    # 3️⃣ Load vector store and LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    vs = Chroma(
        persist_directory=os.path.join(STORE_DIR, version),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    # 4️⃣ Run the small QA loop
    answers, contexts = [], []
    for q in QUESTIONS:
        docs = retriever.invoke(q)               # retrieve context chunks
        ctx = "\n---\n".join(d.page_content for d in docs)  # combine them
        msg = [
            {"role": "system", "content": "Answer only from the provided context."},
            {"role": "user", "content": f"Question: {q}\n\nContext:\n{ctx}"}
        ]
        answers.append(llm.invoke(msg).content)  # model answer
        contexts.append([d.page_content for d in docs])  # save context text

    # 5️⃣ Build dataset for evaluation
    ds = Dataset.from_dict({
        "question": QUESTIONS,
        "answer": answers,
        "contexts": contexts,
        "reference": REFS
    })

    # 6️⃣ Evaluate with RAGAS metrics
    results = evaluate(ds, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        answer_correctness
    ])

    # 7️⃣ Average all metric results
    row = {
        "version": version,
        "faithfulness": float(sum(results["faithfulness"]) / len(results["faithfulness"])),
        "answer_relevancy": float(sum(results["answer_relevancy"]) / len(results["answer_relevancy"])),
        "context_precision": float(sum(results["context_precision"]) / len(results["context_precision"])),
        "answer_correctness": float(sum(results["answer_correctness"]) / len(results["answer_correctness"])),
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 8️⃣ Save result to eval_registry.jsonl (append mode)
    with open(os.path.join(STORE_DIR, "..", "eval_registry.jsonl").replace("\\", "/"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print("✅ RAGAS:", row)


# -----------------------------------------------------------
# 🧠  Command-line interface (CLI)
# -----------------------------------------------------------
if __name__ == "__main__":
    # Create a parser for commands like "build", "use", etc.
    ap = argparse.ArgumentParser(description="Tiny RAG embedding version control")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # "build" command
    b = sub.add_parser("build", help="build an index into store/<version>")
    b.add_argument("version", help="e.g., emb_v3")

    # "use" command
    u = sub.add_parser("use", help="mark store/<version> active")
    u.add_argument("version")

    # "status" command
    s = sub.add_parser("status", help="show active + available versions")

    # "eval" command
    e = sub.add_parser("eval", help="quick RAGAS eval for a version")
    e.add_argument("version")

    # Parse and run the selected command
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.version)
    elif args.cmd == "use":
        use(args.version)
    elif args.cmd == "status":
        status()
    elif args.cmd == "eval":
        eval_small(args.version)
