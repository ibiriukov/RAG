from dotenv import load_dotenv
import os
load_dotenv()  # ✅ Load .env before any os.getenv calls

STORE_DIR = "store"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")


def active_version(default: str = "emb_v1") -> str:
    """
    Returns the active embedding version.
    Priority: EMB_VERSION env var > store/current.txt > default.
    """
    env = os.getenv("EMB_VERSION")
    if env:
        return env
    try:
        with open(os.path.join(STORE_DIR, "current.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return default
