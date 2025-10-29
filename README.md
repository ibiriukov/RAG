# RAG Pipeline with LangChain, OpenAI, and Chroma

> **Overview**  
Lightweight RAG evaluation framework built with LangChain, OpenAI, and Chroma. Measures grounding and factual integrity of model answers.
>
> **Highlights**
- Retrieves context and evaluates LLM answers using RAGAS
- Asserts faithfulness, context precision, and correctness in PyTest
- Reproducible runs via environment variables and fixed parameters
- Structured for clean CI/CD adoption

<details>
  <summary>📄 Full Technical Overview</summary>

## Tech Stack

- **LangChain** — chains LLM and retrieval components  
- **OpenAI** — powers embedding generation and LLM responses  
- **Chroma** — provides fast, local vector storage and retrieval  
- **Pytest** — used for evaluation and test orchestration  
- **RAGAS** — delivers retrieval quality metrics  
- **Environment config** — managed via `.env` and `os.getenv`  
- **Custom logging/assertions** — improves traceability and debugging  

## Document Coverage

- `langchain_cheatsheet.txt` — LangChain patterns and chaining logic  
- `llm_testing_basics.txt` — evaluation strategies and metrics  
- `openai_api_notes.txt` — API usage tips and rate limits  
- `playwright_tips.txt` — UI automation insights  
- `pytest_examples.txt` — idiomatic Pytest usage and flags  
- `rag_concepts.txt` — chunking, retrieval, and generation  
- `rag_version_control.txt` — embedding reproducibility  

## CI/CD Readiness

> Jenkins integration is **not yet implemented**, but the project is structured for easy CI/CD adoption.

- Modular scripts support parameterized execution  
- `.env` secrets managed securely from the project root  
- Embedding version control via `ragvc.py` enables reproducible builds  
- Pytest-based evaluation supports automated test runs  
- Future CI/CD pipeline will include:  
  - Triggered ingestion and query jobs on commit  
  - Environment-specific test runs  
  - HTML report generation and artifact archiving  

## Evaluation & Reporting

- `test_rag_eval.py` uses [RAGAS](https://github.com/explodinggradients/ragas) to evaluate:  
  - **Faithfulness** — Is the answer grounded in the retrieved context?  
  - **Answer Relevancy** — Is the answer relevant to the question?  
  - **Context Precision** — Are the retrieved chunks relevant to the question?  
  - **Answer Correctness** — Does the answer match the reference answer?  
- Evaluation runs on a Hugging Face `Dataset` with:  
  - Sample questions  
  - Generated answers  
  - Retrieved contexts  
  - Ground-truth reference answers  
- Detailed per-question output includes:  
  - Answer vs. reference comparison  
  - Context chunk previews  
  - Metric scores (rounded to 2 decimals)  
- Minimum thresholds enforced via `assert`:  
  - Faithfulness ≥ 0.70  
  - Answer Relevancy ≥ 0.70  
  - Context Precision ≥ 0.55  
  - Answer Correctness ≥ 0.70  

## 📁 Project Structure

- `.venv/` — Virtual environment  
- `data/` — Source documents  
  - `langchain_cheatsheet.txt`  
  - `llm_testing_basics.txt`  
  - `openai_api_notes.txt`  
  - `playwright_tips.txt`  
  - `pytest_examples.txt`  
  - `rag_concepts.txt`  
  - `rag_version_control.txt`  
- `store/` — Vector store versions  
  - `emb_v1/`  
  - `emb_v2/`  
  - `current.txt`  
- `tests/` — Pytest test cases  
  - `test_rag_eval.py`  
- `.env` — API keys and config (root-level)  
- `.gitignore` — Git exclusions  
- `ingest.py` — Document ingestion pipeline  
- `rag_query.py` — Query interface  
- `rag_common.py` — Shared utilities  
- `ragvc.py` — Embedding version control  
- `sanity_check.py` — Quick validation script  
- `README.md` — Project overview  

## 👨‍💻 Author

**Ievgen** — Results-driven engineer focused on scalable LLM pipelines, test automation, and CI/CD best practices. Passionate about clarity, modularity, and production-grade architecture.

</details>
