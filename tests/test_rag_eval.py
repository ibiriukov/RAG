# Import built-in modules for working with file paths and system settings
import os, sys

# Add the parent directory to Python's module search path so we can import local files like rag_query.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Hugging Face's Dataset class to organize data like questions, answers, and references
from datasets import Dataset

# Import the main evaluation function from RAGAS
from ragas import evaluate

# Import specific evaluation metrics from RAGAS
from ragas.metrics import (
    faithfulness,        # Measures if the answer is grounded in the retrieved context
    answer_relevancy,    # Measures if the answer is relevant to the question
    context_precision,   # Measures if the retrieved context is relevant to the question
    answer_correctness,  # Measures if the answer matches the reference answer
)

# Import a custom function that sets up the RAG pipeline (retriever, LLM, and prompt)
from rag_query import build_rag


# Define a function that runs RAG for one question and returns the answer and context
def answer_with_context(q: str):
    # Initialize the language model, retriever, and prompt template
    llm, retriever, prompt = build_rag()

    # Use the retriever to find relevant documents for the question
    docs = retriever.invoke(q)

    # Extract the text content from each retrieved document
    # will contain different chunks,
    ctxs = [d.page_content for d in docs]

    # Format the prompt with the question and combined context
    msg = prompt.format_messages(q=q, ctx="\n\n".join(ctxs))

    # Send the prompt to the language model and get the generated answer
    ans = llm.invoke(msg).content

    # Return both the answer and the list of context documents
    return ans, ctxs


# Define a helper function to normalize metric scores into a single float
def _score(results_dict, key: str) -> float:
    # Get the value for the specified metric key
    v = results_dict[key]
    #  [0.92, 0.88, 0.95, 0.90]

    # If the value is a list or tuple (multiple scores), calculate the average
    if isinstance(v, (list, tuple)): # is a type check in Python. It asks:
        return sum(map(float, v)) / max(1, len(v))
    """
     max(1, len(v) To prevent division by zero when calculating an average:
        If v is empty:
        - len(v) would be 0
        - So max(1, 0) → 1
        - Result: 0 / 1 = 0.0 (safe)
  
     sum() function adds up all the elements in a list
     0.92+ 0.88+ 0.95+ 0.90

     map(float, v)
    - Converts every item in v to a float.
    - Ensures that even if the list contains strings like "0.92", they become 0.92.

    
    """

    # If it's already a single number, just return it as a float
    return float(v)


# Define the main test function to evaluate RAG performance
def test_rag_metrics():
    # List of sample questions to test the RAG system
    questions = [
        "What is RAG?",
        "How to run Playwright tests with pytest?",
        "How to enable headed mode in Playwright?",
        "What does temperature=0 do in OpenAI models?",
    ]


    # Ground-truth answers for each question, used for evaluation
    references = [
        "RAG stands for Retrieval-Augmented Generation. It retrieves relevant chunks from a knowledge store and lets a generator (LLM) answer using those chunks to improve factual accuracy.",
        "Run Playwright tests with pytest via: `pytest -v`. To select a browser and headed mode in PowerShell: `$env:BROWSER=\"firefox\"; $env:HEADED=\"true\"; pytest -v`.",
        "Enable headed mode by setting `HEADED=true`, e.g., `$env:BROWSER=\"firefox\"; $env:HEADED=\"true\"; pytest -v`.",
        "Temperature controls randomness. `temperature=0` makes responses deterministic (repeatable). Higher values increase variety.",
    ]

    # Initialize empty lists to store answers and contexts
    answers, contexts = [], []

    # Loop through each question and run RAG to get the answer and context
    """
    - You run your RAG pipeline for each question.
        - You collect:
        - a: the generated answer
        - ctx: the list of retrieved context chunks
        - You store them in answers and contexts.
    """
    for q in questions:
        a, ctx = answer_with_context(q)  # Run RAG for the question
        answers.append(a)                # Save the generated answer
        contexts.append(ctx)             # Save the retrieved context

    # Create a Hugging Face dataset with questions, answers, contexts, and references
    """
    What Dataset.from_dict() Expects
      It expects a dictionary where:
    - Each key is a column name (like "question")
    - Each value is a list of values for that column
    Example:
        {
      "question": ["What is RAG?", "How to run Playwright...", ...],
      "answer": ["RAG stands for...", "`pytest -v`...", ...],
      "contexts": [[chunk1, chunk2, ...], [chunk1, chunk2, ...], ...],
      "reference": ["RAG stands for...", "`pytest -v`...", ...]
        }
    """
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    })

    # Run RAGAS evaluation on the dataset using selected metrics
    """
    This line calls the evaluate() function from the RAGAS library. Here's what happens:
    You specify 4 evaluation metrics:
     faithfulness -  Is the answer grounded in the retrieved context? (no hallucinations)
     answer_relevancy - Is the answer relevant to the question?
     context_precision - Are the retrieved chunks relevant to the question?
     answer_correctness - Does the answer match the reference answer in meaning?
     
     Result is dic
         {
      "faithfulness": [0.92, 0.88, 0.95, 0.90],
      "answer_relevancy": [0.91, 0.89, 0.93, 0.87],
      "context_precision": [0.78, 0.81, 0.76, 0.80],
      "answer_correctness": [0.90, 0.85, 0.88, 0.86]
         }

    """
    results = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
    )

    # 🔍 Print detailed output per question
    for i in range(len(questions)):
        print(f"\n--- Question {i + 1} ---")
        print(f"Q: {questions[i]}")
        print(f"Generated Answer:\n{answers[i]}")
        print(f"Reference Answer:\n{references[i]}")
        print("Retrieved Contexts:")
        for j, chunk in enumerate(contexts[i]):
            print(f"  Chunk {j + 1}: {chunk[:200]}...")  # Print first 200 chars
        print("Scores:")
        print(f"  Faithfulness:      {results['faithfulness'][i]:.2f}")
        print(f"  Answer Relevancy:  {results['answer_relevancy'][i]:.2f}")
        print(f"  Context Precision: {results['context_precision'][i]:.2f}")
        print(f"  Answer Correctness:{results['answer_correctness'][i]:.2f}")

    # Print the evaluation results to the console
    print("\nRagas:", results)

    # Set minimum acceptable scores for each metric and fail the test if any are too low
    assert _score(results, "faithfulness") >= 0.70
    assert _score(results, "answer_relevancy") >= 0.70
    assert _score(results, "context_precision") >= 0.55
    assert _score(results, "answer_correctness") >= 0.70
