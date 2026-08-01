# L2-3.2 — RAG System Evaluation

**Level:** Practitioner
**Duration:** 1.5 hours

## Overview

Retrieval-Augmented Generation (RAG) systems combine a retrieval step (finding relevant documents) with a generation step (producing an answer from context). Evaluating RAG requires measuring both halves independently: did we retrieve the right documents, and did we generate a faithful, correct answer? This lesson builds a simple RAG system and evaluates it using custom scorers in `mlflow.genai.evaluate()`, then compares two retrieval strategies side by side.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L2-M3.1 (Custom Metrics)
- MLFlow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why RAG Evaluation Is Different

A standard LLM evaluation asks "is the answer correct?" RAG adds two more questions:

1. **Retrieval quality** -- Did we find the right documents from the knowledge base?
2. **Generation quality** -- Given the retrieved context, did the LLM produce a faithful and correct answer?

These two dimensions are largely independent. A system can retrieve the perfect documents but generate a poor answer (bad generation), or generate a plausible answer from irrelevant documents (bad retrieval, hallucinated generation).

### Key Metrics

| Metric | Dimension | What It Measures |
|--------|-----------|-----------------|
| Retrieval Precision | Retrieval | Fraction of retrieved docs that are relevant |
| Retrieval Recall | Retrieval | Fraction of relevant docs that were retrieved |
| Answer Non-Empty | Generation | Basic sanity: did the system produce output? |
| Context Usage | Generation | Does the answer reflect the retrieved context? |
| Faithfulness | Generation | Is the answer grounded in the context (no hallucination)? |
| Answer Correctness | End-to-end | Does the answer match the expected ground truth? |

This lesson implements the first four as custom scorers. Faithfulness and correctness via LLM-as-judge are covered in L2-M3.3.

### Comparing Retrieval Strategies

A common question: how many documents should we retrieve? Retrieving more documents (higher top-k) increases recall but may decrease precision and introduce noise that confuses the generator. We compare top-1 vs. top-3 to see this tradeoff in practice.

## Step-by-Step

### Step 1: Build the Knowledge Base

We create 7 short documents about Python programming concepts. In production, these would come from a document store, but for this lesson we keep them in-memory.

```python
KNOWLEDGE_BASE = [
    {"id": "doc1", "title": "Python Lists", "text": "Python lists are ordered..."},
    {"id": "doc2", "title": "Python Dictionaries", "text": "Dictionaries store..."},
    # ... 7 documents total
]
```

### Step 2: Implement TF-IDF Retrieval

Instead of a vector database, we use TF-IDF cosine similarity for retrieval. This keeps the lesson focused on evaluation rather than infrastructure.

```python
class SimpleRAG:
    def retrieve(self, query: str) -> list[dict]:
        """Return top-k documents by TF-IDF cosine similarity."""
        query_vec = _tfidf_vector(_tokenize(query), self._idf)
        scored = [(doc, cosine_similarity(query_vec, doc_vec)) for ...]
        return top_k_docs

    def answer(self, question: str) -> dict:
        """Retrieve context, then generate an answer with the LLM."""
        retrieved = self.retrieve(question)
        context = "\n\n".join(d["text"] for d in retrieved)
        response = self._chain.invoke({"context": context, "question": question})
        return {"answer": response, "retrieved_ids": [...], "context": context}
```

### Step 3: Define the Evaluation Dataset

Five question-answer pairs with ground truth answers and expected document IDs. The expected doc IDs let us measure retrieval quality objectively.

```python
EVAL_DATASET = [
    {
        "question": "How do you create a list in Python?",
        "expected_answer": "Use square brackets, for example my_list = [1, 2, 3].",
        "expected_doc_ids": ["doc1"],
    },
    # ... 5 questions total
]
```

### Step 4: Create Custom Scorers

We use the `@scorer` decorator from `mlflow.genai.scorers` to define four evaluation metrics:

```python
@scorer(name="retrieval_precision")
def retrieval_precision(inputs, outputs, expectations) -> Feedback:
    """Fraction of retrieved docs that are in the expected set."""
    retrieved = outputs.get("retrieved_ids", [])
    expected = expectations.get("expected_doc_ids", [])
    hits = sum(1 for r in retrieved if r in expected)
    precision = hits / len(retrieved)
    return Feedback(value=precision, rationale=f"Precision={precision:.2f}")

@scorer(name="retrieval_recall")
def retrieval_recall(inputs, outputs, expectations) -> Feedback:
    """Fraction of expected docs that were actually retrieved."""
    ...

@scorer(name="context_used")
def context_used(inputs, outputs) -> Feedback:
    """Does the answer reflect content from the retrieved context?"""
    ...
```

### Step 5: Run Evaluation with `mlflow.genai.evaluate()`

For each retrieval strategy, we generate predictions and evaluate with all scorers:

```python
eval_df = pd.DataFrame([{
    "inputs": {"question": q},
    "outputs": rag_output,         # includes answer, retrieved_ids, context
    "expectations": {"expected_answer": ..., "expected_doc_ids": [...]},
} for ...])

eval_result = mlflow.genai.evaluate(
    data=eval_df,
    scorers=[retrieval_precision, retrieval_recall, answer_not_empty, context_used],
)
```

### Step 6: Compare Strategies

We log both configurations as nested runs under a parent "strategy_comparison" run, making it easy to compare in the MLflow UI.

```python
with mlflow.start_run(run_name="strategy_comparison"):
    with mlflow.start_run(run_name="top_1_retrieval", nested=True):
        mlflow.log_param("top_k", 1)
        mlflow.log_metrics(metrics_top1)
    with mlflow.start_run(run_name="top_3_retrieval", nested=True):
        mlflow.log_param("top_k", 3)
        mlflow.log_metrics(metrics_top3)
```

## Running the Lesson

```bash
cd tutorial/level_2/M3_deep_evaluation/2_rag_evaluation
uv sync
uv run python main.py
```

## Expected Output

The terminal will show evaluation metrics for both strategies:

```
============================================================
L2-3.2 — RAG System Evaluation
============================================================

Knowledge base: 7 documents
Evaluation set:  5 questions

============================================================
Evaluating strategy: top_1 (top_k=1)
============================================================

  Metrics for 'top_1':
    answer_not_empty/mean:              1.0000
    context_used/mean:                  0.4000
    retrieval_precision/mean:           1.0000
    retrieval_recall/mean:              1.0000

============================================================
Evaluating strategy: top_3 (top_k=3)
============================================================

  Metrics for 'top_3':
    answer_not_empty/mean:              1.0000
    context_used/mean:                  0.2000
    retrieval_precision/mean:           0.3333
    retrieval_recall/mean:              1.0000

============================================================
Comparison: Top-1 vs Top-3 Retrieval
============================================================

  Metric                                   Top-1      Top-3     Winner
  -----------------------------------------------------------------
  retrieval_precision/mean                1.0000     0.3333      top_1
  retrieval_recall/mean                   1.0000     1.0000        tie
  answer_not_empty/mean                   1.0000     1.0000        tie
  context_used/mean                       0.4000     0.2000      top_1
```

In the MLflow UI at http://127.0.0.1:5555, navigate to the experiment "L2/M3_deep_evaluation/2_rag_evaluation" to see:
- The parent run "strategy_comparison" with two nested child runs
- Side-by-side metric comparison
- Evaluation results from `mlflow.genai.evaluate()` in the Evaluation tab

## Key Takeaways

- RAG evaluation requires measuring retrieval and generation quality separately.
- **Retrieval precision** tells you how focused your retrieval is; **retrieval recall** tells you how complete it is. Higher top-k typically improves recall but hurts precision.
- Custom scorers via `@scorer` integrate directly with `mlflow.genai.evaluate()` and return `Feedback` objects with both a numeric value and a human-readable rationale.
- Nested MLflow runs make it easy to compare configurations in the UI.
- Even without a vector database, TF-IDF retrieval demonstrates all the evaluation patterns you need; swap in a real retriever and the scorers work the same way.

## Next Steps

In **L2-M3.3 (GenAI Evaluation Framework)**, you will use the full `mlflow.genai.evaluation` framework with built-in scorers and LLM-as-judge to evaluate faithfulness, relevance, and answer correctness -- metrics that require an LLM to assess quality rather than simple programmatic checks.
