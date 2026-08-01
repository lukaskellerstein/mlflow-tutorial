"""
L2-3.2 — RAG System Evaluation

Build a simple RAG system with TF-IDF retrieval and evaluate retrieval
quality vs. generation quality using custom scorers and mlflow.genai.evaluate().
Compare top-1 vs. top-3 retrieval strategies.
"""

import math
import re
from collections import Counter

import mlflow
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from pydantic import SecretStr

# ── Part 1: Build a simple in-memory RAG system ───────────────────────

KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "title": "Python Lists",
        "text": (
            "Python lists are ordered, mutable sequences. They can hold items of "
            "different types. Lists support indexing, slicing, appending, and "
            "removing elements. Use square brackets to create a list: my_list = [1, 2, 3]."
        ),
    },
    {
        "id": "doc2",
        "title": "Python Dictionaries",
        "text": (
            "Dictionaries in Python store key-value pairs. Keys must be immutable "
            "and unique. Dictionaries are unordered in older Python versions but "
            "maintain insertion order since Python 3.7. Access values with dict[key]."
        ),
    },
    {
        "id": "doc3",
        "title": "Python Functions",
        "text": (
            "Functions are defined with the def keyword. They can accept positional "
            "and keyword arguments, have default values, and use *args and **kwargs "
            "for variable-length arguments. Functions are first-class objects in Python."
        ),
    },
    {
        "id": "doc4",
        "title": "Python Classes",
        "text": (
            "Python supports object-oriented programming through classes. Classes "
            "define attributes and methods. Use __init__ for the constructor. Python "
            "supports single and multiple inheritance, and uses the MRO for method "
            "resolution order."
        ),
    },
    {
        "id": "doc5",
        "title": "Python List Comprehensions",
        "text": (
            "List comprehensions provide a concise way to create lists. The syntax "
            "is [expression for item in iterable if condition]. They are faster than "
            "equivalent for-loops and are considered more Pythonic."
        ),
    },
    {
        "id": "doc6",
        "title": "Python Decorators",
        "text": (
            "Decorators are functions that modify the behavior of other functions. "
            "Use the @decorator syntax above a function definition. Common built-in "
            "decorators include @staticmethod, @classmethod, and @property. "
            "Decorators wrap functions without changing their source code."
        ),
    },
    {
        "id": "doc7",
        "title": "Python Error Handling",
        "text": (
            "Python uses try/except blocks for error handling. You can catch "
            "specific exceptions like ValueError or TypeError, use finally for "
            "cleanup, and raise custom exceptions. The else clause runs when no "
            "exception occurs."
        ),
    },
]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for each term."""
    n = len(corpus_tokens)
    df: Counter[str] = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a token list."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


class SimpleRAG:
    """TF-IDF based retrieval + LLM generation."""

    def __init__(self, docs: list[dict], llm: ChatOpenAI, top_k: int = 3) -> None:
        self.docs = docs
        self.llm = llm
        self.top_k = top_k
        # Pre-compute IDF from corpus
        self._corpus_tokens = [_tokenize(d["text"]) for d in docs]
        self._idf = _compute_idf(self._corpus_tokens)
        self._doc_vectors = [_tfidf_vector(t, self._idf) for t in self._corpus_tokens]

        self._chain = (
            ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Answer the question using ONLY the provided context. "
                    "If the context does not contain the answer, say 'I don't know'.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])
            | llm
            | StrOutputParser()
        )

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k documents by TF-IDF cosine similarity."""
        query_tokens = _tokenize(query)
        query_vec = _tfidf_vector(query_tokens, self._idf)
        scored = [
            (self.docs[i], _cosine_similarity(query_vec, dv))
            for i, dv in enumerate(self._doc_vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.top_k]]

    def answer(self, question: str) -> dict:
        """Retrieve context and generate an answer. Return full details."""
        retrieved = self.retrieve(question)
        context = "\n\n".join(d["text"] for d in retrieved)
        response = self._chain.invoke({"context": context, "question": question})
        return {
            "question": question,
            "answer": response,
            "retrieved_ids": [d["id"] for d in retrieved],
            "context": context,
        }


# ── Part 2: Evaluation dataset ────────────────────────────────────────

EVAL_DATASET = [
    {
        "question": "How do you create a list in Python?",
        "expected_answer": "Use square brackets, for example my_list = [1, 2, 3].",
        "expected_doc_ids": ["doc1"],
    },
    {
        "question": "What are Python decorators?",
        "expected_answer": (
            "Decorators are functions that modify the behavior of other functions "
            "using the @decorator syntax."
        ),
        "expected_doc_ids": ["doc6"],
    },
    {
        "question": "How does Python handle errors and exceptions?",
        "expected_answer": (
            "Python uses try/except blocks. You can catch specific exceptions, "
            "use finally for cleanup, and raise custom exceptions."
        ),
        "expected_doc_ids": ["doc7"],
    },
    {
        "question": "What is a list comprehension in Python?",
        "expected_answer": (
            "A concise syntax [expression for item in iterable if condition] "
            "to create lists, faster than equivalent for-loops."
        ),
        "expected_doc_ids": ["doc5"],
    },
    {
        "question": "How do Python dictionaries store data?",
        "expected_answer": (
            "Dictionaries store key-value pairs where keys must be immutable "
            "and unique. They maintain insertion order since Python 3.7."
        ),
        "expected_doc_ids": ["doc2"],
    },
]

# ── Part 3: Custom scorers ────────────────────────────────────────────


@scorer(name="retrieval_precision")
def retrieval_precision(inputs, outputs, expectations) -> Feedback:
    """Did we retrieve the right documents?"""
    retrieved = outputs.get("retrieved_ids", [])
    expected = expectations.get("expected_doc_ids", [])
    if not retrieved:
        return Feedback(value=0.0, rationale="No documents retrieved.")
    hits = sum(1 for r in retrieved if r in expected)
    precision = hits / len(retrieved)
    return Feedback(
        value=precision,
        rationale=f"Retrieved {retrieved}, expected {expected}. Precision={precision:.2f}",
    )


@scorer(name="retrieval_recall")
def retrieval_recall(inputs, outputs, expectations) -> Feedback:
    """Did we find all the expected documents?"""
    retrieved = outputs.get("retrieved_ids", [])
    expected = expectations.get("expected_doc_ids", [])
    if not expected:
        return Feedback(value=1.0, rationale="No expected docs specified.")
    hits = sum(1 for e in expected if e in retrieved)
    recall = hits / len(expected)
    return Feedback(
        value=recall,
        rationale=f"Retrieved {retrieved}, expected {expected}. Recall={recall:.2f}",
    )


@scorer(name="answer_not_empty")
def answer_not_empty(outputs) -> bool:
    """Basic check: did the system produce a non-empty answer?"""
    answer = outputs.get("answer", "") if isinstance(outputs, dict) else str(outputs)
    return len(answer.strip()) > 0


@scorer(name="context_used")
def context_used(inputs, outputs) -> Feedback:
    """Check whether the answer references content from the retrieved context."""
    answer = outputs.get("answer", "").lower()
    context = outputs.get("context", "").lower()
    if not context:
        return Feedback(value=0.0, rationale="No context available.")
    # Check how many context sentences appear (partially) in the answer
    context_sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 15]
    if not context_sentences:
        return Feedback(value=0.0, rationale="Context has no meaningful sentences.")
    overlap = 0
    for sent in context_sentences:
        # Check if key phrases from the sentence appear in the answer
        words = _tokenize(sent)
        key_words = [w for w in words if len(w) > 3]
        if key_words:
            matches = sum(1 for w in key_words if w in answer)
            if matches / len(key_words) > 0.3:
                overlap += 1
    score = overlap / len(context_sentences)
    return Feedback(
        value=round(score, 2),
        rationale=f"{overlap}/{len(context_sentences)} context sentences reflected in answer.",
    )


# ── Part 4: Run evaluation and compare retrieval strategies ───────────


def run_evaluation(rag: SimpleRAG, strategy_name: str) -> dict:
    """Run the RAG system on the eval dataset and evaluate with mlflow.genai.evaluate()."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating strategy: {strategy_name} (top_k={rag.top_k})")
    print("=" * 60)

    # Generate predictions
    results = []
    for item in EVAL_DATASET:
        rag_output = rag.answer(item["question"])
        results.append({
            "inputs": {"question": item["question"]},
            "outputs": rag_output,
            "expectations": {
                "expected_answer": item["expected_answer"],
                "expected_doc_ids": item["expected_doc_ids"],
            },
        })

    eval_df = pd.DataFrame(results)

    # Run evaluation with custom scorers
    eval_result = mlflow.genai.evaluate(
        data=eval_df,
        scorers=[
            retrieval_precision,
            retrieval_recall,
            answer_not_empty,
            context_used,
        ],
    )

    # Print results
    print(f"\n  Metrics for '{strategy_name}':")
    for name, value in sorted(eval_result.metrics.items()):
        print(f"    {name}: {value:.4f}")

    return eval_result.metrics


def main() -> None:
    """Evaluate a RAG system with two retrieval strategies."""
    mlflow.langchain.autolog()

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.0,
    )

    print("=" * 60)
    print("L2-3.2 — RAG System Evaluation")
    print("=" * 60)
    print()
    print(f"Knowledge base: {len(KNOWLEDGE_BASE)} documents")
    print(f"Evaluation set:  {len(EVAL_DATASET)} questions")

    # --- Strategy 1: Top-1 retrieval ---
    rag_top1 = SimpleRAG(KNOWLEDGE_BASE, llm, top_k=1)
    metrics_top1 = run_evaluation(rag_top1, "top_1")

    # --- Strategy 2: Top-3 retrieval ---
    rag_top3 = SimpleRAG(KNOWLEDGE_BASE, llm, top_k=3)
    metrics_top3 = run_evaluation(rag_top3, "top_3")

    # --- Log comparison to MLflow ---
    print(f"\n{'=' * 60}")
    print("Comparison: Top-1 vs Top-3 Retrieval")
    print("=" * 60)

    with mlflow.start_run(run_name="strategy_comparison"):
        with mlflow.start_run(run_name="top_1_retrieval", nested=True):
            mlflow.log_param("retrieval_strategy", "top_1")
            mlflow.log_param("top_k", 1)
            mlflow.log_param("num_docs", len(KNOWLEDGE_BASE))
            mlflow.log_param("num_eval_questions", len(EVAL_DATASET))
            for name, value in metrics_top1.items():
                mlflow.log_metric(name, value)

        with mlflow.start_run(run_name="top_3_retrieval", nested=True):
            mlflow.log_param("retrieval_strategy", "top_3")
            mlflow.log_param("top_k", 3)
            mlflow.log_param("num_docs", len(KNOWLEDGE_BASE))
            mlflow.log_param("num_eval_questions", len(EVAL_DATASET))
            for name, value in metrics_top3.items():
                mlflow.log_metric(name, value)

    # Print side-by-side comparison
    all_metric_names = sorted(set(metrics_top1) | set(metrics_top3))
    print(f"\n  {'Metric':<35} {'Top-1':>10} {'Top-3':>10} {'Winner':>10}")
    print(f"  {'-' * 65}")
    for m in all_metric_names:
        v1 = metrics_top1.get(m, 0.0)
        v3 = metrics_top3.get(m, 0.0)
        winner = "top_1" if v1 > v3 else ("top_3" if v3 > v1 else "tie")
        print(f"  {m:<35} {v1:>10.4f} {v3:>10.4f} {winner:>10}")

    print(f"\n{'=' * 60}")
    print("Done! Open MLflow UI to explore results:")
    print("  http://127.0.0.1:5555")
    print("  Experiment: L2/M3_deep_evaluation/2_rag_evaluation")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L1/M4_evaluation/3_rag_evaluation")
    main()
