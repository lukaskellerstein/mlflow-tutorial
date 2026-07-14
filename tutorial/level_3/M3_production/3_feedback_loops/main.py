"""
L3-3.3 -- Production Feedback Loops

Build a production-quality feedback collection and analysis system that:
  1. Collects user feedback (thumbs up/down, ratings, text comments) per response
  2. Associates feedback with MLflow trace IDs via mlflow.log_feedback()
  3. Generates LLM responses and simulates diverse user feedback
  4. Analyzes feedback to compute satisfaction metrics and find weak spots
  5. Uses negative feedback to drive prompt improvements across iterations
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mlflow.entities import AssessmentSource, AssessmentSourceType


# ---------------------------------------------------------------------------
# 1. Data structures
# ---------------------------------------------------------------------------
@dataclass
class UserFeedback:
    """A single piece of user feedback for an LLM response."""

    trace_id: str
    question: str
    response: str
    thumbs_up: bool
    rating: int  # 1-5
    comment: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 2. Feedback collector
# ---------------------------------------------------------------------------
class FeedbackCollector:
    """Simulates collecting user feedback on LLM responses.

    In production this would come from a UI callback, API endpoint, or
    message queue.  Here we use deterministic simulation so results are
    reproducible.
    """

    # Simulated feedback templates keyed by sentiment
    POSITIVE_COMMENTS = [
        "Very helpful, thanks!",
        "Exactly what I needed.",
        "Clear and concise answer.",
        "Great explanation!",
    ]
    NEGATIVE_COMMENTS = [
        "Too vague, needs more detail.",
        "This is incorrect.",
        "Didn't answer my question.",
        "Response was confusing.",
    ]
    NEUTRAL_COMMENTS = [
        "Okay, but could be better.",
        "Partially helpful.",
        "Decent but missing context.",
    ]

    def simulate_feedback(
        self, trace_id: str, question: str, response: str, *, seed: int = 0
    ) -> UserFeedback:
        """Generate simulated feedback using a seeded RNG for reproducibility."""
        rng = random.Random(seed)

        # Decide sentiment bucket
        bucket = rng.choices(
            ["positive", "neutral", "negative"], weights=[0.5, 0.3, 0.2]
        )[0]

        if bucket == "positive":
            rating = rng.randint(4, 5)
            comment = rng.choice(self.POSITIVE_COMMENTS)
            thumbs_up = True
        elif bucket == "neutral":
            rating = 3
            comment = rng.choice(self.NEUTRAL_COMMENTS)
            thumbs_up = rng.choice([True, False])
        else:
            rating = rng.randint(1, 2)
            comment = rng.choice(self.NEGATIVE_COMMENTS)
            thumbs_up = False

        return UserFeedback(
            trace_id=trace_id,
            question=question,
            response=response[:300],
            thumbs_up=thumbs_up,
            rating=rating,
            comment=comment,
        )


# ---------------------------------------------------------------------------
# 3. Log feedback to MLflow
# ---------------------------------------------------------------------------
def log_feedback_to_mlflow(fb: UserFeedback) -> None:
    """Persist a UserFeedback record as MLflow feedback assessments."""
    source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="simulated_user"
    )

    # Thumbs up / down
    mlflow.log_feedback(
        trace_id=fb.trace_id,
        name="thumbs_up",
        value=fb.thumbs_up,
        source=source,
        rationale=fb.comment,
        metadata={"question": fb.question[:200]},
    )

    # Numeric rating
    mlflow.log_feedback(
        trace_id=fb.trace_id,
        name="user_rating",
        value=fb.rating,
        source=source,
        rationale=fb.comment,
        metadata={"question": fb.question[:200]},
    )

    # Free-text comment
    mlflow.log_feedback(
        trace_id=fb.trace_id,
        name="user_comment",
        value=fb.comment,
        source=source,
        metadata={"question": fb.question[:200]},
    )


# ---------------------------------------------------------------------------
# 4. LLM helper
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What is machine learning?",
    "Explain gradient descent in simple terms.",
    "How does a neural network learn?",
    "What is the difference between supervised and unsupervised learning?",
    "What are transformers in NLP?",
    "How do you prevent overfitting?",
    "What is transfer learning?",
    "Explain the bias-variance tradeoff.",
]


@mlflow.trace(name="qa_response")
def generate_response(llm: ChatOpenAI, question: str, system_prompt: str) -> str:
    """Generate a response to a question using the LLM."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    result = llm.invoke(messages)
    return result.content


# ---------------------------------------------------------------------------
# 5. Feedback analysis
# ---------------------------------------------------------------------------
def analyze_feedback(feedbacks: list[UserFeedback]) -> dict[str, Any]:
    """Compute aggregate feedback metrics and identify weak spots."""
    if not feedbacks:
        return {}

    ratings = [fb.rating for fb in feedbacks]
    thumbs = [fb.thumbs_up for fb in feedbacks]

    avg_rating = sum(ratings) / len(ratings)
    satisfaction_rate = sum(thumbs) / len(thumbs)
    low_rated = [fb for fb in feedbacks if fb.rating <= 2]
    high_rated = [fb for fb in feedbacks if fb.rating >= 4]

    # Count comment categories
    issue_keywords = {
        "vague": ["vague", "detail"],
        "incorrect": ["incorrect", "wrong"],
        "off_topic": ["didn't answer", "question"],
        "confusing": ["confusing", "unclear"],
    }
    issue_counts: dict[str, int] = {k: 0 for k in issue_keywords}
    for fb in low_rated:
        comment_lower = fb.comment.lower()
        for issue, keywords in issue_keywords.items():
            if any(kw in comment_lower for kw in keywords):
                issue_counts[issue] += 1

    return {
        "total_responses": len(feedbacks),
        "avg_rating": round(avg_rating, 2),
        "satisfaction_rate": round(satisfaction_rate, 2),
        "thumbs_up_count": sum(thumbs),
        "thumbs_down_count": len(thumbs) - sum(thumbs),
        "low_rated_count": len(low_rated),
        "high_rated_count": len(high_rated),
        "issue_counts": issue_counts,
        "low_rated_questions": [fb.question for fb in low_rated],
    }


def print_analysis(analysis: dict[str, Any], label: str = "Feedback Analysis") -> None:
    """Pretty-print a feedback analysis report."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total responses:    {analysis['total_responses']}")
    print(f"  Average rating:     {analysis['avg_rating']}/5")
    print(f"  Satisfaction rate:  {analysis['satisfaction_rate']:.0%}")
    print(f"  Thumbs up/down:    {analysis['thumbs_up_count']}/{analysis['thumbs_down_count']}")
    print(f"  High-rated (4-5):   {analysis['high_rated_count']}")
    print(f"  Low-rated (1-2):    {analysis['low_rated_count']}")

    issues = analysis.get("issue_counts", {})
    if any(v > 0 for v in issues.values()):
        print(f"\n  Common issues:")
        for issue, count in sorted(issues.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    - {issue}: {count}")

    low_qs = analysis.get("low_rated_questions", [])
    if low_qs:
        print(f"\n  Questions needing improvement:")
        for q in low_qs:
            print(f"    - {q}")


# ---------------------------------------------------------------------------
# 6. Feedback-driven improvement iteration
# ---------------------------------------------------------------------------
PROMPT_V1 = "You are a helpful assistant. Answer the question concisely."

PROMPT_V2 = (
    "You are a helpful and thorough assistant. Answer the question clearly "
    "and completely. Provide specific examples when possible. If the topic "
    "is technical, explain it step by step so a beginner can follow."
)


def run_iteration(
    llm: ChatOpenAI,
    collector: FeedbackCollector,
    system_prompt: str,
    iteration: int,
    run_name: str,
) -> tuple[list[UserFeedback], dict[str, Any]]:
    """Run one iteration: generate responses, collect feedback, analyze."""
    feedbacks: list[UserFeedback] = []

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({
            "iteration": iteration,
            "system_prompt": system_prompt[:250],
            "model": "google/gemma-4-26b-a4b",
            "num_questions": len(QUESTIONS),
        })

        for idx, question in enumerate(QUESTIONS):
            print(f"    [{idx + 1}/{len(QUESTIONS)}] {question[:50]}...", flush=True)

            response = generate_response(llm, question, system_prompt)
            trace_id = mlflow.get_last_active_trace_id()

            # Flush async trace export so the trace is persisted on the
            # server before we attach feedback assessments to it.
            mlflow.flush_trace_async_logging()

            if trace_id:
                # Seed combines iteration and index for reproducible but
                # varied feedback across iterations
                fb = collector.simulate_feedback(
                    trace_id, question, response, seed=iteration * 100 + idx
                )
                log_feedback_to_mlflow(fb)
                feedbacks.append(fb)
            else:
                print("      (no trace ID captured -- skipping feedback)")

        # Analyze this iteration
        analysis = analyze_feedback(feedbacks)

        # Log aggregate metrics to the run
        mlflow.log_metrics({
            "avg_rating": analysis["avg_rating"],
            "satisfaction_rate": analysis["satisfaction_rate"],
            "low_rated_count": analysis["low_rated_count"],
            "high_rated_count": analysis["high_rated_count"],
        })
        mlflow.set_tags({
            "iteration": str(iteration),
            "prompt_version": f"v{iteration}",
        })

    return feedbacks, analysis


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L3-3.3 -- Production Feedback Loops")
    print("=" * 60)

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )
    collector = FeedbackCollector()

    # Enable LangChain auto-tracing so each LLM call gets a trace
    mlflow.langchain.autolog(log_traces=True)

    with mlflow.start_run(run_name="feedback_loop_experiment"):

        # --- Iteration 1: baseline prompt ------------------------------------
        print("\n--- Iteration 1: Baseline prompt (v1) ---")
        fb1, analysis1 = run_iteration(
            llm, collector, PROMPT_V1, iteration=1, run_name="iteration_1_baseline"
        )
        print_analysis(analysis1, "Iteration 1 -- Baseline")

        # --- Iteration 2: improved prompt based on feedback ------------------
        print("\n--- Iteration 2: Improved prompt (v2) ---")
        print("  (Prompt improved based on feedback: more detail, examples, steps)")
        fb2, analysis2 = run_iteration(
            llm, collector, PROMPT_V2, iteration=2, run_name="iteration_2_improved"
        )
        print_analysis(analysis2, "Iteration 2 -- Improved")

        # --- Cross-iteration comparison --------------------------------------
        print("\n" + "=" * 60)
        print("  Cross-Iteration Comparison")
        print("=" * 60)
        delta_rating = analysis2["avg_rating"] - analysis1["avg_rating"]
        delta_sat = analysis2["satisfaction_rate"] - analysis1["satisfaction_rate"]
        print(f"  Avg rating:        {analysis1['avg_rating']} -> {analysis2['avg_rating']}  ({delta_rating:+.2f})")
        print(f"  Satisfaction rate:  {analysis1['satisfaction_rate']:.0%} -> {analysis2['satisfaction_rate']:.0%}  ({delta_sat:+.0%})")
        print(f"  Low-rated:         {analysis1['low_rated_count']} -> {analysis2['low_rated_count']}")

        mlflow.log_metrics({
            "delta_avg_rating": delta_rating,
            "delta_satisfaction_rate": delta_sat,
            "final_avg_rating": analysis2["avg_rating"],
            "final_satisfaction_rate": analysis2["satisfaction_rate"],
        })

        # --- Build full feedback DataFrame -----------------------------------
        all_fb = fb1 + fb2
        df = pd.DataFrame([
            {
                "trace_id": fb.trace_id,
                "question": fb.question,
                "thumbs_up": fb.thumbs_up,
                "rating": fb.rating,
                "comment": fb.comment,
                "iteration": 1 if fb in fb1 else 2,
            }
            for fb in all_fb
        ])
        csv_path = "/tmp/feedback_report.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="feedback")

        # Log summary table
        mlflow.log_table(
            data=df[["question", "rating", "thumbs_up", "comment", "iteration"]],
            artifact_file="feedback/summary_table.json",
        )

        print(f"\n  Feedback report saved ({len(all_fb)} records)")
        print(f"  View in MLflow UI: http://127.0.0.1:5000")

    print("\n" + "=" * 60)
    print("Done! Check MLflow for traces with attached feedback assessments.")
    print("Compare iteration_1_baseline vs iteration_2_improved runs.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M3_production/3_feedback_loops")
    main()
