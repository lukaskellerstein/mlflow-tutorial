"""
L3-1.1 — Agent Testing Framework

Build a production-quality agent testing framework that:
  1. Creates a LangGraph ReAct agent with calculator and text_analyzer tools
  2. Defines structured test cases with expected outputs and tool usage
  3. Runs an automated test suite with pass/fail tracking per test case
  4. Logs all results to MLflow with nested runs (parent = suite, child = test)
  5. Saves a regression baseline and shows how to compare against it
"""

import json

import mlflow
import mlflow.langchain
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from test_framework import (
    AgentTestRunner,
    TestCase,
    build_results_dataframe,
    compare_to_baseline,
    print_summary,
    save_baseline,
)


# ---------------------------------------------------------------------------
# 1. Tools for the agent
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 3 * 4'.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return f"Error: invalid characters in expression '{expression}'"
    try:
        result = eval(expression)  # safe: only digits and operators allowed
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return statistics: word count, character count,
    sentence count, and average word length.

    Args:
        text: The text to analyze.
    """
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)
    return (
        f"Word count: {word_count}, Character count: {char_count}, "
        f"Sentence count: {sentence_count}, Average word length: {avg_word_len}"
    )


TOOLS = [calculator, text_analyzer]


# ---------------------------------------------------------------------------
# 2. Agent factory
# ---------------------------------------------------------------------------
def build_agent():
    """Create a LangGraph ReAct agent with calculator and text_analyzer tools."""
    llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
    return create_agent(llm, tools=TOOLS)


# ---------------------------------------------------------------------------
# 3. Test suite
# ---------------------------------------------------------------------------
TEST_SUITE: list[TestCase] = [
    TestCase(name="simple_addition", input="What is 25 + 37?",
             expected_output="62", expected_tools=["calculator"], difficulty="easy"),
    TestCase(name="multiplication", input="Calculate 12 * 15.",
             expected_output="180", expected_tools=["calculator"], difficulty="easy"),
    TestCase(name="complex_expression", input="What is (100 - 37) * 2 + 14?",
             expected_output="140", expected_tools=["calculator"], difficulty="medium"),
    TestCase(name="text_word_count",
             input='How many words are in the following text: '
                   '"The quick brown fox jumps over the lazy dog"?',
             expected_output="9", expected_tools=["text_analyzer"], difficulty="easy"),
    TestCase(name="text_analysis_detail",
             input='Analyze this text for me: "Hello world. How are you doing today?"',
             expected_output="word count", expected_tools=["text_analyzer"],
             difficulty="medium", tags={"category": "text"}),
    TestCase(name="multi_tool",
             input='First, calculate 50 * 4. Then analyze the text '
                   '"MLflow is great for tracking experiments."',
             expected_output="200", expected_tools=["calculator", "text_analyzer"],
             difficulty="hard"),
    TestCase(name="no_tool_needed", input="Say hello.",
             expected_output="hello", expected_tools=[], difficulty="easy"),
    TestCase(name="division", input="What is 144 divided by 12?",
             expected_output="12", expected_tools=["calculator"], difficulty="easy"),
]


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L3-1.1 — Agent Testing Framework")
    print("=" * 60)

    # --- Part 1: Build the agent ---
    print("\n--- Part 1: Building LangGraph ReAct agent ---")
    agent = build_agent()
    print(f"  Agent created with tools: {[t.name for t in TOOLS]}")

    mlflow.langchain.autolog(log_traces=True)

    # --- Part 2: Show the test suite ---
    print(f"\n--- Part 2: Test suite ({len(TEST_SUITE)} cases) ---")
    for tc in TEST_SUITE:
        print(f"  [{tc.difficulty:6s}] {tc.name}: {tc.input[:60]}")

    # --- Part 3: Run tests with nested MLflow runs ---
    print(f"\n--- Part 3: Running automated test suite ---")
    with mlflow.start_run(run_name="agent_test_suite") as parent_run:
        mlflow.log_params({
            "agent_model": "gemma4:e2b",
            "num_tests": len(TEST_SUITE),
            "tools": json.dumps([t.name for t in TOOLS]),
        })
        mlflow.set_tags({"test_type": "agent_test_suite", "framework": "langgraph"})

        runner = AgentTestRunner(agent, TEST_SUITE)
        results = runner.run_suite()

        # Aggregate metrics on the parent run
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        mlflow.log_metrics({
            "pass_rate": passed / max(total, 1),
            "total_passed": passed,
            "total_failed": total - passed,
            "avg_duration_s": round(
                sum(r.duration_s for r in results) / max(total, 1), 2
            ),
        })

        print_summary(results, TEST_SUITE)

        # --- Part 4: Save baseline and demonstrate comparison ---
        print("--- Part 4: Regression baseline ---")
        df = build_results_dataframe(results)

        csv_path = "/tmp/agent_test_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="test_results")

        baseline_path = save_baseline(df, parent_run.info.run_id)
        print(f"  Baseline saved: {baseline_path}")
        print(f"  (logged as MLflow artifact under 'baselines/')\n")

        compare_to_baseline(df, baseline_path)

        print(f"  Parent run ID: {parent_run.info.run_id}")
        print(f"  View in MLflow UI: http://127.0.0.1:5000")

    print("=" * 60)
    print("Done! Explore nested test runs in the MLflow UI.")
    print("Re-run after changing the agent to see regression diffs.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M1_agent_evaluation/1_agent_testing")
    main()
