"""
L3-M4.2 — Framework Benchmark Capstone

Systematic cross-framework agent benchmark using MLflow.  Compares three
agent implementation approaches on the same task set with standardized
metrics, then generates a benchmark report with recommendations.

Approaches:
  A. Simple LLM Chain — prompt -> LLM -> answer (no tools)
  B. ReAct Agent      — langgraph.prebuilt.create_react_agent with tools
  C. Custom StateGraph — classify -> route -> process -> respond

All approaches use ChatOpenAI(model="google/gemma-4-26b-a4b") and the same tool set.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

import mlflow
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from typing_extensions import TypedDict

# ── MLflow setup ──────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L3/M4_capstones/2_framework_benchmark")

# ── Shared LLM and tools ──────────────────────────────────
LLM = ChatOpenAI(
    model="google/gemma-4-26b-a4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("lm-studio"),
    temperature=0.0,
)

KNOWLEDGE: dict[str, str] = {
    "python": "Python is a high-level programming language created by Guido van Rossum, "
    "known for readability and a vast ecosystem of libraries.",
    "mlflow": "MLflow is an open-source platform for the ML lifecycle providing "
    "tracking, model registry, evaluation, and deployment capabilities.",
    "docker": "Docker is a containerization platform that packages applications "
    "and dependencies into lightweight, portable containers.",
    "kubernetes": "Kubernetes is an open-source container orchestration system that "
    "automates deployment, scaling, and management of containers.",
    "langchain": "LangChain is a framework for building LLM-powered applications "
    "with abstractions for chains, agents, memory, and tools.",
    "langgraph": "LangGraph builds stateful multi-actor LLM applications using "
    "graph-based workflows with nodes, edges, and state management.",
}


@tool
def lookup(topic: str) -> str:
    """Look up factual information about a technology topic.

    Args:
        topic: The technology topic to look up (e.g. 'python', 'mlflow').
    """
    key = topic.strip().lower()
    for k, v in KNOWLEDGE.items():
        if k in key or key in k:
            return v
    return f"No information found for '{topic}'. Known topics: {', '.join(KNOWLEDGE)}."


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3' or '10 * 5'.

    Args:
        expression: A math expression containing only numbers and basic operators.
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))  # nosec: reached only for whitelisted arithmetic chars
        return "Invalid expression — only basic arithmetic is supported."
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = [lookup, calculate]


# ── Agent implementations ─────────────────────────────────


def build_simple_chain() -> Callable[[str], dict]:
    """Approach A: prompt -> LLM -> answer (no tools)."""

    def run(question: str) -> dict:
        start = time.time()
        response = LLM.invoke(
            [
                {"role": "system", "content": "Answer the question concisely in 1-2 sentences."},
                {"role": "user", "content": question},
            ]
        )
        elapsed = time.time() - start
        answer = str(response.content)
        return {
            "answer": answer,
            "latency_s": elapsed,
            "tool_calls": 0,
            "tokens_est": len(answer.split()),
        }

    return run


def build_react_agent() -> Callable[[str], dict]:
    """Approach B: ReAct agent with tool access."""
    agent = create_react_agent(
        model=LLM,
        tools=TOOLS,
        prompt="You are a helpful assistant. Use the provided tools when the "
        "question is about a technology topic or requires calculation. "
        "Answer concisely.",
    )

    def run(question: str) -> dict:
        start = time.time()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        elapsed = time.time() - start
        messages = result["messages"]
        answer = messages[-1].content if messages else ""
        tc = sum(1 for m in messages if m.type == "tool")
        tokens = sum(len(m.content.split()) for m in messages if hasattr(m, "content") and m.content)
        return {
            "answer": answer,
            "latency_s": elapsed,
            "tool_calls": tc,
            "tokens_est": tokens,
        }

    return run


def build_stategraph_agent() -> Callable[[str], dict]:
    """Approach C: Custom StateGraph with classify -> route -> process -> respond."""

    class GraphState(TypedDict):
        messages: Annotated[list, add_messages]
        category: str
        context: str
        answer: str

    def classify_node(state: GraphState) -> dict:
        user_msg = state["messages"][-1]
        q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
        resp = LLM.invoke(
            [
                {
                    "role": "system",
                    "content": "Classify the following question into exactly one category. "
                    "Reply with ONLY the category name, nothing else.\n"
                    "Categories: tech_lookup, math, general",
                },
                {"role": "user", "content": q},
            ]
        )
        cat = str(resp.content).strip().lower().replace("'", "").replace('"', "")
        if "tech" in cat or "lookup" in cat:
            cat = "tech_lookup"
        elif "math" in cat or "calc" in cat:
            cat = "math"
        else:
            cat = "general"
        return {"category": cat}

    def route_node(state: GraphState) -> str:
        category = state.get("category", "general")
        if category == "tech_lookup":
            return "process_tech"
        elif category == "math":
            return "process_math"
        return "process_general"

    def process_tech(state: GraphState) -> dict:
        user_msg = state["messages"][0]
        q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
        ctx = lookup.invoke(q)
        return {"context": ctx}

    def process_math(state: GraphState) -> dict:
        user_msg = state["messages"][0]
        q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
        resp = LLM.invoke(
            [
                {
                    "role": "system",
                    "content": "Extract ONLY the math expression from this question. "
                    "Reply with just the expression, nothing else.",
                },
                {"role": "user", "content": q},
            ]
        )
        expr = str(resp.content).strip()
        result = calculate.invoke(expr)
        return {"context": f"Calculation result: {result}"}

    def process_general(state: GraphState) -> dict:
        return {"context": ""}

    def respond_node(state: GraphState) -> dict:
        user_msg = state["messages"][0]
        q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
        ctx = state.get("context", "")
        prompt_parts = [{"role": "system", "content": "Answer concisely in 1-2 sentences."}]
        if ctx:
            prompt_parts.append({"role": "system", "content": f"Context: {ctx}"})
        prompt_parts.append({"role": "user", "content": q})
        resp = LLM.invoke(prompt_parts)
        return {
            "answer": resp.content,
            "messages": [{"role": "assistant", "content": resp.content}],
        }

    g = StateGraph(GraphState)
    g.add_node("classify", classify_node)
    g.add_node("process_tech", process_tech)
    g.add_node("process_math", process_math)
    g.add_node("process_general", process_general)
    g.add_node("respond", respond_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_node)
    g.add_edge("process_tech", "respond")
    g.add_edge("process_math", "respond")
    g.add_edge("process_general", "respond")
    g.add_edge("respond", END)
    compiled = g.compile()

    def run(question: str) -> dict:
        start = time.time()
        result = compiled.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "category": "",
                "context": "",
                "answer": "",
            }
        )
        elapsed = time.time() - start
        answer = result.get("answer", "")
        tool_calls = 1 if result.get("context", "") else 0
        return {
            "answer": answer,
            "latency_s": elapsed,
            "tool_calls": tool_calls,
            "tokens_est": len(answer.split()),
        }

    return run


# ── Benchmark Suite ───────────────────────────────────────


@dataclass
class TestCase:
    question: str
    expected_keyword: str
    category: str  # "simple", "tool_required", "multi_step"
    needs_tool: bool


@dataclass
class AgentEntry:
    name: str
    run_fn: Callable[[str], dict]
    description: str


@dataclass
class BenchmarkSuite:
    """Reusable benchmarking infrastructure for comparing agent approaches."""

    agents: list[AgentEntry] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    summary_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def add_agent(self, name: str, run_fn: Callable[[str], dict], description: str = "") -> None:
        """Register an agent for benchmarking."""
        self.agents.append(AgentEntry(name=name, run_fn=run_fn, description=description))

    def add_test_cases(self, cases: list[TestCase]) -> None:
        """Add test cases to the benchmark."""
        self.test_cases.extend(cases)

    def _score_correctness(self, answer: str, expected: str) -> float:
        return 1.0 if expected.lower() in answer.lower() else 0.0

    def _score_tool_usage(self, tool_calls: int, needs_tool: bool) -> float:
        if needs_tool:
            return 1.0 if tool_calls > 0 else 0.0
        return 1.0 if tool_calls == 0 else 0.5

    def run_benchmark(self) -> list[dict]:
        """Execute all agents on all test cases with MLflow logging."""
        self.results = []

        with mlflow.start_run(run_name="benchmark") as parent:
            mlflow.set_tags(
                {
                    "benchmark_type": "framework_comparison",
                    "num_agents": str(len(self.agents)),
                    "num_test_cases": str(len(self.test_cases)),
                    "model": "google/gemma-4-26b-a4b",
                }
            )

            for agent_entry in self.agents:
                print(f"\n{'─' * 70}")
                print(f"  Benchmarking: {agent_entry.name}")
                print(f"  {agent_entry.description}")
                print(f"{'─' * 70}")

                with mlflow.start_run(run_name=agent_entry.name, nested=True):
                    mlflow.log_params(
                        {
                            "agent": agent_entry.name,
                            "model": "google/gemma-4-26b-a4b",
                            "num_cases": len(self.test_cases),
                        }
                    )

                    agent_rows: list[dict] = []
                    for i, tc in enumerate(self.test_cases, 1):
                        with mlflow.start_run(run_name=f"case_{i}_{tc.category}", nested=True):
                            try:
                                result = agent_entry.run_fn(tc.question)
                            except Exception as e:
                                result = {
                                    "answer": f"ERROR: {e}",
                                    "latency_s": 0.0,
                                    "tool_calls": 0,
                                    "tokens_est": 0,
                                }

                            correctness = self._score_correctness(result["answer"], tc.expected_keyword)
                            tool_score = self._score_tool_usage(result["tool_calls"], tc.needs_tool)

                            row = {
                                "agent": agent_entry.name,
                                "case": i,
                                "question": tc.question,
                                "category": tc.category,
                                "answer": result["answer"][:150],
                                "correctness": correctness,
                                "tool_usage": tool_score,
                                "latency_s": round(result["latency_s"], 3),
                                "tool_calls": result["tool_calls"],
                                "tokens_est": result["tokens_est"],
                            }
                            agent_rows.append(row)
                            self.results.append(row)

                            mlflow.log_params(
                                {
                                    "question": tc.question[:250],
                                    "category": tc.category,
                                    "needs_tool": str(tc.needs_tool),
                                }
                            )
                            mlflow.log_metrics(
                                {
                                    "correctness": correctness,
                                    "tool_usage": tool_score,
                                    "latency_s": result["latency_s"],
                                    "tokens_est": result["tokens_est"],
                                }
                            )

                            status = "PASS" if correctness == 1.0 else "FAIL"
                            print(f"  [{status}] Q{i} ({tc.category}): {tc.question[:45]}")
                            print(
                                f"         Correctness={correctness:.0f}  "
                                f"ToolUse={tool_score:.1f}  "
                                f"Latency={result['latency_s']:.2f}s"
                            )

                    # Log agent-level aggregate metrics
                    n = len(agent_rows)
                    agg = {
                        "avg_correctness": round(sum(r["correctness"] for r in agent_rows) / n, 3),
                        "avg_tool_usage": round(sum(r["tool_usage"] for r in agent_rows) / n, 3),
                        "avg_latency_s": round(sum(r["latency_s"] for r in agent_rows) / n, 3),
                        "total_tokens_est": sum(r["tokens_est"] for r in agent_rows),
                    }
                    mlflow.log_metrics(agg)

            # Build summary and log to parent
            self._build_summary()
            self._log_artifacts(parent)

        return self.results

    def _build_summary(self) -> None:
        """Compute per-agent aggregate metrics."""
        df = pd.DataFrame(self.results)
        self.summary_df = cast(
            pd.DataFrame,
            df.groupby("agent")
            .agg(
                correctness=("correctness", "mean"),
                tool_usage=("tool_usage", "mean"),
                latency_s=("latency_s", "mean"),
                tokens_est=("tokens_est", "sum"),
                tool_calls=("tool_calls", "sum"),
            )
            .round(3),
        )
        self.summary_df["quality"] = ((self.summary_df["correctness"] + self.summary_df["tool_usage"]) / 2).round(3)
        self.summary_df["token_efficiency"] = (
            self.summary_df["quality"] / self.summary_df["tokens_est"].clip(lower=1) * 100
        ).round(3)

    def _log_artifacts(self, parent_run: Any) -> None:
        """Log benchmark artifacts to the parent MLflow run."""
        # Comparison CSV
        csv_path = "/tmp/benchmark_comparison.csv"
        self.summary_df.to_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # Full results CSV
        results_path = "/tmp/benchmark_results.csv"
        pd.DataFrame(self.results).to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Benchmark report
        report = self.generate_report()
        report_path = "/tmp/benchmark_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log parent-level tags
        if not self.summary_df.empty:
            best_quality = self.summary_df["quality"].idxmax()
            fastest = self.summary_df["latency_s"].idxmin()
            most_efficient = self.summary_df["token_efficiency"].idxmax()
            mlflow.log_params(
                {
                    "best_quality_agent": best_quality,
                    "fastest_agent": fastest,
                    "most_efficient_agent": most_efficient,
                }
            )

    def generate_report(self) -> str:
        """Generate a full benchmark report string."""
        lines: list[str] = []

        # --- Comparison table ---
        lines.append("=" * 70)
        lines.append("  FRAMEWORK BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append("  Comparison Table: Agent x Metric")
        lines.append("  " + "-" * 66)
        header = (
            f"  {'Agent':<22} {'Correct':>8} {'ToolUse':>8} {'Latency':>8} {'Tokens':>7} {'Quality':>8} {'Effic.':>8}"
        )
        lines.append(header)
        lines.append("  " + "-" * 66)
        for agent_name, row in self.summary_df.iterrows():
            lines.append(
                f"  {agent_name:<22} {row['correctness']:>8.3f} "
                f"{row['tool_usage']:>8.3f} {row['latency_s']:>7.2f}s "
                f"{int(row['tokens_est']):>7} {row['quality']:>8.3f} "
                f"{row['token_efficiency']:>8.3f}"
            )

        # --- Cost-quality tradeoff ---
        lines.append("")
        lines.append("  " + "-" * 66)
        lines.append("  Cost-Quality Tradeoff Analysis")
        lines.append("  " + "-" * 66)

        if not self.summary_df.empty:
            best_q = self.summary_df["quality"].idxmax()
            fastest = self.summary_df["latency_s"].idxmin()
            most_eff = self.summary_df["token_efficiency"].idxmax()

            lines.append(f"  Best quality:     {best_q} (score={self.summary_df.loc[best_q, 'quality']:.3f})")
            lines.append(f"  Fastest:          {fastest} (latency={self.summary_df.loc[fastest, 'latency_s']:.3f}s)")
            lines.append(
                f"  Most efficient:   {most_eff} (efficiency={self.summary_df.loc[most_eff, 'token_efficiency']:.3f})"
            )

        # --- Pareto frontier ---
        lines.append("")
        lines.append("  Pareto Frontier (quality vs latency):")
        pareto = self._compute_pareto()
        for arch in pareto:
            lines.append(
                f"    * {arch}: quality={self.summary_df.loc[arch, 'quality']:.3f}, "
                f"latency={self.summary_df.loc[arch, 'latency_s']:.3f}s"
            )

        # --- Recommendations ---
        lines.append("")
        lines.append("  " + "-" * 66)
        lines.append("  Recommendations")
        lines.append("  " + "-" * 66)
        lines.extend(self._generate_recommendations())

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def _compute_pareto(self) -> list[str]:
        """Find Pareto-optimal agents (not dominated on quality AND latency)."""
        pareto: list[str] = []
        for arch in self.summary_df.index:
            dominated = False
            for other in self.summary_df.index:
                if other == arch:
                    continue
                other_better_quality = self.summary_df.loc[other, "quality"] >= self.summary_df.loc[arch, "quality"]
                other_lower_latency = self.summary_df.loc[other, "latency_s"] <= self.summary_df.loc[arch, "latency_s"]
                strictly_better = (
                    self.summary_df.loc[other, "quality"] > self.summary_df.loc[arch, "quality"]
                    or self.summary_df.loc[other, "latency_s"] < self.summary_df.loc[arch, "latency_s"]
                )
                if other_better_quality and other_lower_latency and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(arch)
        return pareto

    def _generate_recommendations(self) -> list[str]:
        """Produce per-use-case recommendations based on benchmark data."""
        recs: list[str] = []
        if self.summary_df.empty:
            return ["  No data to generate recommendations."]

        fastest = self.summary_df["latency_s"].idxmin()
        best_q = self.summary_df["quality"].idxmax()
        most_eff = self.summary_df["token_efficiency"].idxmax()

        recs.append(
            f"  - Latency-sensitive apps: use '{fastest}' ({self.summary_df.loc[fastest, 'latency_s']:.2f}s avg)"
        )
        recs.append(
            f"  - Quality-critical apps:  use '{best_q}' ({self.summary_df.loc[best_q, 'quality']:.3f} quality)"
        )
        recs.append(
            f"  - Cost-constrained apps:  use '{most_eff}' "
            f"({self.summary_df.loc[most_eff, 'token_efficiency']:.3f} efficiency)"
        )

        # Category-specific analysis
        df = pd.DataFrame(self.results)
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            cat_agg = cat_df.groupby("agent")["correctness"].mean()
            best_for_cat = cat_agg.idxmax()
            recs.append(f"  - '{cat}' tasks:  best agent is '{best_for_cat}' ({cat_agg[best_for_cat]:.1%} correct)")

        return recs


# ── Evaluation dataset ────────────────────────────────────


def create_test_cases() -> list[TestCase]:
    """6 test cases across three categories."""
    return [
        # Simple questions (no tools needed)
        TestCase(
            question="What is 2 + 2?",
            expected_keyword="4",
            category="simple",
            needs_tool=False,
        ),
        TestCase(
            question="Say hello in French.",
            expected_keyword="bonjour",
            category="simple",
            needs_tool=False,
        ),
        # Tool-required questions
        TestCase(
            question="What is Python and what is it known for?",
            expected_keyword="readability",
            category="tool_required",
            needs_tool=True,
        ),
        TestCase(
            question="Describe what MLflow does.",
            expected_keyword="tracking",
            category="tool_required",
            needs_tool=True,
        ),
        # Multi-step questions (reasoning + tools)
        TestCase(
            question="What is 125 * 8?",
            expected_keyword="1000",
            category="multi_step",
            needs_tool=True,
        ),
        TestCase(
            question="What is Docker and how does it relate to Kubernetes?",
            expected_keyword="container",
            category="multi_step",
            needs_tool=True,
        ),
    ]


# ── Main ──────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  L3-M4.2 — Framework Benchmark Capstone")
    print("=" * 70)

    # Build the benchmark suite
    suite = BenchmarkSuite()

    # Register agents
    suite.add_agent(
        name="simple_chain",
        run_fn=build_simple_chain(),
        description="Prompt -> LLM -> answer (no tools, single call)",
    )
    suite.add_agent(
        name="react_agent",
        run_fn=build_react_agent(),
        description="ReAct loop with tool access (langgraph prebuilt)",
    )
    suite.add_agent(
        name="custom_stategraph",
        run_fn=build_stategraph_agent(),
        description="StateGraph: classify -> route -> process -> respond",
    )

    # Add test cases
    suite.add_test_cases(create_test_cases())

    print(f"\n  Agents registered: {len(suite.agents)}")
    print(f"  Test cases loaded: {len(suite.test_cases)}")

    # Run the benchmark
    suite.run_benchmark()

    # Print the full report
    report = suite.generate_report()
    print(f"\n{report}")

    print("\n  View detailed runs in MLflow UI: http://127.0.0.1:5555")
    print("  Experiment: L3/M4_capstones/2_framework_benchmark")
    print("=" * 70)

    # Cleanup temp files
    for path in [
        "/tmp/benchmark_comparison.csv",
        "/tmp/benchmark_results.csv",
        "/tmp/benchmark_report.txt",
    ]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
