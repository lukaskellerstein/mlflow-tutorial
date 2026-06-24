"""
L3-4.3 — MLflow MCP (Model Context Protocol) Integration

Demonstrates MCP patterns with MLflow tracing:
  1. MCP concept overview — what MCP is and how MLflow implements it
  2. MCP-style tool server — standardized tool interface with schema
  3. MCP-style client — discovers and calls tools, integrates with LLM
  4. Traced MCP interactions — multiple queries with full MLflow tracing
  5. MCP metrics and analysis — latency, success rate, overhead comparison
"""

import json
from typing import Any

import mlflow
import pandas as pd

from mcp_tools import MCPClient, build_server


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L3-4.3 — MLflow MCP (Model Context Protocol) Integration")
    print("=" * 60)

    # ---- Part 1: MCP concept overview ------------------------------------
    print("\n" + "=" * 60)
    print("Part 1: MCP Concept Overview")
    print("=" * 60)
    print("""
  Model Context Protocol (MCP) standardizes how AI applications
  connect to external tools and data sources. Key concepts:

  - Tool Server: Exposes tools with name, description, and JSON schema
  - Tool Client: Discovers tools, selects the right one, calls it
  - Protocol:    tools/list  -> enumerate available tools
                 tools/call  -> invoke a tool with arguments

  MLflow implements MCP natively — 'mlflow mcp run' starts a server
  that exposes MLflow operations (traces, experiments, runs, scorers,
  models) as MCP tools. AI assistants can query MLflow directly.

  This lesson simulates the MCP pattern in-process and traces every
  step with MLflow to show how interactions map to spans and metrics.
""")

    # ---- Part 2: MCP-style tool server -----------------------------------
    print("=" * 60)
    print("Part 2: MCP-Style Tool Server")
    print("=" * 60)
    server = build_server()
    manifests = server.list_tools()
    print(f"\n  Registered {len(manifests)} tools:")
    for m in manifests:
        print(f"    - {m['name']}: {m['description'][:60]}")

    # ---- Part 3: MCP-style client ----------------------------------------
    print("\n" + "=" * 60)
    print("Part 3: MCP-Style Client with LLM Tool Selection")
    print("=" * 60)
    client = MCPClient(server)
    discovered = client.discover_tools()
    print(f"\n  Client discovered {len(discovered)} tools from server")

    # ---- Part 4: Run MCP-style interactions ------------------------------
    print("\n" + "=" * 60)
    print("Part 4: MCP Interactions (fully traced in MLflow)")
    print("=" * 60)

    queries = [
        "What is 145 * 23 + 17?",
        "What is the weather like in Tokyo?",
        "Tell me about MLflow and its tracing capabilities.",
        "What is the weather in London and also calculate 99 / 3?",
    ]

    interaction_records: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="mcp_interactions") as parent_run:
        mlflow.log_params({
            "num_tools": len(manifests),
            "tool_names": json.dumps([m["name"] for m in manifests]),
            "llm_model": "gemma4:e2b",
            "num_queries": len(queries),
        })

        for idx, query in enumerate(queries, 1):
            print(f"\n  Query {idx}: {query}")
            with mlflow.start_run(run_name=f"query_{idx}", nested=True):
                result = client.execute_query(query)
                mlflow.log_params({
                    "query": query[:250],
                    "tool_selected": result["tool_selected"],
                    "arguments": json.dumps(result["arguments"])[:250],
                })
                mlflow.log_metrics({
                    "discovery_ms": result["timing"]["discovery_ms"],
                    "selection_ms": result["timing"]["selection_ms"],
                    "call_ms": result["timing"]["call_ms"],
                    "total_ms": result["timing"]["total_ms"],
                })
                has_error = "error" in result.get("result", {})
                mlflow.log_metric("success", 0 if has_error else 1)
                mlflow.set_tag("status", "ERROR" if has_error else "OK")

                interaction_records.append({
                    "query": query,
                    "tool_selected": result["tool_selected"],
                    "success": not has_error,
                    **result["timing"],
                })

                tool_result = result["result"].get("result", result["result"])
                t = result["timing"]
                print(f"    Tool:   {result['tool_selected']}")
                print(f"    Args:   {result['arguments']}")
                print(f"    Result: {json.dumps(tool_result)[:120]}")
                print(f"    Timing: {t['total_ms']:.0f}ms total "
                      f"(discover={t['discovery_ms']:.0f}ms, "
                      f"select={t['selection_ms']:.0f}ms, "
                      f"call={t['call_ms']:.0f}ms)")

        # ---- Part 5: MCP metrics and analysis ----------------------------
        print("\n" + "=" * 60)
        print("Part 5: MCP Metrics and Analysis")
        print("=" * 60)

        df = pd.DataFrame(interaction_records)
        success_rate = df["success"].mean()
        avg_total = df["total_ms"].mean()
        avg_discovery = df["discovery_ms"].mean()
        avg_selection = df["selection_ms"].mean()
        avg_call = df["call_ms"].mean()

        print(f"\n  Success rate:         {success_rate:.0%}")
        print(f"  Avg total latency:    {avg_total:.1f}ms")
        print(f"  Avg discovery time:   {avg_discovery:.1f}ms")
        print(f"  Avg LLM selection:    {avg_selection:.1f}ms")
        print(f"  Avg tool call:        {avg_call:.1f}ms")

        # Overhead comparison: MCP flow vs direct tool call
        print("\n  Overhead comparison (MCP protocol vs direct call):")
        overhead_records = []
        for _, row in df.iterrows():
            mcp_overhead = row["total_ms"] - row["call_ms"]
            overhead_records.append({
                "query": row["query"][:40],
                "tool": row["tool_selected"],
                "mcp_total_ms": row["total_ms"],
                "direct_ms": row["call_ms"],
                "overhead_ms": round(mcp_overhead, 1),
            })
            print(f"    {row['tool_selected']:25s} | MCP: {row['total_ms']:8.1f}ms | "
                  f"Direct: {row['call_ms']:6.1f}ms | "
                  f"Overhead: {mcp_overhead:8.1f}ms")

        overhead_df = pd.DataFrame(overhead_records)
        avg_overhead = overhead_df["overhead_ms"].mean()
        print(f"\n  Average MCP overhead:  {avg_overhead:.1f}ms")
        print("  (Overhead = discovery + LLM tool selection; the value-add")
        print("   is automatic tool routing without hard-coded logic.)")

        # Log aggregate metrics on parent run
        mlflow.log_metrics({
            "success_rate": success_rate,
            "avg_total_ms": round(avg_total, 1),
            "avg_discovery_ms": round(avg_discovery, 1),
            "avg_selection_ms": round(avg_selection, 1),
            "avg_call_ms": round(avg_call, 1),
            "avg_overhead_ms": round(avg_overhead, 1),
        })

        # Save detailed results as artifacts
        results_path = "/tmp/mcp_interaction_results.csv"
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, artifact_path="mcp_results")

        overhead_path = "/tmp/mcp_overhead_analysis.csv"
        overhead_df.to_csv(overhead_path, index=False)
        mlflow.log_artifact(overhead_path, artifact_path="mcp_results")

        print(f"\n  Parent run ID: {parent_run.info.run_id}")
        print(f"  View in MLflow UI: http://127.0.0.1:5000")

    print("\n" + "=" * 60)
    print("Done! Check MLflow UI for traces showing the full MCP flow:")
    print("  discover tools -> LLM selects tool -> call tool -> result")
    print("Each query creates a nested run with timing metrics.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M4_advanced_features/3_mcp")
    main()
