"""Claude Agent SDK coding agent that runs INSIDE the evaluation container.

The SDK on the host does not spawn a local `claude` binary: cli_path points
at container_claude.sh, which execs `podman exec -i ... claude`, so the
agent process -- and every tool it uses (Bash, Read, Edit, Grep, Glob) --
runs inside the instance's container, next to the repo at /workspace/repo.
The host keeps only orchestration, scoring, and MLflow tracing.

Isolation comes from the container boundary itself: nothing of the agent
runs on the host, so no tool restriction is needed to protect it.

Auth: the in-container CLI cannot reach the host's Keychain login, so it
needs CLAUDE_CODE_OAUTH_TOKEN in the host environment (created once with
`claude setup-token`); the wrapper forwards it into the container.
"""

import json
import os
from typing import Any

import mlflow
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    EffortLevel,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

MODEL = "claude-sonnet-5"
MAX_TURNS = 50
MAX_BUDGET_USD = 1.0
CLI_WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "container_claude.sh")

# The SDK's pre-connect version check spawns cli_path with the plain host
# environment -- before ClaudeAgentOptions.env is applied -- so the wrapper
# would run without SWE_CONTAINER_ID and fail. Skip it; the CLI version in
# the image is pinned by the Dockerfile build, not worth a handshake.
os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")

SYSTEM_PROMPT = (
    "You are a software engineer fixing a bug in an open-source Python repository. "
    "You are working inside a sandbox container; the repository is checked out at "
    "/workspace/repo, which is your working directory.\n\n"
    "Your workflow:\n"
    "1. Read the problem statement carefully\n"
    "2. Use Grep and Glob to find the relevant source files\n"
    "3. Read the code around the bug\n"
    "4. Edit the source files to fix the bug\n"
    "5. Run a quick sanity check with Bash if appropriate\n\n"
    "Edit files directly -- do NOT generate diff/patch text. "
    "Make minimal, targeted changes that fix the issue without side effects."
)


def build_options(effort: EffortLevel, container_id: str) -> ClaudeAgentOptions:
    """Agent options for one configuration, bound to one instance's container.

    tools=[...] grants the CLI's own built-in tools -- they operate inside
    the container because the CLI process itself does. setting_sources=[]
    and strict_mcp_config keep the run hermetic.
    """
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        effort=effort,
        tools=["Bash", "Read", "Edit", "Write", "Grep", "Glob"],
        cli_path=CLI_WRAPPER,
        env={"SWE_CONTAINER_ID": container_id},
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        permission_mode="bypassPermissions",
        strict_mcp_config=True,
        setting_sources=[],
    )


@mlflow.trace(name="swe_bench_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> dict:
    """Execute one agent session and capture response, tool calls, and SDK metrics."""
    response_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    num_turns = 0
    cost_usd: float | None = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes({"tool_name": block.name, "tool_use_id": block.id})
                        tool_calls.append({"name": block.name, "input": block.input})
            elif isinstance(message, ResultMessage):
                num_turns = message.num_turns
                cost_usd = message.total_cost_usd

    return {
        "response": "".join(response_parts),
        "tool_calls": tool_calls,
        "num_turns": num_turns,
        "cost_usd": cost_usd,
    }


def build_prompt(instance: dict) -> str:
    """Build the user message from a SWE-Bench instance."""
    hints = instance.get("hints_text", "") or ""
    prompt = (
        f"Repository: {instance['repo']}\n"
        f"Instance: {instance['instance_id']}\n\n"
        f"## Problem Statement\n{instance['problem_statement']}\n"
    )
    if hints.strip():
        prompt += f"\n## Hints\n{hints}\n"
    prompt += "\nExplore the codebase to understand the issue, then edit the source files to fix the bug."
    return prompt


def summarize_tool_calls(tool_calls: list[dict]) -> str:
    """Compact per-call log for the MLflow artifact."""
    lines = []
    for tc in tool_calls:
        lines.append(f"{tc['name']}({json.dumps(tc['input'])[:200]})")
    return "\n".join(lines) or "(no tool calls)"
