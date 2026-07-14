"""Coding agent with container-bound tools for SWE-Bench evaluation.

The agent explores a repository inside a Docker/Podman container,
understands the bug, and generates a unified diff patch.
"""

import re

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from harness import CONTAINER_RUNTIME, exec_in_container

_active_container: str | None = None


def set_active_container(container_id: str) -> None:
    """Set the container that tools will execute in."""
    global _active_container
    _active_container = container_id


# -- Container-bound tools -----------------------------------------------------

@tool
def read_file(path: str) -> str:
    """Read a file from the repository. Path is relative to the repo root."""
    if not _active_container:
        return "Error: no active container"
    rc, out, err = exec_in_container(
        _active_container, f"cat /workspace/repo/{path}"
    )
    if rc != 0:
        return f"Error reading {path}: {err.strip()}"
    if len(out) > 3000:
        return out[:3000] + f"\n... (truncated, {len(out)} chars total)"
    return out


@tool
def search_code(pattern: str) -> str:
    """Search for a pattern in the codebase using grep. Returns matching lines with file paths."""
    if not _active_container:
        return "Error: no active container"
    safe = pattern.replace("'", "'\\''")
    rc, out, err = exec_in_container(
        _active_container,
        f"cd /workspace/repo && grep -rn '{safe}' --include='*.py' | head -30",
    )
    if rc != 0 or not out.strip():
        return "No matches found."
    return out


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory. Path is relative to the repo root."""
    if not _active_container:
        return "Error: no active container"
    rc, out, err = exec_in_container(
        _active_container, f"ls -la /workspace/repo/{directory}"
    )
    if rc != 0:
        return f"Error listing {directory}: {err.strip()}"
    return out


TOOLS = [read_file, search_code, list_files]

SYSTEM_PROMPT = (
    "You are a software engineer fixing a bug in an open-source Python repository. "
    "You have tools to read files, search code, and list directories in the repo.\n\n"
    "Your workflow:\n"
    "1. Read the problem statement carefully\n"
    "2. Use search_code and list_files to find the relevant source files\n"
    "3. Use read_file to understand the code around the bug\n"
    "4. Generate a unified diff patch that fixes the issue\n\n"
    "Your final answer MUST be ONLY a valid unified diff patch "
    "(lines starting with ---, +++, @@, +, -, and context lines). "
    "No explanatory text — just the patch."
)


def build_agent(temperature: float):
    """Create a coding agent for SWE-Bench tasks."""
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-4-26b-a4b",
        temperature=temperature,
        max_tokens=1024,
    )
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


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
    prompt += (
        "\nExplore the codebase to understand the issue, "
        "then generate a unified diff patch that fixes it."
    )
    return prompt


def extract_patch(agent_output: str) -> str:
    """Extract a unified diff patch from the agent's response."""
    if not agent_output:
        return ""

    # Try to find a diff inside code fences
    fence_match = re.search(
        r"```(?:diff|patch)?\s*\n(.*?)```", agent_output, re.DOTALL
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        if "---" in candidate or "+++" in candidate:
            return candidate

    # Try to extract raw diff lines (--- / +++ / @@ blocks)
    lines = agent_output.split("\n")
    diff_lines: list[str] = []
    in_diff = False
    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            if line.strip() == "" and not any(
                l.startswith(("---", "+++", "@@", "+", "-", " "))
                for l in [line]
            ):
                # End of diff block on blank non-context line
                if len(diff_lines) > 2:
                    break

    if diff_lines and len(diff_lines) >= 3:
        return "\n".join(diff_lines)

    return ""
