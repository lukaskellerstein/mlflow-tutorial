"""DeepAgents-based coding agent with container sandbox for SWE-Bench evaluation.

Uses DeepAgents' BaseSandbox protocol to give the agent full file operations
(read, write, edit, grep, glob, ls, execute) inside a Docker/Podman container.
The agent edits files directly — no patch extraction needed.

LLM provider is configured via llm_config.yaml (LMStudio, OpenRouter, LiteLLM,
or MLflow AI Gateway). API keys starting with $ are read from env vars.
"""

import os
import subprocess
from pathlib import Path

import yaml
from deepagents import create_deep_agent
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from langchain_openai import ChatOpenAI

from harness import CONTAINER_RUNTIME, exec_in_container


# ── LLM config ──────────────────────────────────────────────────────────────

def load_llm_config() -> dict:
    """Load active LLM provider from llm_config.yaml."""
    config_path = Path(__file__).parent / "llm_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    active = config["active"]
    provider = config["providers"][active]

    api_key = provider["api_key"]
    if isinstance(api_key, str) and api_key.startswith("$"):
        env_var = api_key[1:]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise ValueError(
                f"Set {env_var} environment variable for provider '{active}'"
            )

    return {
        "provider": active,
        "base_url": provider["base_url"],
        "api_key": api_key,
        "model": provider["model"],
        "max_tokens": provider.get("max_tokens", 4096),
    }


# ── ContainerSandbox ─────────────────────────────────────────────────────────

class ContainerSandbox(BaseSandbox):
    """Sandbox backed by a running Docker/Podman container."""

    def __init__(self, container_id: str, runtime: str = CONTAINER_RUNTIME) -> None:
        self._container_id = container_id
        self._runtime = runtime

    @property
    def id(self) -> str:
        return self._container_id

    def execute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        t = timeout or 120
        rc, stdout, stderr = exec_in_container(self._container_id, command, timeout=t)
        output = stdout + stderr
        return ExecuteResponse(output=output, exit_code=rc, truncated=False)

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                proc = subprocess.run(
                    [self._runtime, "exec", "-i", self._container_id, "bash", "-c",
                     f"cat > {path}"],
                    input=content,
                    capture_output=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    results.append(FileUploadResponse(path=path, error=proc.stderr.decode()))
                else:
                    results.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                results.append(FileUploadResponse(path=path, error=str(e)))
        return results

    def download_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        results: list[FileDownloadResponse] = []
        for path in paths:
            rc, stdout, stderr = exec_in_container(
                self._container_id, f"cat {path}"
            )
            if rc != 0:
                results.append(FileDownloadResponse(
                    path=path, content=None, error=stderr.strip(),
                ))
            else:
                results.append(FileDownloadResponse(
                    path=path, content=stdout.encode(), error=None,
                ))
        return results


# ── Agent creation ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a software engineer fixing a bug in an open-source Python repository. "
    "The repository is checked out at /workspace/repo.\n\n"
    "You have tools to explore the codebase (ls, read_file, grep, glob), "
    "edit files (edit_file, write_file), and run commands (execute).\n\n"
    "Your workflow:\n"
    "1. Read the problem statement carefully\n"
    "2. Use grep and ls to find the relevant source files\n"
    "3. Use read_file to understand the code around the bug\n"
    "4. Use edit_file to fix the bug directly in the source files\n"
    "5. Use execute to run a quick sanity check if appropriate\n\n"
    "Edit files directly — do NOT generate diff/patch text. "
    "Make minimal, targeted changes that fix the issue without side effects."
)


def build_agent(temperature: float, container_id: str):
    """Create a DeepAgents coding agent backed by a container sandbox."""
    cfg = load_llm_config()
    print(f"    LLM: {cfg['provider']} / {cfg['model']}")
    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=temperature,
        max_tokens=cfg["max_tokens"],
    )
    sandbox = ContainerSandbox(container_id)
    return create_deep_agent(model=llm, backend=sandbox, system_prompt=SYSTEM_PROMPT)


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
        "then edit the source files to fix the bug."
    )
    return prompt
