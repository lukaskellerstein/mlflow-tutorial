"""
L3-2.4 — Building Custom Autolog Integrations

Build a production-quality custom autolog implementation that:
  1. Explains how MLflow autolog works (monkey-patching / wrapper pattern)
  2. Creates a "SimpleChat" framework wrapping ChatOpenAI
  3. Builds an autolog function that monkey-patches SimpleChat methods
  4. Demonstrates enable/disable lifecycle with configurable options
  5. Adds callback hooks for extensibility
"""

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# ---------------------------------------------------------------------------
# 1. SimpleChat framework — a thin wrapper around ChatOpenAI
# ---------------------------------------------------------------------------
class SimpleChat:
    """A minimal chat framework that wraps ChatOpenAI.

    This represents any third-party framework that lacks native MLflow
    integration. Our custom autolog will instrument it transparently.
    """

    def __init__(self, model: str = "google/gemma-4-26b-a4b", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self._llm = ChatOpenAI(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key=SecretStr("lm-studio"),
            temperature=temperature,
        )
        self._call_count = 0

    def chat(self, message: str) -> str:
        """Send a single message and return the response text."""
        self._call_count += 1
        response = self._llm.invoke(message)
        return str(response.content)

    def chat_with_history(self, messages: list[dict[str, str]]) -> str:
        """Send a conversation history and return the response.

        Each dict has keys 'role' ('user' or 'assistant') and 'content'.
        """
        self._call_count += 1
        response = self._llm.invoke(messages)
        return str(response.content)

    def batch_chat(self, messages: list[str]) -> list[str]:
        """Send multiple independent messages and return all responses."""
        self._call_count += len(messages)
        return [self.chat(m) for m in messages]

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# 2. Autolog state — tracks patches and configuration
# ---------------------------------------------------------------------------
@dataclass
class _AutologState:
    """Internal state for the SimpleChat autolog integration."""

    enabled: bool = False
    log_traces: bool = True
    log_metrics: bool = True
    callbacks: list[Callable] = field(default_factory=list)
    _original_chat: Callable | None = None
    _original_chat_with_history: Callable | None = None
    _original_batch_chat: Callable | None = None


_state = _AutologState()


# ---------------------------------------------------------------------------
# 3. Wrapper functions that add MLflow instrumentation
# ---------------------------------------------------------------------------
def _make_chat_wrapper(original_fn: Callable) -> Callable:
    """Create a wrapper for SimpleChat.chat that logs to MLflow."""

    @functools.wraps(original_fn)
    def wrapper(self: SimpleChat, message: str) -> str:
        start = time.time()

        # Create an MLflow trace span for this call
        with mlflow.start_span(name="SimpleChat.chat") as span:
            span.set_inputs({"message": message})
            span.set_attributes(
                {
                    "simplechat.model": self.model,
                    "simplechat.temperature": self.temperature,
                    "simplechat.method": "chat",
                }
            )

            result = original_fn(self, message)
            elapsed = time.time() - start

            span.set_outputs({"response": result})
            span.set_attributes({"simplechat.latency_s": round(elapsed, 3)})

        # Log metrics if enabled
        if _state.log_metrics:
            call_num = self.call_count
            mlflow.log_metrics(
                {
                    "chat_latency_s": round(elapsed, 3),
                    "chat_input_chars": len(message),
                    "chat_output_chars": len(result),
                },
                step=call_num,
            )

        # Fire registered callbacks
        for cb in _state.callbacks:
            try:
                cb("chat", message, result, elapsed)
            except Exception:
                pass

        return result

    return wrapper


def _make_chat_with_history_wrapper(original_fn: Callable) -> Callable:
    """Create a wrapper for SimpleChat.chat_with_history."""

    @functools.wraps(original_fn)
    def wrapper(self: SimpleChat, messages: list[dict[str, str]]) -> str:
        start = time.time()

        with mlflow.start_span(name="SimpleChat.chat_with_history") as span:
            span.set_inputs({"messages": messages, "turn_count": len(messages)})
            span.set_attributes(
                {
                    "simplechat.model": self.model,
                    "simplechat.temperature": self.temperature,
                    "simplechat.method": "chat_with_history",
                    "simplechat.history_length": len(messages),
                }
            )

            result = original_fn(self, messages)
            elapsed = time.time() - start

            span.set_outputs({"response": result})
            span.set_attributes({"simplechat.latency_s": round(elapsed, 3)})

        if _state.log_metrics:
            call_num = self.call_count
            total_input_chars = sum(len(m["content"]) for m in messages)
            mlflow.log_metrics(
                {
                    "history_latency_s": round(elapsed, 3),
                    "history_turns": len(messages),
                    "history_input_chars": total_input_chars,
                    "history_output_chars": len(result),
                },
                step=call_num,
            )

        for cb in _state.callbacks:
            try:
                cb("chat_with_history", messages, result, elapsed)
            except Exception:
                pass

        return result

    return wrapper


def _make_batch_chat_wrapper(original_fn: Callable) -> Callable:
    """Create a wrapper for SimpleChat.batch_chat."""

    @functools.wraps(original_fn)
    def wrapper(self: SimpleChat, messages: list[str]) -> list[str]:
        start = time.time()

        with mlflow.start_span(name="SimpleChat.batch_chat") as span:
            span.set_inputs({"messages": messages, "batch_size": len(messages)})
            span.set_attributes(
                {
                    "simplechat.model": self.model,
                    "simplechat.temperature": self.temperature,
                    "simplechat.method": "batch_chat",
                    "simplechat.batch_size": len(messages),
                }
            )

            result = original_fn(self, messages)
            elapsed = time.time() - start

            span.set_outputs({"responses": result})
            span.set_attributes({"simplechat.latency_s": round(elapsed, 3)})

        if _state.log_metrics:
            call_num = self.call_count
            mlflow.log_metrics(
                {
                    "batch_latency_s": round(elapsed, 3),
                    "batch_size": len(messages),
                    "batch_avg_latency_s": round(elapsed / max(len(messages), 1), 3),
                },
                step=call_num,
            )

        for cb in _state.callbacks:
            try:
                cb("batch_chat", messages, result, elapsed)
            except Exception:
                pass

        return result

    return wrapper


# ---------------------------------------------------------------------------
# 4. Public autolog API — the main entry point
# ---------------------------------------------------------------------------
def simplechat_autolog(
    disable: bool = False,
    log_traces: bool = True,
    log_metrics: bool = True,
) -> None:
    """Enable or disable automatic MLflow logging for SimpleChat.

    When enabled, every call to SimpleChat.chat(), chat_with_history(),
    and batch_chat() is automatically instrumented with:
      - MLflow trace spans (inputs, outputs, attributes)
      - Metrics (latency, character counts, batch sizes)
      - Registered callback hooks

    This mirrors how mlflow.langchain.autolog() works internally:
      1. Save references to the original (unpatched) methods
      2. Replace them with instrumented wrappers via monkey-patching
      3. On disable, restore the originals

    Args:
        disable:     If True, remove patches and restore original methods.
        log_traces:  If True, create MLflow trace spans for each call.
        log_metrics: If True, log per-call metrics to the active run.
    """
    # NOTE: no `global _state` needed -- we only mutate the object's attributes,
    # never rebind the module-level name itself.
    if disable:
        # Restore original methods if we previously patched them
        if _state._original_chat is not None:
            SimpleChat.chat = _state._original_chat
        if _state._original_chat_with_history is not None:
            SimpleChat.chat_with_history = _state._original_chat_with_history
        if _state._original_batch_chat is not None:
            SimpleChat.batch_chat = _state._original_batch_chat
        _state.enabled = False
        _state._original_chat = None
        _state._original_chat_with_history = None
        _state._original_batch_chat = None
        return

    # Save originals (only if not already patched)
    if not _state.enabled:
        _state._original_chat = SimpleChat.chat
        _state._original_chat_with_history = SimpleChat.chat_with_history
        _state._original_batch_chat = SimpleChat.batch_chat

    # Apply configuration
    _state.log_traces = log_traces
    _state.log_metrics = log_metrics
    _state.enabled = True

    # Monkey-patch the class methods
    assert _state._original_chat is not None
    assert _state._original_chat_with_history is not None
    assert _state._original_batch_chat is not None
    SimpleChat.chat = _make_chat_wrapper(_state._original_chat)
    SimpleChat.chat_with_history = _make_chat_with_history_wrapper(_state._original_chat_with_history)
    SimpleChat.batch_chat = _make_batch_chat_wrapper(_state._original_batch_chat)


def register_callback(callback: Callable) -> None:
    """Register a callback that fires after every SimpleChat call.

    The callback signature is:
        callback(method_name: str, input: Any, output: Any, elapsed: float)
    """
    _state.callbacks.append(callback)


def clear_callbacks() -> None:
    """Remove all registered callbacks."""
    _state.callbacks.clear()


# ---------------------------------------------------------------------------
# 5. Main — demonstrate the full autolog lifecycle
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L3-2.4 — Building Custom Autolog Integrations")
    print("=" * 60)

    # ── Part 1: Autolog Architecture ──────────────────────────────
    print("\n--- Part 1: How MLflow Autolog Works ---")
    print("""
  MLflow's built-in autolog (e.g. mlflow.langchain.autolog) works by:

    1. SAVE originals — keep a reference to the unpatched method
    2. WRAP — create a new function that adds logging around the original
    3. MONKEY-PATCH — replace the class method with the wrapped version
    4. RESTORE on disable — swap the original method back in

  Key MLflow internals used:
    - @autologging_integration(name) — decorator that registers the flavor
    - safe_patch(flavor, class, method, patch_fn) — applies the monkey-patch
    - gorilla.get_original_attribute() — retrieves the pre-patch method
    - AutoLoggingConfig — stores per-flavor enable/disable state

  Our simplechat_autolog() implements the same pattern from scratch.
""")

    # ── Part 2: Build and test SimpleChat ─────────────────────────
    print("--- Part 2: SimpleChat Framework (unpatched) ---")
    bot = SimpleChat(model="google/gemma-4-26b-a4b", temperature=0.5)
    print(f"  Created SimpleChat(model={bot.model}, temperature={bot.temperature})")

    response = bot.chat("What is 2 + 2? Reply with just the number.")
    print(f"  Unpatched chat response: {response[:80]}")
    print(f"  Call count: {bot.call_count}")

    # ── Part 3: Enable autolog and run calls ──────────────────────
    print("\n--- Part 3: Enable Autolog & Run Calls ---")
    simplechat_autolog(log_traces=True, log_metrics=True)
    print("  simplechat_autolog() enabled")
    print("  Patches applied to: chat, chat_with_history, batch_chat")

    with mlflow.start_run(run_name="autolog_enabled") as run:
        print(f"\n  Run ID: {run.info.run_id[:8]}...")

        # Call 1: simple chat
        print("\n  [Call 1] chat()")
        r1 = bot.chat("Name three primary colors. Be brief.")
        print(f"    Response: {r1[:80]}")

        # Call 2: chat with history
        print("\n  [Call 2] chat_with_history()")
        r2 = bot.chat_with_history(
            [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Name one popular Python framework. Be brief."},
            ]
        )
        print(f"    Response: {r2[:80]}")

        # Call 3: another chat
        print("\n  [Call 3] chat()")
        r3 = bot.chat("What is the capital of France? Reply with just the city name.")
        print(f"    Response: {r3[:80]}")

        print(f"\n  Total calls so far: {bot.call_count}")
        print("  Check MLflow UI for auto-logged traces and metrics.")

    # ── Part 4: Disable and verify no logging ─────────────────────
    print("\n--- Part 4: Disable Autolog ---")
    simplechat_autolog(disable=True)
    print("  simplechat_autolog(disable=True) called")
    print("  Original methods restored")

    with mlflow.start_run(run_name="autolog_disabled") as run:
        print(f"\n  Run ID: {run.info.run_id[:8]}...")
        r4 = bot.chat("Say hello. Be brief.")
        print(f"  [Call 4] Unpatched response: {r4[:80]}")
        print("  No traces or metrics should appear for this call.")

    # ── Part 5: Callback hooks ────────────────────────────────────
    print("\n--- Part 5: Callback Hooks ---")

    call_log: list[dict[str, Any]] = []

    def logging_callback(method: str, inp: Any, out: Any, elapsed: float) -> None:
        """Custom callback that collects call metadata."""
        entry = {
            "method": method,
            "input_preview": str(inp)[:50],
            "output_preview": str(out)[:50],
            "latency_s": round(elapsed, 3),
        }
        call_log.append(entry)
        print(f"    [callback] {method} completed in {elapsed:.3f}s")

    register_callback(logging_callback)
    simplechat_autolog(log_traces=True, log_metrics=True)
    print("  Autolog re-enabled with custom callback registered")

    with mlflow.start_run(run_name="autolog_with_callbacks") as run:
        print(f"\n  Run ID: {run.info.run_id[:8]}...")

        r5 = bot.chat("What is 10 * 5? Reply with just the number.")
        print(f"    Response: {r5[:80]}")

        r6 = bot.chat("What color is the sky? Reply in one word.")
        print(f"    Response: {r6[:80]}")

        # Log the callback data as a table artifact
        if call_log:
            df = pd.DataFrame(call_log)
            table_path = "/tmp/autolog_callback_log.csv"
            df.to_csv(table_path, index=False)
            mlflow.log_artifact(table_path, artifact_path="callback_logs")
            print(f"\n  Callback log ({len(call_log)} entries) saved as artifact")
            for entry in call_log:
                print(f"    - {entry['method']}: {entry['latency_s']}s")

    # Clean up
    simplechat_autolog(disable=True)
    clear_callbacks()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total SimpleChat calls: {bot.call_count}")
    print("  Autolog pattern: monkey-patch class methods with wrappers")
    print("  Features demonstrated:")
    print("    - Enable/disable lifecycle")
    print("    - Automatic trace spans with inputs/outputs/attributes")
    print("    - Per-call metric logging (latency, char counts)")
    print("    - Extensible callback hooks")
    print("    - Restore original methods on disable")
    print("\n  View runs in MLflow UI: http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L3/M3_extensibility/1_custom_autolog")
    main()
