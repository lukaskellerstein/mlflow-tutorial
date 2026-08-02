# L3-2.4 — Building Custom Autolog Integrations

**Level:** Expert
**Duration:** 2 hours

## Overview

MLflow provides built-in autolog integrations for major frameworks (LangChain, scikit-learn, PyTorch, etc.), but many real-world frameworks lack native support. This lesson teaches you how to build your own autolog integration from scratch, using the same monkey-patching pattern that MLflow uses internally. You will create a reusable autolog function for a custom chat framework, complete with trace spans, metric logging, enable/disable lifecycle, and callback hooks.

## Prerequisites

- Completed: L1-M3.2 (LLM/GenAI Autologging), L1-M5.2 (Manual Tracing)
- Completed: L3-M2.1 through L3-M2.3 (Custom integration patterns)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### How MLflow Autolog Works Internally

MLflow's autolog integrations (e.g., `mlflow.langchain.autolog()`) follow a consistent architecture:

1. **Save originals** -- keep a reference to the unpatched class method
2. **Create wrappers** -- write new functions that add logging around the original
3. **Monkey-patch** -- replace the class method on the class itself (not on instances)
4. **Restore on disable** -- swap the original method back when autolog is turned off

The key MLflow internals involved are:
- `@autologging_integration(name)` -- decorator that registers the flavor with MLflow's autolog registry
- `safe_patch(flavor, class, method_name, patch_fn)` -- applies the monkey-patch safely, catching errors in the patch code without breaking the original method
- `gorilla.get_original_attribute()` -- retrieves the pre-patch method reference
- `AutoLoggingConfig` -- stores per-flavor configuration (enabled, log_traces, etc.)

### Monkey-Patching vs. Decorator-Based Approaches

| Approach | Pros | Cons |
|----------|------|------|
| Monkey-patching (class-level) | Transparent to users, no code changes needed | Must carefully manage original references, thread safety |
| Decorator-based | Explicit, easy to understand | Requires user to modify their code |
| Callback/hook-based | Framework provides extension points | Only works if framework supports it |

MLflow uses monkey-patching because it provides the most transparent experience -- users call `autolog()` once and everything is instrumented. Our implementation follows the same pattern.

### What Gets Logged

A well-designed autolog integration captures:
- **Trace spans** with inputs, outputs, and attributes (model name, parameters)
- **Metrics** per call (latency, token counts, input/output sizes)
- **Model parameters** as run params or span attributes
- **Errors** as span events or logged exceptions

## Step-by-Step

### Step 1: Define the Target Framework

We create `SimpleChat`, a minimal chat framework wrapping ChatOpenAI. It has three methods that we want to instrument:

```python
class SimpleChat:
    def chat(self, message: str) -> str: ...
    def chat_with_history(self, messages: list[dict]) -> str: ...
    def batch_chat(self, messages: list[str]) -> list[str]: ...
```

### Step 2: Build Wrapper Functions

Each wrapper follows the same pattern -- save the start time, create an MLflow span, call the original, log outputs and metrics:

```python
def _make_chat_wrapper(original_fn):
    @functools.wraps(original_fn)
    def wrapper(self, message):
        start = time.time()
        with mlflow.start_span(name="SimpleChat.chat") as span:
            span.set_inputs({"message": message})
            result = original_fn(self, message)
            span.set_outputs({"response": result})
        if _state.log_metrics:
            mlflow.log_metrics({"chat_latency_s": time.time() - start})
        return result

    return wrapper
```

### Step 3: Implement the Autolog Function

The public API mirrors MLflow's convention with `disable`, `log_traces`, and `log_metrics` parameters:

```python
def simplechat_autolog(disable=False, log_traces=True, log_metrics=True):
    if disable:
        # Restore originals
        SimpleChat.chat = _state._original_chat
        ...
        return
    # Save originals, then monkey-patch
    _state._original_chat = SimpleChat.chat
    SimpleChat.chat = _make_chat_wrapper(_state._original_chat)
    ...
```

### Step 4: Add Callback Hooks

For extensibility, we support registering callbacks that fire after every call:

```python
def register_callback(callback):
    _state.callbacks.append(callback)
```

### Step 5: Test the Full Lifecycle

The main function demonstrates: enable autolog, run calls (verify logging), disable autolog, run another call (verify no logging), then re-enable with callbacks.

## Running the Lesson

```bash
cd tutorial/level_3/M2_custom_integrations/4_custom_autolog
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-2.4 — Building Custom Autolog Integrations
============================================================

--- Part 1: How MLflow Autolog Works ---
  [architecture explanation printed]

--- Part 2: SimpleChat Framework (unpatched) ---
  Created SimpleChat(model=google/gemma-4-26b-a4b, temperature=0.5)
  Unpatched chat response: 4
  Call count: 1

--- Part 3: Enable Autolog & Run Calls ---
  simplechat_autolog() enabled
  [Call 1] chat()    Response: Red, blue, yellow...
  [Call 2] chat_with_history()    Response: Django...
  [Call 3] chat()    Response: Paris

--- Part 4: Disable Autolog ---
  [Call 4] Unpatched response: Hello!
  No traces or metrics should appear for this call.

--- Part 5: Callback Hooks ---
  [callback] chat completed in 1.234s
  [callback] chat completed in 0.987s
  Callback log (2 entries) saved as artifact

============================================================
Summary
============================================================
  Total SimpleChat calls: 8
```

In the MLflow UI you will see:
- Three runs: `autolog_enabled`, `autolog_disabled`, `autolog_with_callbacks`
- The `autolog_enabled` run has trace spans for all three call types
- The `autolog_disabled` run has no traces
- The `autolog_with_callbacks` run has traces plus a callback log artifact

## Key Takeaways

- MLflow autolog works by monkey-patching class methods with instrumented wrappers
- The pattern is: save original -> wrap with logging -> replace on class -> restore on disable
- Custom autolog integrations should log trace spans (inputs/outputs), metrics (latency), and model attributes
- Callback hooks make autolog extensible without modifying the core integration
- Always test the disable path to ensure original behavior is fully restored

## Next Steps

Continue to L3-M3.1 (Production Tracing at Scale) to learn how to handle high-volume trace collection, sampling strategies, and trace-based SLO monitoring in production environments.
