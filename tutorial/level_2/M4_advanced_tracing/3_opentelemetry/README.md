# L2-M4.3 -- OpenTelemetry Integration

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

MLflow's tracing system is built directly on the OpenTelemetry (OTel) SDK. This lesson explores that foundation: how MLflow manages its TracerProvider, how to add custom SpanProcessors, how to export traces to OTel-compatible backends, and how to combine MLflow-managed spans with raw OTel spans in a single trace.

## Prerequisites

- Completed: L1-M5.1 (Auto Tracing), L1-M5.2 (Manual Tracing)
- MLflow server running at http://127.0.0.1:5000
- Basic understanding of distributed tracing concepts (spans, traces, context propagation)

## Concepts

### MLflow's OTel Architecture

When you call `@mlflow.trace` or `mlflow.start_span()`, MLflow does not use a proprietary tracing format. Instead, it creates standard OpenTelemetry spans via the OTel SDK:

```
@mlflow.trace  -->  mlflow.start_span()  -->  OTel Tracer.start_span()
                                                     |
                                              TracerProvider
                                                     |
                                         +----------+----------+
                                         |                     |
                                   MLflow Span           Custom Span
                                   Processor             Processors
                                         |                     |
                                   MLflow Server         Jaeger / Zipkin /
                                                         Tempo / Console
```

By default, MLflow creates an **isolated** TracerProvider that does not interfere with any global OTel setup in your application. This means MLflow tracing and other OTel instrumentation (e.g., FastAPI, gRPC) can coexist without conflict.

### SpanProcessors

OTel SpanProcessors sit between span creation and export. They receive `on_start()` and `on_end()` callbacks for every span, allowing you to:

- Debug by printing span details
- Filter spans before export
- Add or transform span attributes
- Route spans to multiple backends

### Export Options

MLflow exports traces to its tracking server by default. You can additionally export to any OTel-compatible backend using:

1. **OTLP Dual Export** -- Set `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true` and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to send traces to both MLflow and an OTel collector.
2. **Unified Provider Mode** -- Set `MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false` to share the global OTel TracerProvider, then configure exporters using standard OTel APIs.

### Distributed Tracing

Because MLflow uses standard OTel, traces can propagate across service boundaries using W3C Trace Context headers. MLflow provides helper functions for HTTP-based context propagation:

- `mlflow.tracing.get_tracing_context_headers_for_http_request()` -- extract headers to send
- `mlflow.tracing.set_tracing_context_from_http_request_headers()` -- inject headers on the receiving side

## Step-by-Step

### Step 1: MLflow Traces Use OTel Spans

The lesson first demonstrates that MLflow's TracerProvider is a standard OTel `TracerProvider` from `opentelemetry.sdk.trace`. When you create a trace with `@mlflow.trace`, it ultimately calls `Tracer.start_span()` from the OTel SDK.

```python
from mlflow.tracing.provider import provider as mlflow_provider

tracer_provider = mlflow_provider.get()
# This is an instance of opentelemetry.sdk.trace.TracerProvider
```

### Step 2: Custom SpanProcessor

We create a `PrintingSpanProcessor` that logs every span start and end to the console, then register it with MLflow's TracerProvider:

```python
tracer_provider.add_span_processor(custom_processor)
```

Every MLflow trace created after this will trigger the custom processor's callbacks, giving visibility into the span lifecycle.

### Step 3: ConsoleSpanExporter

The `ConsoleSpanExporter` from the OTel SDK prints full span JSON to stdout. Adding it to MLflow's TracerProvider shows exactly what data each span carries -- name, trace ID, span ID, attributes, timestamps, and status.

### Step 4: Mixed MLflow + OTel Spans

Finally, we create traces that mix MLflow-managed spans (with inputs/outputs visible in the MLflow UI) and raw OTel spans (with custom attributes and events). Both participate in the same trace context:

```python
custom_tracer = tracer_provider.get_tracer("custom-otel-instrumentation")

@mlflow.trace(name="pipeline")
def pipeline():
    with custom_tracer.start_as_current_span("db_query") as span:
        span.set_attribute("db.system", "postgresql")
        span.add_event("query_completed", {"row_count": 10})
```

## Running the Lesson

```bash
cd tutorial/level_2/M4_advanced_tracing/3_opentelemetry
uv sync
uv run python main.py
```

## Expected Output

You will see four parts printed to the console:

1. **Part 1** -- Confirms MLflow's TracerProvider is an OTel `TracerProvider` and lists the registered span processors.
2. **Part 2** -- The custom `PrintingSpanProcessor` prints `STARTED` and `ENDED` messages for each span as MLflow traces execute.
3. **Part 3** -- The `ConsoleSpanExporter` outputs raw JSON span data to stdout, plus configuration examples for Jaeger/Zipkin/Tempo.
4. **Part 4** -- Shows a data pipeline trace with both MLflow and raw OTel spans, demonstrating coexistence.

In the MLflow UI at http://127.0.0.1:5000, navigate to the experiment `L2/M4_advanced_tracing/3_opentelemetry` and explore the traces. MLflow-managed spans will show structured inputs and outputs; raw OTel spans will appear with their attributes but without MLflow-specific metadata.

## Key Takeaways

- MLflow tracing is built on OpenTelemetry -- every MLflow span is a standard OTel span.
- MLflow uses an isolated TracerProvider by default, avoiding conflicts with other OTel instrumentation.
- You can add custom SpanProcessors to MLflow's TracerProvider for debugging, filtering, or routing spans.
- Traces can be exported to any OTel-compatible backend (Jaeger, Zipkin, Tempo) via OTLP dual export or unified provider mode.
- MLflow-managed spans and raw OTel spans can coexist in the same trace, combining MLflow's structured tracking with OTel's flexibility.

## Next Steps

In L2-M4.4 (Trace Analysis), you will learn how to query, analyze, and extract insights from the traces stored in MLflow -- building on the OTel foundation covered here to understand trace data at scale.
