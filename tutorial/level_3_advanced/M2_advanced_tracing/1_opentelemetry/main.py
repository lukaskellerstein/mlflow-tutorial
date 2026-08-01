"""
L2-M4.3 — OpenTelemetry Integration

Demonstrates that MLflow tracing is built on OpenTelemetry (OTel):
- Part 1: MLflow traces use OTel spans under the hood
- Part 2: Custom OTel SpanProcessor with MLflow's tracer provider
- Part 3: Exporting traces via ConsoleSpanExporter
- Part 4: Combining MLflow traces with custom OTel spans
"""

import time

import mlflow
from mlflow.tracing.provider import provider as mlflow_provider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L3/M2_advanced_tracing/1_opentelemetry")


# ── Helper classes ────────────────────────────────────────


class _NoOpExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class PrintingSpanProcessor(SimpleSpanProcessor):
    """Custom OTel SpanProcessor that prints span lifecycle events."""

    def __init__(self) -> None:
        self.started = 0
        self.ended = 0
        super().__init__(span_exporter=_NoOpExporter())

    def on_start(self, span, parent_context=None) -> None:
        self.started += 1
        print(f"    [Processor] STARTED: {span.name!r}")

    def on_end(self, span) -> None:
        self.ended += 1
        dur_ms = ((span.end_time or 0) - (span.start_time or 0)) / 1_000_000
        print(
            f"    [Processor] ENDED:   {span.name!r} ({dur_ms:.1f}ms, {span.status.status_code.name})"
        )


def _remove_processor(tp, target) -> None:
    """Remove a specific processor instance from a TracerProvider."""
    if hasattr(tp, "_active_span_processor"):
        proc = tp._active_span_processor
        if hasattr(proc, "_span_processors"):
            proc._span_processors = tuple(sp for sp in proc._span_processors if sp is not target)


# ── Part 1: MLflow tracing IS OpenTelemetry ───────────────


def part1_otel_foundation() -> None:
    print("=" * 60)
    print("Part 1: MLflow Tracing is Built on OpenTelemetry")
    print("=" * 60)
    print("  MLflow creates standard OTel spans via an isolated TracerProvider.")
    print("  Every @mlflow.trace call goes through OTel's Tracer.start_span().\n")

    @mlflow.trace(name="otel_demo_function")
    def demo(x: int, y: int) -> int:
        return x + y

    print(f"  demo(3, 7) = {demo(3, 7)}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}")

    tp = mlflow_provider.get()
    print(f"\n  TracerProvider type: {type(tp).__name__}")
    print(f"  Is OTel SDK TracerProvider? {isinstance(tp, TracerProvider)}")
    if hasattr(tp, "_active_span_processor") and hasattr(
        tp._active_span_processor, "_span_processors"
    ):
        for i, sp in enumerate(tp._active_span_processor._span_processors):
            print(f"  Registered processor [{i}]: {type(sp).__name__}")
    print()


# ── Part 2: Custom SpanProcessor ──────────────────────────


def part2_custom_span_processor() -> None:
    print("=" * 60)
    print("Part 2: Custom SpanProcessor")
    print("=" * 60)
    print("  SpanProcessors receive on_start/on_end callbacks for every span.")
    print("  Use them for debugging, filtering, or routing to extra backends.\n")

    tp = mlflow_provider.get()
    proc = PrintingSpanProcessor()
    tp.add_span_processor(proc)

    @mlflow.trace(name="math_pipeline")
    def pipeline(vals: list[int]) -> dict:
        s = add(vals)
        return {"sum": s, "avg": s / len(vals)}

    @mlflow.trace(name="add")
    def add(vals: list[int]) -> int:
        return sum(vals)

    result = pipeline([10, 20, 30, 40, 50])
    print(f"\n  Result: {result}")
    print(f"  Processor saw {proc.started} starts, {proc.ended} ends")
    _remove_processor(tp, proc)
    print()


# ── Part 3: Export concepts ───────────────────────────────


def part3_export_concepts() -> None:
    print("=" * 60)
    print("Part 3: Exporting Traces to OTel Backends")
    print("=" * 60)
    print("  MLflow can dual-export to OTel collectors via env vars:")
    print("    MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true")
    print("    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317")
    print()
    print("  Live demo with ConsoleSpanExporter (prints span JSON):\n")

    tp = mlflow_provider.get()
    console_proc = SimpleSpanProcessor(ConsoleSpanExporter())
    tp.add_span_processor(console_proc)

    @mlflow.trace(name="console_export_demo")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    greet("OpenTelemetry")
    time.sleep(0.2)
    _remove_processor(tp, console_proc)
    print()


# ── Part 4: Combining MLflow + custom OTel spans ─────────


def part4_combined_tracing() -> None:
    print("=" * 60)
    print("Part 4: Combining MLflow + Custom OTel Spans")
    print("=" * 60)
    print("  Raw OTel spans coexist with MLflow spans in the same trace.")
    print("  MLflow spans have inputs/outputs; OTel spans have attributes/events.\n")

    custom_tracer = mlflow_provider.get().get_tracer("custom-instrumentation")

    @mlflow.trace(name="data_pipeline")
    def pipeline(records: list[dict]) -> dict:
        valid = validate(records)
        return {"processed": len(valid), "records": enrich(valid)}

    def validate(records: list[dict]) -> list[dict]:
        with custom_tracer.start_as_current_span("validate_records") as span:
            valid = [r for r in records if r.get("name") and r.get("value")]
            span.set_attribute("input_count", len(records))
            span.set_attribute("valid_count", len(valid))
            span.add_event("validation_done", {"rejected": len(records) - len(valid)})
            return valid

    @mlflow.trace(name="enrich_records")
    def enrich(records: list[dict]) -> list[dict]:
        return [{**r, "score": len(r["name"]) * r["value"]} for r in records]

    data = [
        {"name": "alpha", "value": 10},
        {"name": "beta", "value": 20},
        {"name": "", "value": 30},
        {"name": "delta", "value": 40},
        {"value": 50},
    ]
    result = pipeline(data)
    print(f"  Processed {result['processed']}/{len(data)} records")
    for r in result["records"]:
        print(f"    {r['name']}: value={r['value']}, score={r['score']}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}\n")


# ── Main ──────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("L2-M4.3 -- OpenTelemetry Integration")
    print("=" * 60 + "\n")

    part1_otel_foundation()
    part2_custom_span_processor()
    part3_export_concepts()
    part4_combined_tracing()

    print("=" * 60)
    print("Done! Check MLflow UI at http://127.0.0.1:5555")
    print("Experiment: L2/M4_advanced_tracing/3_opentelemetry")
    print("=" * 60)


if __name__ == "__main__":
    main()
