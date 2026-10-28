"""OpenTelemetry's SDK with its in-memory exporter, for the tests that read spans.

A process has one tracer provider, so `install` sets it once, the first time a test
asks for spans; tests that run before then send no trace context, as a deployment
without OpenTelemetry configured sends none. Every span any owner in the process ends
is exported to EXPORTER; `install` clears it, and a test reads what its own run ended.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

EXPORTER = InMemorySpanExporter()


def install() -> None:
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(provider)
    EXPORTER.clear()


def ended(prefix: str = "") -> list[ReadableSpan]:
    """Every span ended since the last clear whose name starts with `prefix`."""
    return [s for s in EXPORTER.get_finished_spans() if s.name.startswith(prefix)]
