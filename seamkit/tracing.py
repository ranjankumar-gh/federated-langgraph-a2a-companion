"""Trace what you own: one span per crossing, ended at the seam.

The caller's span covers what the caller's code did: the send or the read, up to the
answer coming back. It is named after the seam, never after anything inside the
partner, and its attributes are names and ids: the seam, the skill, the call, the task
id, the key, the partner's release, and the names of the fields sent. Never a value: a
tracing backend is one more place data goes, and the data-class policy never approved
it.

The trace context travels in the A2A message's metadata, under `traceparent` (and
`tracestate`, when there is one), as W3C Trace Context spells them. A task outlives the
request that opened it, so an HTTP header is not enough: a read 21 minutes later, or a
reply to a paused task, is another request, and the ledger entry keeps the traceparent
of the send that opened the task.

The partner starts a trace of its own, with a span link to the caller's context. It
never makes a child span of it, so the caller's sampling and retention decide nothing
about the partner's spans, and the partner's decide nothing about the caller's.
"""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from typing import Any

from google.protobuf.struct_pb2 import Struct
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, Span, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

TRACER = trace.get_tracer("seamkit.seam")
W3C = TraceContextTextMapPropagator()
CARRIED = ("traceparent", "tracestate")


def inject(metadata: MutableMapping[str, Any] | Struct) -> None:
    """Write the current span's context into a message's metadata, W3C-spelled."""
    carrier: dict[str, str] = {}
    W3C.inject(carrier)
    metadata.update(carrier)


def links(metadata: Mapping[str, Any] | Struct) -> list[Link]:
    """A link to the context a message or a ledger entry carries; [] if none."""
    carrier = {k: str(metadata[k]) for k in CARRIED if k in metadata and metadata[k]}
    span = trace.get_current_span(W3C.extract(carrier))
    context = span.get_span_context()
    return [Link(context)] if context.is_valid else []


def traceparent(span: Span) -> str:
    """The span's context as one W3C traceparent string; "" for a span not recorded."""
    carrier: dict[str, str] = {}
    W3C.inject(carrier, trace.set_span_in_context(span))
    return carrier.get("traceparent", "")


def trace_id(parent: str) -> str:
    """The trace id inside a traceparent string."""
    parts = parent.split("-")
    return parts[1] if len(parts) == 4 else ""


def current_trace_id() -> str:
    """The trace the current span belongs to, in hex; "" when nothing is traced."""
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else ""


@contextmanager
def seam_span(seam: str, call: str, skill: str, fields: Sequence[str] = (),
              after: str = "") -> Iterator[Span]:
    """The caller's span for one crossing. `after` is the traceparent of the send
    that opened the task: a later read links to it, in whatever trace it runs."""
    linked = links({"traceparent": after}) if after else []
    with TRACER.start_as_current_span(
            f"{seam} {call}", kind=SpanKind.CLIENT, links=linked,
            attributes={"seam.name": seam, "seam.call": call, "seam.skill": skill,
                        "seam.fields": list(fields)}) as span:
        yield span


@contextmanager
def served(skill: str, metadata: Mapping[str, Any] | Struct) -> Iterator[Span]:
    """The partner's span for one request: a new trace, linked to the caller's."""
    with TRACER.start_as_current_span(
            f"serve {skill}", context=Context(), kind=SpanKind.SERVER,
            links=links(metadata), attributes={"seam.skill": skill}) as span:
        yield span


__all__ = ["CARRIED", "TRACER", "current_trace_id", "inject", "links", "seam_span",
           "served", "trace_id", "traceparent"]
