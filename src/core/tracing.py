"""OpenTelemetry tracing setup - exports spans to LangSmith via OTLP.

Each agent gets its own span, all nested under a single parent 'sql_pipeline' span
so the full request flow is visible in one trace.
"""

import os
import time
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_initialized = False

# Store parent trace context per session_id (thread-safe for sync generators)
_pipeline_contexts: dict = {}


def setup_otel():
    """Initialize OTEL tracing with LangSmith OTLP endpoint."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.smith.langchain.com/otel")
    api_key = os.environ.get("LANGCHAIN_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
    service_name = os.environ.get("OTEL_SERVICE_NAME", "GraphChainSQL")

    if not api_key:
        return  # No API key, skip OTEL setup

    project = os.environ.get("LANGCHAIN_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GraphChainSQL"))

    resource = Resource.create({
        "service.name": service_name,
        "project.name": project,
    })

    provider = TracerProvider(resource=resource)

    headers = {
        "x-api-key": api_key,
        "Langsmith-Project": project,
    }

    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "graphchainsql"):
    """Get an OTEL tracer."""
    return trace.get_tracer(name)


def set_pipeline_context(session_id: str, ctx):
    """Store the root pipeline span context for a session."""
    _pipeline_contexts[session_id] = ctx


def clear_pipeline_context(session_id: str):
    """Remove pipeline context after request completes."""
    _pipeline_contexts.pop(session_id, None)


def _get_parent_context(state: dict):
    """Get parent context from the pipeline context store."""
    session_id = state.get("session_id", "")
    return _pipeline_contexts.get(session_id)


def trace_agent_node(node_name: str):
    """Decorator that wraps an agent node function with OTEL span tracing.

    Creates a child span under the pipeline root span.
    Captures: input state, output state, latency, errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: dict) -> dict:
            tracer = get_tracer()
            parent_ctx = _get_parent_context(state)
            start_time = time.time()

            with tracer.start_as_current_span(f"agent.{node_name}", context=parent_ctx) as span:
                span.set_attribute("langsmith.span.kind", "chain")
                span.set_attribute("agent.name", node_name)

                # Input attributes
                span.set_attribute("input.value", state.get("original_query", "")[:500])
                span.set_attribute("input.rewritten_query", state.get("rewritten_query", "")[:500])
                span.set_attribute("input.has_schema", bool(state.get("schema_context")))
                span.set_attribute("input.has_sql", bool(state.get("generated_sql")))
                span.set_attribute("input.has_embedding", bool(state.get("query_embedding")))
                span.set_attribute("input.retry_count", state.get("retry_count", 0))
                span.set_attribute("input.status", state.get("status", "processing"))
                span.set_attribute("input.cache_hit", state.get("cache_hit", False))

                try:
                    result = func(state)
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("latency_ms", latency_ms)

                    if isinstance(result, dict):
                        output_summary = (
                            result.get("explanation")
                            or result.get("generated_sql")
                            or result.get("rewritten_query")
                            or ""
                        )
                        span.set_attribute("output.value", str(output_summary)[:500])
                        span.set_attribute("output.status", result.get("status", ""))

                        if result.get("generated_sql"):
                            span.set_attribute("output.sql", result["generated_sql"][:1000])
                        if result.get("sql_confidence"):
                            span.set_attribute("output.confidence", result["sql_confidence"])
                        if result.get("validation_errors"):
                            span.set_attribute("output.validation_errors", str(result["validation_errors"])[:500])
                        if result.get("sql_validated") is not None:
                            span.set_attribute("output.sql_validated", result["sql_validated"])
                        if result.get("results"):
                            span.set_attribute("output.result_count", len(result["results"]))
                        if result.get("explanation"):
                            span.set_attribute("output.explanation", result["explanation"][:500])
                        if result.get("error"):
                            span.set_attribute("output.error", result["error"][:500])
                            span.set_status(trace.Status(trace.StatusCode.ERROR, result["error"][:200]))
                        if result.get("schema_context"):
                            span.set_attribute("output.schema_length", len(result["schema_context"]))
                        if result.get("tables_used"):
                            span.set_attribute("output.tables_used", str(result["tables_used"]))
                        if result.get("cache_hit"):
                            span.set_attribute("output.cache_hit", True)
                        if result.get("history_token_usage"):
                            span.set_attribute("output.token_usage", result["history_token_usage"])
                        if result.get("is_ambiguous") is not None:
                            span.set_attribute("output.is_ambiguous", result["is_ambiguous"])
                        if result.get("ambiguity_score") is not None:
                            span.set_attribute("output.ambiguity_score", result["ambiguity_score"])
                        if result.get("rewrite_confidence") is not None:
                            span.set_attribute("output.rewrite_confidence", result["rewrite_confidence"])
                        if result.get("estimated_cost"):
                            span.set_attribute("output.estimated_cost", result["estimated_cost"])
                        if result.get("approval_explanation"):
                            span.set_attribute("output.approval_explanation", result["approval_explanation"][:500])
                        if result.get("query_complexity"):
                            span.set_attribute("output.query_complexity", result["query_complexity"])

                    # Record decision trace entry for this node
                    decision_entry = {
                        "node": node_name,
                        "latency_ms": round(latency_ms, 1),
                    }
                    if isinstance(result, dict):
                        if result.get("error"):
                            decision_entry["outcome"] = "error"
                            decision_entry["reason"] = result["error"][:100]
                        elif result.get("status") == "awaiting_clarification":
                            decision_entry["outcome"] = "clarification_needed"
                        elif result.get("cache_hit"):
                            decision_entry["outcome"] = "cache_hit"
                        else:
                            decision_entry["outcome"] = "success"
                        # Node-specific decision reasons
                        if node_name == "ambiguity_resolver" and result.get("is_ambiguous") is not None:
                            decision_entry["reason"] = f"ambiguous={result['is_ambiguous']}, score={result.get('ambiguity_score', 'N/A')}"
                        elif node_name == "sql_generator" and result.get("generated_sql"):
                            decision_entry["reason"] = f"confidence={result.get('sql_confidence', 'N/A')}"
                        elif node_name == "sql_validator" and result.get("validation_errors"):
                            decision_entry["reason"] = f"errors={len(result['validation_errors'])}"
                        elif node_name == "approval_agent":
                            decision_entry["reason"] = f"approved={result.get('approved', 'N/A')}"

                    # Append to decision_trace in result (state update)
                    if isinstance(result, dict):
                        existing_trace = state.get("decision_trace", [])
                        result["decision_trace"] = existing_trace + [decision_entry]

                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("latency_ms", latency_ms)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def trace_supervisor(func):
    """Legacy decorator kept for backward compatibility. Now just passes through."""
    @wraps(func)
    def wrapper(state: dict):
        return func(state)
    return wrapper
