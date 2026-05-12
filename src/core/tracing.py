"""OpenTelemetry tracing setup - exports spans to LangSmith via OTLP.

Each agent gets its own span, all nested under a single parent 'sql_pipeline' span
so the full request flow is visible in one trace.

Thread safety note:
  OTEL context is thread-local by default. Spans created inside ThreadPoolExecutor
  workers won't automatically nest under the parent span unless the parent context
  is explicitly propagated. We carry it via _pipeline_contexts[session_id] and
  attach/detach it in each worker via run_in_context().

Span attributes use a consistent layout for LangSmith / OTLP viewers:
  app.input.primary / app.input.sub — user question and supporting inputs
  app.output.primary / app.output.sub — main model output and structured extras
  app.prompt.name / app.prompt.version / app.prompt.body_excerpt — DB-backed prompt
  gen_ai.request.model — chat or embedding model id from Settings
  latency_ms — wall time for the node

LangSmith maps OpenTelemetry attributes to run I/O (see LangSmith OTEL docs):
  input.value / output.value → Inputs / Outputs (use JSON strings for rich payloads)
  gen_ai.prompt.{n}.content / gen_ai.completion.{n}.content → message-style I/O
  langsmith.metadata.* → Metadata panel
"""

import json
import os
import time
from functools import wraps

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False

# Store parent trace context per session_id (thread-safe: written once before threads spawn)
_pipeline_contexts: dict = {}


def _trunc(value: object, max_len: int) -> str:
    if value is None:
        return ""
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _span_json(data: dict, max_len: int = 7500) -> str:
    """Serialize dict for OTEL span attribute (LangSmith maps input.value / output.value)."""
    return _trunc(json.dumps(data, default=str, ensure_ascii=False), max_len)


def _load_prompt_meta(prompt_key: str | None) -> tuple[int | None, str | None]:
    if not prompt_key:
        return None, None
    try:
        from src.core.prompts import get_prompt_with_version

        tmpl, ver = get_prompt_with_version(prompt_key)
        return int(ver), _trunc(tmpl, 1200)
    except Exception:
        return None, None


def _langsmith_input_dict(
    state: dict,
    node_name: str,
    prompt_key: str | None,
    prompt_version: int | None,
    prompt_excerpt: str | None,
) -> dict:
    sch = (state.get("schema_context") or "").strip()
    d: dict = {
        "node": node_name,
        "original_query": state.get("original_query") or "",
        "rewritten_query": (state.get("rewritten_query") or None) or None,
        "cache_hit": bool(state.get("cache_hit")),
        "l1_checked": bool(state.get("l1_checked")),
        "l2_hit": bool(state.get("l2_hit")),
        "intent": state.get("intent") or None,
        "query_complexity": state.get("query_complexity"),
        "retry_count": state.get("retry_count", 0),
        "has_schema_context": bool(sch),
        "schema_context_chars": len(sch) if sch else 0,
    }
    if prompt_key:
        d["prompt_name"] = prompt_key
        d["prompt_version"] = prompt_version
        if prompt_excerpt:
            d["prompt_excerpt"] = prompt_excerpt
    if state.get("validation_errors"):
        d["validation_errors"] = state.get("validation_errors")
    if sch and node_name in (
        "sql_generator",
        "schema_retriever",
        "sql_validator",
        "sql_executor",
        "parallel_init",
    ):
        d["schema_excerpt"] = _trunc(sch, 2000)
    return d


def _langsmith_output_dict(result: dict, node_name: str) -> dict:
    """Per-node summary for LangSmith output.value JSON."""
    d: dict = {
        "node": node_name,
        "cache_hit": bool(result.get("cache_hit")),
        "l1_checked": bool(result.get("l1_checked")),
        "l2_hit": bool(result.get("l2_hit")),
        "status": result.get("status"),
    }
    if result.get("intent"):
        d["intent"] = result.get("intent")
    if result.get("query_complexity"):
        d["query_complexity"] = result.get("query_complexity")
    if result.get("generated_sql"):
        d["generated_sql"] = result.get("generated_sql")
    if result.get("sql_confidence") is not None:
        d["sql_confidence"] = result.get("sql_confidence")
    if result.get("tables_used"):
        d["tables_used"] = result.get("tables_used")
    if result.get("validation_errors"):
        d["validation_errors"] = result.get("validation_errors")
    if result.get("sql_validated") is not None:
        d["sql_validated"] = result.get("sql_validated")
    if result.get("estimated_cost"):
        d["estimated_cost"] = result.get("estimated_cost")
    if result.get("results") is not None:
        d["result_row_count"] = len(result["results"]) if isinstance(result.get("results"), list) else None
    if result.get("explanation"):
        d["explanation"] = _trunc(result.get("explanation"), 1500)
    if result.get("error"):
        d["error"] = _trunc(result.get("error"), 1500)
    if result.get("is_ambiguous") is not None:
        d["is_ambiguous"] = result.get("is_ambiguous")
        d["ambiguity_score"] = result.get("ambiguity_score")
    if result.get("rewrite_confidence") is not None:
        d["rewrite_confidence"] = result.get("rewrite_confidence")
    if result.get("approved") is not None:
        d["approved"] = result.get("approved")
    if result.get("approval_explanation"):
        d["approval_explanation"] = _trunc(result.get("approval_explanation"), 800)
    if result.get("conversation_summary"):
        d["conversation_summary_excerpt"] = _trunc(result.get("conversation_summary"), 800)
    if result.get("history_token_usage"):
        d["history_token_usage"] = result.get("history_token_usage")
    if result.get("embedding_done"):
        d["embedding_done"] = result.get("embedding_done")
    emb = result.get("query_embedding")
    emb_len = len(emb) if isinstance(emb, list) else 0
    if emb_len:
        d["embedding_dims"] = emb_len
    if result.get("schema_context"):
        d["schema_context_chars"] = len(result["schema_context"])
    # Human-readable one-liner for UIs that prefer plain text
    summary = (
        result.get("explanation")
        or result.get("generated_sql")
        or result.get("rewritten_query")
        or ""
    )
    if not (summary and str(summary).strip()):
        if node_name == "cache_l1":
            summary = (
                f"L1 exact cache: hit={result.get('cache_hit')} checked={result.get('l1_checked')}"
            )
        elif node_name == "cache_l2":
            summary = (
                f"L2 semantic cache: l2_hit={result.get('l2_hit')} pipeline_cache_hit={result.get('cache_hit')}"
            )
        elif node_name == "embedding_agent":
            summary = f"Embedding ready={result.get('embedding_done')} dims={emb_len}"
        elif node_name == "schema_retriever":
            summary = f"Schema retrieved tables={result.get('tables_used')} chars={len(result.get('schema_context') or '')}"
        elif node_name == "sql_validator":
            ve = result.get("validation_errors") or []
            summary = f"Validated errors={len(ve)} cost={result.get('estimated_cost')}"
        elif node_name == "sql_executor":
            rc = len(result["results"]) if isinstance(result.get("results"), list) else 0
            summary = f"Executed rows={rc}"
        elif node_name == "approval_agent":
            summary = f"Approved={result.get('approved')}"
        elif node_name == "parallel_init":
            summary = f"intent={result.get('intent')} complexity={result.get('query_complexity')}"
        elif node_name == "intent_detector":
            summary = f"intent={result.get('intent')}"
        elif node_name == "memory_agent":
            summary = f"memory tokens={result.get('history_token_usage', 0)}"
        else:
            summary = f"{node_name} completed"
    d["summary"] = _trunc(summary, 2000)
    return d


def _genai_user_content(state: dict) -> str:
    parts = [state.get("original_query") or ""]
    rw = (state.get("rewritten_query") or "").strip()
    if rw:
        parts.append(f"(rewritten: {rw})")
    return _trunc("\n".join(parts), 4000)


def _genai_assistant_content(result: dict, node_name: str) -> str | None:
    out = _langsmith_output_dict(result, node_name).get("summary")
    if not out or not str(out).strip():
        return None
    return _trunc(out, 4000)


def setup_otel():
    """Initialize OTEL tracing with LangSmith OTLP endpoint (URLs from Settings / env)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    from src.core import get_settings

    s = get_settings()
    endpoint = (s.otel_exporter_otlp_endpoint or "").strip().rstrip("/")
    api_key = (
        s.langsmith_api_key
        or os.environ.get("LANGCHAIN_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY", "")
    ).strip()
    if not api_key:
        return  # No API key, skip OTEL setup

    if not endpoint:
        return

    resource = Resource.create(
        {
            "service.name": s.service_name,
            "project.name": s.langsmith_project,
        }
    )

    provider = TracerProvider(resource=resource)

    headers = {
        "x-api-key": api_key,
        "Langsmith-Project": s.langsmith_project,
    }

    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)


def get_tracer(name: str | None = None):
    """Get an OTEL tracer (instrumentation scope name from SERVICE_NAME when omitted)."""
    if name is None:
        from src.core import get_settings

        raw = (get_settings().service_name or "tracer").replace("-", "_")
        name = "".join(c if c.isalnum() or c in "_." else "_" for c in raw)[:64] or "tracer"
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


_LLM_NODES = frozenset(
    {
        "sql_generator",
        "intent_detector",
        "ambiguity_agent",
        "memory_agent",
        "response_synthesizer",
        "react_agent",
    }
)


def trace_agent_node(node_name: str, prompt_key: str | None = None):
    """Decorator that wraps an agent node function with OTEL span tracing.

    Creates a child span under the pipeline root span.
    Captures: structured input/output, optional DB prompt id/version/body excerpt,
    model id, latency, errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(state: dict) -> dict:
            tracer = get_tracer()
            parent_ctx = _get_parent_context(state)
            start_time = time.time()
            from src.core import get_settings

            settings = get_settings()

            with tracer.start_as_current_span(f"agent.{node_name}", context=parent_ctx) as span:
                span.set_attribute("langsmith.span.kind", "chain")
                span.set_attribute("agent.name", node_name)

                prompt_ver, prompt_excerpt = _load_prompt_meta(prompt_key)

                # ── Structured I/O (LangSmith-friendly) ──────────────────────────
                span.set_attribute("app.input.primary", _trunc(state.get("original_query", ""), 2000))
                sub_in_parts: list[str] = []
                rw = (state.get("rewritten_query") or "").strip()
                if rw:
                    sub_in_parts.append(f"rewritten={_trunc(rw, 900)}")
                sch = (state.get("schema_context") or "").strip()
                if sch and node_name in (
                    "sql_generator",
                    "schema_retriever",
                    "sql_validator",
                    "sql_executor",
                    "parallel_init",
                ):
                    sub_in_parts.append(f"schema_excerpt={_trunc(sch, 1400)}")
                if state.get("validation_errors"):
                    sub_in_parts.append(f"validation_errors={_trunc(state.get('validation_errors'), 800)}")
                span.set_attribute("app.input.sub", "|".join(sub_in_parts)[:2500])

                # ── Model (GenAI semantic convention) ───────────────────────────
                span.set_attribute("gen_ai.system", "openai")
                if node_name == "embedding_agent":
                    span.set_attribute("gen_ai.request.model", settings.openai_embedding_model)
                    span.set_attribute("gen_ai.operation.name", "embedding")
                elif prompt_key or node_name in _LLM_NODES:
                    span.set_attribute("gen_ai.request.model", settings.openai_chat_model)
                    span.set_attribute("gen_ai.operation.name", "chat")
                else:
                    span.set_attribute("gen_ai.operation.name", "chain")

                # ── DB prompt metadata (when this node uses a named template) ─────
                if prompt_key:
                    span.set_attribute("app.prompt.name", prompt_key)
                    span.set_attribute("langsmith.metadata.prompt_name", prompt_key)
                    if prompt_ver is not None:
                        span.set_attribute("app.prompt.version", prompt_ver)
                        span.set_attribute("langsmith.metadata.prompt_version", str(prompt_ver))
                    else:
                        span.set_attribute("app.prompt.version", -1)
                        span.set_attribute("langsmith.metadata.prompt_version", "unknown")
                    if prompt_excerpt:
                        span.set_attribute("app.prompt.body_excerpt", prompt_excerpt)

                # LangSmith Inputs: OpenInference input.value + GenAI user message
                span.set_attribute(
                    "input.value",
                    _span_json(_langsmith_input_dict(state, node_name, prompt_key, prompt_ver, prompt_excerpt)),
                )
                span.set_attribute("gen_ai.prompt.0.role", "user")
                span.set_attribute("gen_ai.prompt.0.content", _genai_user_content(state))

                span.set_attribute("input.rewritten_query", _trunc(state.get("rewritten_query", ""), 500))
                span.set_attribute("input.has_schema", bool(state.get("schema_context")))
                span.set_attribute("input.has_sql", bool(state.get("generated_sql")))
                span.set_attribute("input.has_embedding", bool(state.get("query_embedding")))
                span.set_attribute("input.retry_count", state.get("retry_count", 0))
                span.set_attribute("input.status", state.get("status", "processing"))
                span.set_attribute("input.cache_hit", bool(state.get("cache_hit")))
                span.set_attribute("input.l1_checked", bool(state.get("l1_checked")))
                span.set_attribute("input.l2_hit", bool(state.get("l2_hit")))

                try:
                    result = func(state)
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("latency_ms", latency_ms)
                    span.set_attribute("timing.duration_ms", round(latency_ms, 2))

                    if isinstance(result, dict):
                        out_payload = _langsmith_output_dict(result, node_name)
                        output_summary = out_payload.get("summary") or ""
                        span.set_attribute("app.output.primary", _trunc(output_summary, 2000))

                        sub_out: list[str] = []
                        if result.get("generated_sql"):
                            sub_out.append(f"sql={_trunc(result['generated_sql'], 700)}")
                        if result.get("tables_used"):
                            sub_out.append(f"tables={result.get('tables_used')}")
                        if result.get("sql_confidence") is not None:
                            sub_out.append(f"sql_confidence={result.get('sql_confidence')}")
                        if result.get("validation_errors"):
                            sub_out.append(f"validation_errors={_trunc(result.get('validation_errors'), 500)}")
                        if result.get("results"):
                            sub_out.append(f"result_count={len(result['results'])}")
                        if result.get("intent"):
                            sub_out.append(f"intent={result.get('intent')}")
                        sub_out.append(f"cache_hit={result.get('cache_hit')}")
                        sub_out.append(f"l1_checked={result.get('l1_checked')}")
                        sub_out.append(f"l2_hit={result.get('l2_hit')}")
                        span.set_attribute("app.output.sub", "|".join(sub_out)[:2500])

                        # LangSmith Outputs: OpenInference output.value + GenAI assistant message
                        span.set_attribute("output.value", _span_json(out_payload))
                        span.set_attribute("output.status", result.get("status", ""))
                        assist = _genai_assistant_content(result, node_name)
                        if assist:
                            span.set_attribute("gen_ai.completion.0.role", "assistant")
                            span.set_attribute("gen_ai.completion.0.content", assist)

                        if result.get("generated_sql"):
                            span.set_attribute("output.sql", result["generated_sql"][:1000])
                        if result.get("sql_confidence"):
                            span.set_attribute("output.confidence", result["sql_confidence"])
                        if result.get("validation_errors"):
                            span.set_attribute(
                                "output.validation_errors", str(result["validation_errors"])[:500]
                            )
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
                    else:
                        span.set_attribute(
                            "output.value",
                            _span_json({"node": node_name, "summary": str(result), "raw_type": type(result).__name__}),
                        )

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
                        if node_name == "ambiguity_agent" and result.get("is_ambiguous") is not None:
                            decision_entry["reason"] = (
                                f"ambiguous={result['is_ambiguous']}, score={result.get('ambiguity_score', 'N/A')}"
                            )
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
                    span.set_attribute("timing.duration_ms", round(latency_ms, 2))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def span_io_payload(data: dict, max_len: int = 7500) -> str:
    """Public helper: JSON for LangSmith ``input.value`` / ``output.value`` (e.g. root ``sql_pipeline`` span)."""
    return _span_json(data, max_len)


def run_in_context(fn, state: dict):
    """Execute fn(state) with the pipeline's parent OTEL context attached.

    Use this when submitting agent node functions to a ThreadPoolExecutor so that
    the child spans they create appear nested under the root 'sql_pipeline' span
    in LangSmith rather than floating as orphaned root spans.

    Usage:
        future = executor.submit(run_in_context, some_agent_node, state)
    """
    parent_ctx = _get_parent_context(state)
    if parent_ctx is not None:
        token = otel_context.attach(parent_ctx)
        try:
            return fn(state)
        finally:
            otel_context.detach(token)
    return fn(state)


def trace_supervisor(func):
    """Legacy decorator kept for backward compatibility. Now just passes through."""

    @wraps(func)
    def wrapper(state: dict):
        return func(state)

    return wrapper
