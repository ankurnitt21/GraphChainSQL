"""FastAPI routes with streaming support."""

import json
import time
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from src.api.models import QueryRequest, QueryResponse, ClarifyRequest, ApproveRequest, ActionApproveRequest, ClarifyOption, FeedbackRequest
from src.agents.pipeline import build_graph
from src.services.cache import semantic_cache_get
from src.core.database import save_conversation
from src.core.tracing import get_tracer, set_pipeline_context, clear_pipeline_context
from opentelemetry import trace
import structlog

log = structlog.get_logger()
router = APIRouter()

# Compiled graph singleton
_graph = None

# Store pending approval spans so /api/approve can nest under the same root
_pending_approval_spans: dict = {}


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _build_initial_state(query: str, session_id: str, require_approval: bool = False) -> dict:
    """Build initial state for the graph."""
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "original_query": query,
        # Memory & History (None signals "not yet loaded")
        "conversation_history": None,
        "conversation_summary": "",
        "history_token_usage": 0,
        # Ambiguity
        "rewritten_query": "",
        "is_ambiguous": None,
        "clarification_message": "",
        "clarification_options": [],
        # Dual-layer Cache
        "cache_hit": False,
        "l1_checked": False,
        "l2_hit": False,
        "cached_response": {},
        # Embedding
        "query_embedding": [],
        "embedding_done": False,
        # Schema
        "schema_context": "",
        "tables_used": [],
        "schema_relationships": [],
        # SQL
        "generated_sql": "",
        "sql_confidence": 0.0,
        "validation_errors": [],
        "retry_count": 0,
        "sql_validated": False,
        # Results
        "results": [],
        "explanation": "",
        # Control
        "status": "processing",
        "error": "",
        "next_agent": "",
        # HITL (SQL approval)
        "require_approval": require_approval,
        "approved": None,
        # Intent
        "intent": "",
        # ReAct
        "react_steps": [],
        "react_result": "",
        "pending_tool_call": None,
    }


def _state_to_response(state: dict, session_id: str, start_ms: float, run_id: str | None = None) -> QueryResponse:
    """Convert graph state to API response."""
    status = state.get("status", "completed")
    clarify_opts = None
    if state.get("clarification_options"):
        clarify_opts = [
            ClarifyOption(**o) if isinstance(o, dict) else o
            for o in state["clarification_options"]
        ]

    return QueryResponse(
        session_id=session_id,
        status=status,
        original_query=state.get("original_query"),
        rewritten_query=state.get("rewritten_query") or None,
        generated_sql=state.get("generated_sql") or None,
        confidence=state.get("sql_confidence") or None,
        results=state.get("results") or None,
        explanation=state.get("explanation") or None,
        tables_used=state.get("tables_used") or None,
        cache_hit=state.get("cache_hit", False),
        error=state.get("error") or None,
        clarification_message=state.get("clarification_message") or None,
        clarification_options=clarify_opts,
        duration_ms=int((time.time() * 1000) - start_ms),
        structured_output=state.get("structured_output") or None,
        decision_trace=state.get("decision_trace") or None,
        estimated_cost=state.get("estimated_cost") or None,
        approval_explanation=state.get("approval_explanation") or None,
        intent=state.get("intent") or None,
        react_steps=state.get("react_steps") or None,
        react_result=state.get("react_result") or None,
        pending_tool_call=state.get("pending_tool_call") or None,
        run_id=run_id,
    )


@router.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Execute a natural language query against the warehouse database."""
    start_ms = time.time() * 1000
    session_id = req.session_id or str(uuid.uuid4())
    tracer = get_tracer()

    root_span = tracer.start_span("sql_pipeline")
    root_span.set_attribute("langsmith.span.kind", "chain")
    root_span.set_attribute("input.value", req.query)
    root_span.set_attribute("session_id", session_id)
    root_ctx = trace.set_span_in_context(root_span)
    # Extract run_id (trace ID) for feedback linking in LangSmith
    run_id = format(root_span.get_span_context().trace_id, '032x')

    try:
        # Run the multi-agent graph
        graph = get_graph()
        initial_state = _build_initial_state(req.query, session_id, req.require_approval)
        # Register trace context for child span nesting
        set_pipeline_context(session_id, root_ctx)

        config = {"configurable": {"thread_id": session_id}}

        try:
            result = graph.invoke(initial_state, config=config)

            # Check if graph was interrupted (awaiting approval)
            graph_state = graph.get_state(config)
            if graph_state.next:
                values = graph_state.values
                intent = values.get("intent", "read")
                if intent == "action":
                    # ReAct tool approval interrupt
                    pending = _extract_pending_tool(graph_state)
                    return QueryResponse(
                        session_id=session_id,
                        status="awaiting_tool_approval",
                        original_query=req.query,
                        intent="action",
                        react_steps=values.get("react_steps") or [],
                        pending_tool_call=pending,
                        duration_ms=int((time.time() * 1000) - start_ms),
                        run_id=run_id,
                    )
                else:
                    # SQL approval interrupt - keep root span OPEN for /api/approve to continue
                    _pending_approval_spans[session_id] = root_span
                    return QueryResponse(
                        session_id=session_id,
                        status="awaiting_approval",
                        original_query=req.query,
                        rewritten_query=values.get("rewritten_query"),
                        generated_sql=values.get("generated_sql"),
                        confidence=values.get("sql_confidence"),
                        tables_used=values.get("tables_used"),
                        duration_ms=int((time.time() * 1000) - start_ms),
                        run_id=run_id,
                    )

            root_span.set_attribute("output.value", (result.get("explanation") or result.get("error") or "")[:500])
            root_span.set_attribute("cache_hit", result.get("cache_hit", False))
            root_span.set_status(trace.Status(trace.StatusCode.OK))
            root_span.end()
            return _state_to_response(result, session_id, start_ms, run_id=run_id)
        except Exception as e:
            log.error("query_pipeline_error", error=str(e))
            root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
            root_span.record_exception(e)
            root_span.end()
            return QueryResponse(
                session_id=session_id,
                status="failed",
                original_query=req.query,
                error=str(e),
                duration_ms=int((time.time() * 1000) - start_ms),
            )
        finally:
            # Only clear context if not awaiting approval
            if session_id not in _pending_approval_spans:
                clear_pipeline_context(session_id)
    except Exception as e:
        root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
        root_span.end()
        raise


@router.post("/api/query/stream")
def query_stream(req: QueryRequest):
    """Stream the multi-agent pipeline execution steps."""
    session_id = req.session_id or str(uuid.uuid4())

    def event_generator():
        graph = get_graph()
        initial_state = _build_initial_state(req.query, session_id, req.require_approval)
        config = {"configurable": {"thread_id": session_id}}

        # Create root pipeline span and register context for child nesting
        tracer = get_tracer()
        root_span = tracer.start_span("sql_pipeline")
        root_span.set_attribute("langsmith.span.kind", "chain")
        root_span.set_attribute("input.value", req.query)
        root_span.set_attribute("session_id", session_id)
        root_ctx = trace.set_span_in_context(root_span)
        set_pipeline_context(session_id, root_ctx)

        try:
            # First event: announce session_id so client can set it before any interrupt
            yield f"data: {json.dumps({'node': 'session_init', 'session_id': session_id})}\n\n"

            # Stream node-by-node execution
            interrupted = False
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():

                    # ── LangGraph interrupt signal ────────────────────────
                    if node_name == "__interrupt__":
                        # node_output is a tuple of Interrupt objects
                        graph_state = graph.get_state(config)
                        values = graph_state.values
                        intent = values.get("intent", "read")
                        if intent == "action":
                            pending = _extract_pending_tool(graph_state)
                            payload = json.dumps({
                                "node": "tool_approval_required",
                                "status": "awaiting_tool_approval",
                                "session_id": session_id,
                                "react_steps": values.get("react_steps", []),
                                "original_query": values.get("original_query", ""),
                                "pending_tool_call": pending,
                            })
                        else:
                            payload = json.dumps({
                                "node": "approval_required",
                                "status": "awaiting_approval",
                                "session_id": session_id,
                                "sql": values.get("generated_sql", ""),
                                "confidence": values.get("sql_confidence", 0),
                                "tables_used": values.get("tables_used", []),
                            })
                        yield f"data: {payload}\n\n"
                        interrupted = True
                        break  # stop streaming — graph is paused

                    if not isinstance(node_output, dict):
                        node_output = {}

                    step_data = {
                        "node": node_name,
                        "status": node_output.get("status", "processing"),
                    }
                    if node_output.get("intent"):
                        step_data["intent"] = node_output["intent"]
                    if node_output.get("rewritten_query"):
                        step_data["rewritten_query"] = node_output["rewritten_query"]
                    if node_output.get("generated_sql"):
                        step_data["sql"] = node_output["generated_sql"]
                    if node_output.get("results"):
                        step_data["result_count"] = len(node_output["results"])
                    if node_output.get("explanation"):
                        step_data["explanation"] = node_output["explanation"]
                    if node_output.get("error"):
                        step_data["error"] = node_output["error"]
                    if node_output.get("cache_hit"):
                        step_data["cache_hit"] = True
                    if node_output.get("is_ambiguous"):
                        step_data["is_ambiguous"] = True

                    yield f"data: {json.dumps(step_data)}\n\n"

                if interrupted:
                    break

            # Final result - check for interrupt (approval pending, backup path)
            if not interrupted:
                try:
                    graph_state = graph.get_state(config)
                    if graph_state.next:
                        values = graph_state.values
                        intent = values.get('intent', 'read')
                        if intent == 'action':
                            pending = _extract_pending_tool(graph_state)
                            react_payload = json.dumps({
                                'node': 'tool_approval_required',
                                'status': 'awaiting_tool_approval',
                                'session_id': session_id,
                                'react_steps': values.get('react_steps', []),
                                'original_query': values.get('original_query', ''),
                                'pending_tool_call': pending,
                            })
                            yield f"data: {react_payload}\n\n"
                        else:
                            sql_payload = json.dumps({
                                'node': 'approval_required',
                                'status': 'awaiting_approval',
                                'session_id': session_id,
                                'sql': values.get('generated_sql', ''),
                                'confidence': values.get('sql_confidence', 0),
                                'tables_used': values.get('tables_used', []),
                            })
                            yield f"data: {sql_payload}\n\n"
                    else:
                        final_state = graph_state.values
                        response = _state_to_response(final_state, session_id, time.time()*1000)
                        root_span.set_attribute("output.value", (final_state.get("explanation") or final_state.get("error") or "")[:500])
                        yield f"data: {json.dumps({'node': 'complete', 'status': 'done', 'result': response.model_dump()})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'node': 'complete', 'status': 'done', 'error': str(e)})}\n\n"

            root_span.set_status(trace.Status(trace.StatusCode.OK))

        except Exception as e:
            root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
            root_span.record_exception(e)
            yield f"data: {json.dumps({'node': 'error', 'error': str(e)})}\n\n"
        finally:
            root_span.end()
            clear_pipeline_context(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/api/clarify", response_model=QueryResponse)
def clarify(req: ClarifyRequest):
    """Handle clarification response."""
    start_ms = time.time() * 1000

    resolved_query = req.custom_query or ""
    if not resolved_query and req.selected_index is not None:
        # Get from graph state via checkpointer
        graph = get_graph()
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            state = graph.get_state(config).values
            options = state.get("clarification_options", [])
            for opt in options:
                if opt.get("index") == req.selected_index:
                    resolved_query = opt["query"]
                    break
        except Exception:
            pass

    if not resolved_query:
        raise HTTPException(status_code=400, detail="No query resolved from clarification")

    save_conversation(req.session_id, "USER", f"Clarified: {resolved_query}")

    # Run graph with resolved query
    graph = get_graph()
    initial_state = _build_initial_state(resolved_query, req.session_id, False)
    config = {"configurable": {"thread_id": f"{req.session_id}-clarify"}}

    try:
        result = graph.invoke(initial_state, config=config)
        return _state_to_response(result, req.session_id, start_ms)
    except Exception as e:
        log.error("clarify_error", error=str(e))
        return QueryResponse(
            session_id=req.session_id,
            status="failed",
            error=str(e),
            duration_ms=int((time.time() * 1000) - start_ms),
        )


@router.post("/api/approve", response_model=QueryResponse)
def approve(req: ApproveRequest):
    """Handle human approval - resumes the interrupted graph via Command(resume=...).

    Nests the resumed execution under the original sql_pipeline span.
    """
    start_ms = time.time() * 1000

    # Retrieve the parent root span from the original /api/query call
    root_span = _pending_approval_spans.pop(req.session_id, None)

    if not req.approved:
        graph = get_graph()
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            graph.invoke(Command(resume={"approved": False}), config=config)
        except Exception:
            pass
        if root_span:
            root_span.set_attribute("output.value", "rejected_by_user")
            root_span.set_status(trace.Status(trace.StatusCode.OK))
            root_span.end()
        clear_pipeline_context(req.session_id)
        return QueryResponse(
            session_id=req.session_id,
            status="failed",
            error="Query rejected by user",
            duration_ms=int((time.time() * 1000) - start_ms),
        )

    # Re-register the pipeline context so child spans nest under root
    if root_span:
        root_ctx = trace.set_span_in_context(root_span)
        set_pipeline_context(req.session_id, root_ctx)

    # Resume the interrupted graph with approval
    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = graph.invoke(Command(resume={"approved": True}), config=config)
        if root_span:
            root_span.set_attribute("output.value", (result.get("explanation") or result.get("error") or "")[:500])
            root_span.set_status(trace.Status(trace.StatusCode.OK))
            root_span.end()
        return _state_to_response(result, req.session_id, start_ms)
    except Exception as e:
        log.error("approve_resume_error", error=str(e))
        if root_span:
            root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
            root_span.record_exception(e)
            root_span.end()
        return QueryResponse(
            session_id=req.session_id,
            status="failed",
            error=str(e),
            duration_ms=int((time.time() * 1000) - start_ms),
        )
    finally:
        clear_pipeline_context(req.session_id)


@router.get("/api/prompts")
async def list_prompts():
    """List all prompt templates."""
    from src.core.prompts import list_prompts
    return list_prompts()


@router.post("/api/action/approve", response_model=QueryResponse)
def action_approve(req: ActionApproveRequest):
    """Resume the ReAct agent after human approves or rejects a tool call.

    Called by the UI when the user clicks Approve/Reject on a pending tool call.
    Resumes the interrupted graph via Command(resume={approved, feedback}).
    """
    start_ms = time.time() * 1000
    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = graph.invoke(
            Command(resume={"approved": req.approved, "feedback": req.feedback or ""}),
            config=config,
        )

        # Graph may pause again (next tool) or complete
        graph_state = graph.get_state(config)
        if graph_state.next:
            # Still interrupted - another tool needs approval
            values = graph_state.values
            return QueryResponse(
                session_id=req.session_id,
                status="awaiting_tool_approval",
                original_query=values.get("original_query"),
                intent="action",
                react_steps=values.get("react_steps") or [],
                pending_tool_call=_extract_pending_tool(graph_state),
                duration_ms=int((time.time() * 1000) - start_ms),
            )

        return _state_to_response(result, req.session_id, start_ms)
    except Exception as e:
        log.error("action_approve_error", error=str(e))
        return QueryResponse(
            session_id=req.session_id,
            status="failed",
            error=str(e),
            duration_ms=int((time.time() * 1000) - start_ms),
        )


def _extract_pending_tool(graph_state) -> dict | None:
    """Extract the pending interrupt payload from graph state tasks."""
    try:
        for task in graph_state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                interrupt_val = task.interrupts[0].value
                if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "tool_approval":
                    return interrupt_val
    except Exception:
        pass
    return None


@router.get("/api/action/tools")
def list_action_tools():
    """List available ReAct action tools and their signatures."""
    from src.agents.action_tools import TOOL_DESCRIPTIONS
    return {"tools": TOOL_DESCRIPTIONS}


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "GraphChainSQLPython", "version": "7.0.0"}


# ─── Feedback Endpoints ──────────────────────────────────────────────────────

@router.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit user feedback (thumbs up/down) on a query result.

    Stores in DB and syncs to LangSmith for trace-level visibility.
    """
    from src.services.feedback import save_feedback

    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 (thumbs up) or -1 (thumbs down)")

    result = save_feedback(
        session_id=req.session_id,
        query=req.query,
        rating=req.rating,
        generated_sql=req.generated_sql,
        comment=req.comment,
        correction=req.correction,
        run_id=req.run_id,
    )
    return {"status": "ok", **result}


@router.get("/api/feedback/stats")
def feedback_stats(session_id: str | None = None):
    """Get feedback statistics (overall or per session)."""
    from src.services.feedback import get_feedback_stats
    return get_feedback_stats(session_id)


@router.get("/api/feedback/negative")
def negative_feedback(limit: int = 50):
    """Get recent negative feedback for prompt improvement analysis."""
    from src.services.feedback import get_negative_feedback
    return get_negative_feedback(limit)

