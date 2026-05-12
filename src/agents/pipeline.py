"""Pipeline - Phase-based DAG builder for the multi-agent SQL pipeline.

Architecture: Direct-edge graph with conditional routing (no hub-and-spoke).
Uses LangGraph conditional edges — no central supervisor LLM.

Pipeline Phases:
  Phase 1 (Parallel): Intent + Memory + Cache L1 + Embedding (4-way concurrent)
                       Keyword-based complexity estimate runs here too (0 ms, no LLM).
  Phase 2 (Conditional): Ambiguity Agent — always runs for read queries.
                          Returns quickly ("not ambiguous") for clear queries.
  Phase 3 (Conditional): Cache L2
  Phase 4 (Sequential): Schema Retrieval
  Phase 5 (Core SQL): SQL Generation → SQL Validation
  Phase 6 (HITL): Approval Agent (dev=auto-approve, prod=human gate)
  Phase 7 (Output): SQL Execution → Response Synthesizer

  Action Pipeline (intent="action" detected in Phase 1):
  Phase 1 → ReAct Agent Loop (LLM thinks → interrupt(approve) → execute → repeat)

Latency improvements:
  • Intent detection merged into Phase 1 parallel block → saves ~1-5 s serial LLM call
  • complexity_detector LLM node removed — replaced with a 0 ms keyword classifier
    inside parallel_init. query_complexity is still set for response_synthesizer.
  • OTEL context explicitly propagated into ThreadPoolExecutor workers so all
    child spans appear nested under the root 'sql_pipeline' span in LangSmith
"""

import concurrent.futures
from langgraph.graph import StateGraph, START, END
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver
from src.core import get_settings
from src.core.state import AgentState
from src.core.tracing import trace_agent_node, run_in_context
from src.agents.memory_agent import memory_agent_node
from src.agents.ambiguity_agent import ambiguity_agent_node
from src.agents.cache_agent import cache_l1_node, cache_l2_node
from src.agents.embedding_agent import embedding_agent_node
from src.agents.schema_agent import schema_retriever_node
from src.agents.sql_generator_agent import sql_generator_node
from src.agents.sql_validator_agent import sql_validator_node
from src.agents.approval_agent import approval_agent_node
from src.agents.executor_agent import sql_executor_node
from src.agents.response_agent import response_synthesizer_node
from src.agents.intent_detector import intent_detector_node
from src.agents.react_agent import react_agent_node
import structlog

# ─── Keyword-based complexity classifier (no LLM, ~0 ms) ─────────────────────

_COMPLEX_KWORDS = {
    "trend", "forecast", "percentile", "rank over", "year over year",
    "percent change", "pivot", "moving average", "cumulative", "rolling",
    "compare", "vs ", "versus",
}
_SIMPLE_KWORDS = {
    "top", "list", "show", "what is", "how many", "count", "sum", "total",
    "latest", "first", "last", "cheapest", "most expensive", "highest", "lowest",
}


def _fast_complexity(query: str) -> str:
    """Classify query complexity using keyword heuristics — zero LLM, zero latency.

    Used to guide response_synthesizer (template vs LLM) and PII guard level.
    Simple  → template response, regex PII only.
    Moderate → LLM response, regex PII only.
    Complex → LLM response, full LLM PII guard.
    """
    q = query.lower()
    if any(k in q for k in _COMPLEX_KWORDS):
        return "complex"
    if any(k in q for k in _SIMPLE_KWORDS):
        return "simple"
    return "moderate"

log = structlog.get_logger()
settings = get_settings()


# ─── ReAct Loop Router ───────────────────────────────────────────────────────

def after_react(state: AgentState) -> str:
    """Loop react_agent while status=processing; exit on terminal statuses."""
    status = state.get("status", "processing")
    if status in ("completed", "failed", "action_rejected"):
        return END
    return "react_agent"


# ─── Phase 1: 4-Way Parallel Init ────────────────────────────────────────────

@trace_agent_node("parallel_init")
def parallel_init_node(state: AgentState) -> dict:
    """Phase 1: Run Intent + Memory + Cache L1 + Embedding concurrently.

    Also runs keyword-based complexity classification (0 ms, no LLM) so that
    response_synthesizer can choose template vs LLM without any extra node.

    OTEL context is propagated into each worker via run_in_context() so that
    child spans appear properly nested under 'sql_pipeline' in LangSmith.
    """
    merged = {"messages": state.get("messages", [])}

    # Keyword complexity — 0 ms, available for response_synthesizer
    query = state.get("original_query", "")
    merged["query_complexity"] = _fast_complexity(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_intent = executor.submit(run_in_context, intent_detector_node, state)
        future_memory = executor.submit(run_in_context, memory_agent_node, state)
        future_cache  = executor.submit(run_in_context, cache_l1_node, state)
        future_embed  = executor.submit(run_in_context, embedding_agent_node, state)

        # ── Intent ──────────────────────────────────────────────────────────
        try:
            intent_result = future_intent.result(timeout=15)
            merged["intent"] = intent_result.get("intent", "read")
        except Exception as e:
            log.warning("parallel_intent_failed", error=str(e))
            merged["intent"] = "read"

        # ── Memory ──────────────────────────────────────────────────────────
        try:
            memory_result = future_memory.result(timeout=15)
            merged["conversation_history"] = memory_result.get("conversation_history", [])
            merged["conversation_summary"] = memory_result.get("conversation_summary", "")
            merged["history_token_usage"]  = memory_result.get("history_token_usage", 0)
        except Exception as e:
            log.warning("parallel_memory_failed", error=str(e))
            merged["conversation_history"] = []
            merged["conversation_summary"] = ""
            merged["history_token_usage"]  = 0

        # ── Cache L1 ─────────────────────────────────────────────────────────
        try:
            cache_result = future_cache.result(timeout=10)
            merged["cache_hit"]       = cache_result.get("cache_hit", False)
            merged["l1_checked"]      = True
            merged["cached_response"] = cache_result.get("cached_response", {})
            if cache_result.get("cache_hit"):
                merged["generated_sql"] = cache_result.get("generated_sql", "")
                merged["results"]       = cache_result.get("results", [])
                merged["explanation"]   = cache_result.get("explanation", "")
                merged["status"]        = "completed"
        except Exception as e:
            log.warning("parallel_cache_l1_failed", error=str(e))
            merged["cache_hit"]       = False
            merged["l1_checked"]      = True
            merged["cached_response"] = {}

        # ── Embedding ────────────────────────────────────────────────────────
        try:
            embed_result = future_embed.result(timeout=15)
            merged["query_embedding"] = embed_result.get("query_embedding", [])
            merged["embedding_done"]  = True
        except Exception as e:
            log.warning("parallel_embed_failed", error=str(e))
            merged["query_embedding"] = []
            merged["embedding_done"]  = True

    return merged


# ─── Conditional Edge Routers ────────────────────────────────────────────────

def after_parallel_init(state: AgentState) -> str:
    """Route after Phase 1.

    - action intent  → react_agent
    - L1 cache hit   → respond_from_cache
    - failed         → END
    - keyword simple → cache_l2   (skip ambiguity — query is unambiguous by keyword)
    - else           → ambiguity_agent

    complexity_detector LLM node is gone. Routing uses the 0 ms keyword classifier
    set inside parallel_init_node, so no extra latency is added.
    """
    if state.get("intent") == "action":
        return "react_agent"
    if state.get("cache_hit") and state.get("cached_response"):
        return "respond_from_cache"
    if state.get("status") == "failed":
        return END
    # Keyword-classified simple queries skip the 10 s ambiguity LLM call
    if state.get("query_complexity") == "simple":
        return "cache_l2"
    return "ambiguity_agent"


def after_ambiguity(state: AgentState) -> str:
    """Route after ambiguity: awaiting_clarification/failed → END, else → cache L2."""
    status = state.get("status", "processing")
    if status in ("awaiting_clarification", "failed"):
        return END
    return "cache_l2"


def after_cache_l2(state: AgentState) -> str:
    """Route after Cache L2: If hit → respond, else → schema retrieval."""
    if state.get("l2_hit") and state.get("cached_response"):
        return "respond_from_cache"
    return "schema_retriever"


def after_sql_gen(state: AgentState) -> str:
    """Route after SQL generation: failed → fallback/END, else → validation."""
    if state.get("status") == "failed":
        # Intelligent fallback: if we have cache, try simplified query
        if state.get("cached_response"):
            return "respond_from_cache"
        return END
    if not state.get("generated_sql"):
        # Fallback: if schema was loaded but SQL gen failed, retry is possible
        if state.get("schema_context") and state.get("retry_count", 0) < 1:
            return "sql_generator"
        return END
    return "sql_validator"


def after_sql_validation(state: AgentState) -> str:
    """Route after validation: errors + retries → regen, else → approval."""
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)
    if errors:
        if retry_count < settings.max_retries:
            return "sql_generator"
        return "validation_failed"
    return "approval_agent"


def after_approval(state: AgentState) -> str:
    """Route after approval: approved → execute, else → END."""
    if state.get("approved") is False or state.get("status") == "failed":
        return END
    return "sql_executor"


def after_execution(state: AgentState) -> str:
    """Route after execution: error → fallback or END, else → response."""
    if state.get("status") == "failed" or state.get("error"):
        # Intelligent fallback: execution error with retries left → regen SQL
        retry_count = state.get("retry_count", 0)
        if retry_count < settings.max_retries:
            log.info("execution_fallback_retry", retry=retry_count)
            return "sql_generator"
        return END
    return "response_synthesizer"


# ─── Utility Nodes ───────────────────────────────────────────────────────────

@trace_agent_node("respond_from_cache")
def respond_from_cache_node(state: AgentState) -> dict:
    """Terminal node: Return cached response as final result."""
    cached = state.get("cached_response", {})
    return {
        "messages": state.get("messages", []),
        "generated_sql": cached.get("sql", ""),
        "results": cached.get("results", []),
        "explanation": cached.get("explanation", ""),
        "status": "completed",
        "cache_hit": True,
    }


@trace_agent_node("validation_failed")
def validation_failed_node(state: AgentState) -> dict:
    """Terminal node: Retries exhausted with validation errors."""
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)
    return {
        "messages": state.get("messages", []),
        "status": "failed",
        "error": f"SQL validation failed after {retry_count} retries: {'; '.join(errors)}",
    }


# ─── Build the DAG Graph ─────────────────────────────────────────────────────

def build_graph():
    """Build the phase-based DAG graph with parallelism and conditional routing.

    Phase 1  START → parallel_init  (intent + memory + cache-L1 + embedding, 4-way)
                     keyword complexity set here at 0 ms (no LLM node)
    Phase 1a intent=="action"       → react_agent loop
    Phase 1b L1 cache hit           → respond_from_cache → END
    Phase 2  → ambiguity_agent      (always; quick pass for unambiguous queries)
    Phase 3  → cache_l2
    Phase 3a L2 hit                 → respond_from_cache → END
    Phase 4  → schema_retriever     (DB)
    Phase 5  → sql_generator → sql_validator (retry loop)
    Phase 6  → approval_agent → sql_executor → response_synthesizer → END
    """
    graph = StateGraph(AgentState)

    # ── Add all nodes ──────────────────────────────────────────────────────
    graph.add_node("react_agent",          react_agent_node)
    graph.add_node("parallel_init",        parallel_init_node)
    graph.add_node("ambiguity_agent",      ambiguity_agent_node)
    graph.add_node("cache_l2",             cache_l2_node)
    graph.add_node("respond_from_cache",   respond_from_cache_node)
    graph.add_node("schema_retriever",     schema_retriever_node)
    graph.add_node("sql_generator",        sql_generator_node)
    graph.add_node("sql_validator",        sql_validator_node)
    graph.add_node("validation_failed",    validation_failed_node)
    graph.add_node("approval_agent",       approval_agent_node)
    graph.add_node("sql_executor",         sql_executor_node)
    graph.add_node("response_synthesizer", response_synthesizer_node)

    # ── Phase 1: entry → 4-way parallel (intent + keyword complexity inside) ─
    graph.add_edge(START, "parallel_init")

    # ── Phase 1 → route: action | L1 hit | keyword-simple | else → ambiguity
    graph.add_conditional_edges("parallel_init", after_parallel_init, {
        "react_agent":        "react_agent",
        "respond_from_cache": "respond_from_cache",
        "cache_l2":           "cache_l2",
        "ambiguity_agent":    "ambiguity_agent",
        END:                  END,
    })

    # ── ReAct loop ─────────────────────────────────────────────────────────
    graph.add_conditional_edges("react_agent", after_react, {
        "react_agent": "react_agent",
        END: END,
    })

    # ── Phase 2: Ambiguity → route ─────────────────────────────────────────
    graph.add_conditional_edges("ambiguity_agent", after_ambiguity, {
        "cache_l2": "cache_l2",
        END: END,
    })

    # ── Phase 3: Cache L2 → route ─────────────────────────────────────────
    graph.add_conditional_edges("cache_l2", after_cache_l2, {
        "respond_from_cache": "respond_from_cache",
        "schema_retriever":   "schema_retriever",
    })

    # ── Cache terminal ─────────────────────────────────────────────────────
    graph.add_edge("respond_from_cache", END)

    # ── Phase 4: Schema → SQL Gen ─────────────────────────────────────────
    graph.add_edge("schema_retriever", "sql_generator")

    # ── Phase 5: SQL Gen → Validation (retry loop) ────────────────────────
    graph.add_conditional_edges("sql_generator", after_sql_gen, {
        "sql_validator":      "sql_validator",
        "sql_generator":      "sql_generator",
        "respond_from_cache": "respond_from_cache",
        END:                  END,
    })
    graph.add_conditional_edges("sql_validator", after_sql_validation, {
        "sql_generator":     "sql_generator",
        "approval_agent":    "approval_agent",
        "validation_failed": "validation_failed",
    })
    graph.add_edge("validation_failed", END)

    # ── Phase 6: Approval → Execution → Response ──────────────────────────
    graph.add_conditional_edges("approval_agent", after_approval, {
        "sql_executor": "sql_executor",
        END: END,
    })
    graph.add_conditional_edges("sql_executor", after_execution, {
        "response_synthesizer": "response_synthesizer",
        "sql_generator":        "sql_generator",
        END:                    END,
    })
    graph.add_edge("response_synthesizer", END)

    # ── Compile with PostgresSaver for distributed checkpointing ──────────
    db_url = settings.database_url_sync
    conn = Connection.connect(db_url, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)

