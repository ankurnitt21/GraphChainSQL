"""Pipeline - Phase-based DAG builder for the multi-agent SQL pipeline.

Architecture: Direct-edge graph with conditional routing (no hub-and-spoke).
Uses LangGraph conditional edges — no central supervisor LLM.

Pipeline Phases:
  Phase 0 (Entry): Intent Detection → "read" or "action"
  Phase 1 (Parallel): Memory + Cache L1 + Embedding (concurrent via ThreadPool)
  Phase 2 (Conditional): Ambiguity Agent (pre-check skips if clearly structured)
  Phase 3 (Conditional): Cache L2 (canonical/rewritten query lookup)
  Phase 4 (Core SQL): Schema Retrieval → SQL Generation → SQL Validation
  Phase 5 (HITL): Approval Agent (dev=auto-approve, prod=human gate)
  Phase 6 (Output): SQL Execution → Response Synthesizer

  Action Pipeline (intent="action"):
  Phase 0 → ReAct Agent Loop (LLM thinks → interrupt(approve) → execute → repeat)
"""

import concurrent.futures
from langgraph.graph import StateGraph, START, END
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver
from src.core import get_settings
from src.core.state import AgentState
from src.core.tracing import trace_agent_node
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

log = structlog.get_logger()
settings = get_settings()


# ─── Phase 0: Intent Router ─────────────────────────────────────────────────

def after_intent_detection(state: AgentState) -> str:
    """Route based on detected intent: read → SQL pipeline, action → ReAct agent."""
    intent = state.get("intent", "read")
    if intent == "action":
        return "react_agent"
    return "parallel_init"


# ─── ReAct Loop Router ───────────────────────────────────────────────────────

def after_react(state: AgentState) -> str:
    """Loop react_agent while status=processing; exit on terminal statuses."""
    status = state.get("status", "processing")
    if status in ("completed", "failed", "action_rejected"):
        return END
    return "react_agent"


# ─── Phase 1: Parallel Init ─────────────────────────────────────────────────

@trace_agent_node("parallel_init")
def parallel_init_node(state: AgentState) -> dict:
    """Phase 1: Run Memory, Cache L1, and Embedding concurrently.

    Uses ThreadPoolExecutor for I/O-bound parallel execution.
    Merges results into a single state update.
    """
    merged = {"messages": state.get("messages", [])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_memory = executor.submit(memory_agent_node, state)
        future_cache = executor.submit(cache_l1_node, state)
        future_embed = executor.submit(embedding_agent_node, state)

        # Collect results
        try:
            memory_result = future_memory.result(timeout=15)
            merged["conversation_history"] = memory_result.get("conversation_history", [])
            merged["conversation_summary"] = memory_result.get("conversation_summary", "")
            merged["history_token_usage"] = memory_result.get("history_token_usage", 0)
        except Exception as e:
            log.warning("parallel_memory_failed", error=str(e))
            merged["conversation_history"] = []
            merged["conversation_summary"] = ""
            merged["history_token_usage"] = 0

        try:
            cache_result = future_cache.result(timeout=10)
            merged["cache_hit"] = cache_result.get("cache_hit", False)
            merged["l1_checked"] = True
            merged["cached_response"] = cache_result.get("cached_response", {})
            if cache_result.get("cache_hit"):
                merged["generated_sql"] = cache_result.get("generated_sql", "")
                merged["results"] = cache_result.get("results", [])
                merged["explanation"] = cache_result.get("explanation", "")
                merged["status"] = "completed"
        except Exception as e:
            log.warning("parallel_cache_l1_failed", error=str(e))
            merged["cache_hit"] = False
            merged["l1_checked"] = True
            merged["cached_response"] = {}

        try:
            embed_result = future_embed.result(timeout=15)
            merged["query_embedding"] = embed_result.get("query_embedding", [])
            merged["embedding_done"] = True
        except Exception as e:
            log.warning("parallel_embed_failed", error=str(e))
            merged["query_embedding"] = []
            merged["embedding_done"] = True

    return merged


# ─── Conditional Edge Routers ────────────────────────────────────────────────

def after_parallel_init(state: AgentState) -> str:
    """Route after Phase 1: If L1 cache hit → respond, else → complexity detection."""
    if state.get("cache_hit") and state.get("cached_response"):
        return "respond_from_cache"
    if state.get("status") == "failed":
        return END
    return "complexity_detector"


def after_complexity(state: AgentState) -> str:
    """Route after complexity detection: simple → skip ambiguity, else → ambiguity."""
    complexity = state.get("query_complexity", "moderate")
    if complexity == "simple":
        # Simple queries skip ambiguity check entirely
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

@trace_agent_node("complexity_detector")
def complexity_detector_node(state: AgentState) -> dict:
    """Classify query complexity using LLM with dynamic DB prompt.

    Simple: Direct data lookups, single-table queries, counts
    Moderate: Multi-table joins, filtering with conditions
    Complex: Aggregations with grouping, subqueries, temporal analysis
    """
    import json as _json
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_groq import ChatGroq
    from src.core.prompts import get_prompt
    from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter

    query = (state.get("rewritten_query", "") or state.get("original_query", ""))
    messages = state.get("messages", [])

    try:
        system_content = get_prompt("complexity_detection")
    except Exception as e:
        log.error("prompt_load_failed", prompt="complexity_detection", error=str(e))
        # Default to moderate complexity if prompt unavailable
        return {
            "messages": messages,
            "query_complexity": "moderate",
        }

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_fast_model,
            temperature=0,
        )
        response = resilient_call(
            llm.invoke,
            [
                SystemMessage(content=system_content),
                HumanMessage(content=query),
            ],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )
        text = response.content.strip()
        data = _json.loads(text)
        complexity = data.get("complexity", "moderate")
        if complexity not in ("simple", "moderate", "complex"):
            complexity = "moderate"
    except Exception as e:
        log.warning("complexity_detection_failed", error=str(e), fallback="moderate")
        complexity = "moderate"

    log.info("complexity_detected", complexity=complexity, query=query[:50])
    return {
        "messages": messages,
        "query_complexity": complexity,
    }

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


# ─── Build the DAG Graph ────────────────────────────────────────────────────

def build_graph():
    """Build the phase-based DAG graph with conditional routing.

    Phase 0: intent_detector → "read" → parallel_init | "action" → react_agent
    Phase 1 runs memory/cache/embedding in parallel (read pipeline only).
    """
    graph = StateGraph(AgentState)

    # ── Add all nodes ──
    graph.add_node("intent_detector", intent_detector_node)
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("parallel_init", parallel_init_node)
    graph.add_node("complexity_detector", complexity_detector_node)
    graph.add_node("ambiguity_agent", ambiguity_agent_node)
    graph.add_node("cache_l2", cache_l2_node)
    graph.add_node("respond_from_cache", respond_from_cache_node)
    graph.add_node("schema_retriever", schema_retriever_node)
    graph.add_node("sql_generator", sql_generator_node)
    graph.add_node("sql_validator", sql_validator_node)
    graph.add_node("validation_failed", validation_failed_node)
    graph.add_node("approval_agent", approval_agent_node)
    graph.add_node("sql_executor", sql_executor_node)
    graph.add_node("response_synthesizer", response_synthesizer_node)

    # ── Entry → Phase 0: Intent detection ──
    graph.add_edge(START, "intent_detector")

    # ── Phase 0 → route by intent ──
    graph.add_conditional_edges("intent_detector", after_intent_detection, {
        "react_agent": "react_agent",
        "parallel_init": "parallel_init",
    })

    # ── ReAct loop: react_agent → loop or END ──
    graph.add_conditional_edges("react_agent", after_react, {
        "react_agent": "react_agent",
        END: END,
    })

    # ── Phase 1 → conditional routing ──
    graph.add_conditional_edges("parallel_init", after_parallel_init)

    # ── Complexity → conditional routing ──
    graph.add_conditional_edges("complexity_detector", after_complexity)

    # ── Phase 2: Ambiguity → conditional routing ──
    graph.add_conditional_edges("ambiguity_agent", after_ambiguity)

    # ── Phase 3: Cache L2 → conditional routing ──
    graph.add_conditional_edges("cache_l2", after_cache_l2)

    # ── Cache terminal → END ──
    graph.add_edge("respond_from_cache", END)

    # ── Phase 4: Schema → SQL Gen → Validation (with retry loop) ──
    graph.add_edge("schema_retriever", "sql_generator")
    graph.add_conditional_edges("sql_generator", after_sql_gen)
    graph.add_conditional_edges("sql_validator", after_sql_validation)
    graph.add_edge("validation_failed", END)

    # ── Phase 5: Approval → conditional routing ──
    graph.add_conditional_edges("approval_agent", after_approval)

    # ── Phase 6: Execution → Response → END ──
    graph.add_conditional_edges("sql_executor", after_execution)
    graph.add_edge("response_synthesizer", END)

    # ── Compile with PostgresSaver for distributed checkpointing ──
    db_url = settings.database_url_sync.replace("postgresql://", "postgresql://", 1)
    # psycopg expects raw postgresql:// URI (not sqlalchemy+driver format)
    conn = Connection.connect(db_url, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)

