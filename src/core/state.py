"""LangGraph Agent State definition for the multi-agent SQL pipeline."""

from __future__ import annotations
from typing import TypedDict, Annotated, Literal
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the phase-based DAG graph (v6.0).

    Pipeline: Parallel(Memory + CacheL1 + Embedding) → Ambiguity → CacheL2 →
              Schema → SQL Gen → SQL Validate → Approval → Execute → Response

    Includes: relevance filtering, confidence scoring, cost awareness,
    adaptive complexity, decision tracing, structured output.
    """

    # Session
    session_id: str
    original_query: str

    # Memory & History
    conversation_history: list[dict]  # Recent messages (relevance-filtered)
    conversation_summary: str  # Incremental summary of older messages
    history_token_usage: int  # Token count of history payload

    # Ambiguity Resolution
    rewritten_query: str  # Clarified/rewritten query (canonical form)
    is_ambiguous: bool
    ambiguity_score: float  # 0.0 (clear) to 1.0 (fully ambiguous)
    rewrite_confidence: float  # Confidence in the rewrite quality
    clarification_message: str
    clarification_options: list[dict]

    # Dual-Layer Cache
    cache_hit: bool  # Final cache hit flag
    l1_checked: bool  # Whether L1 (raw query) was checked
    l2_hit: bool  # Whether L2 (canonical query) hit
    cached_response: dict  # Cached result if found

    # Embedding
    query_embedding: list[float]  # Dense embedding vector
    embedding_done: bool  # Whether embedding was attempted

    # Schema Retrieval
    schema_context: str
    tables_used: list[str]
    schema_relationships: list[str]

    # SQL Generation
    generated_sql: str
    sql_confidence: float

    # SQL Validation
    validation_errors: list[str]
    retry_count: int
    sql_validated: bool
    estimated_cost: str  # Query cost assessment (low/medium/high)

    # Execution
    results: list[dict]

    # Response
    explanation: str
    structured_output: dict  # Machine-friendly structured response

    # Control
    status: Literal[
        "processing",
        "completed",
        "failed",
        "awaiting_approval",
        "awaiting_clarification",
        "awaiting_tool_approval",
        "action_rejected",
    ]
    error: str
    next_agent: str

    # Human-in-the-loop
    require_approval: bool
    approved: bool
    approval_explanation: str  # NL explanation of SQL intent for user

    # Adaptive Pipeline
    query_complexity: str  # "simple", "moderate", "complex"

    # Decision Tracing
    decision_trace: list[dict]  # [{agent, decision, reason, timestamp}]

    # Intent Detection (read vs action)
    intent: str  # "read" | "action" | "unknown"

    # ReAct Agent (action pipeline)
    react_steps: list[dict]   # [{step, tool, args, reasoning, approved, success, message, result}]
    react_result: str          # Final summary produced by ReAct agent
    pending_tool_call: dict    # Set during interrupt: {tool_name, args, reasoning, step}
