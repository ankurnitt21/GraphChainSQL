"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    require_approval: bool = False


class ClarifyRequest(BaseModel):
    session_id: str
    selected_index: Optional[int] = None
    custom_query: Optional[str] = None


class ApproveRequest(BaseModel):
    session_id: str
    approved: bool
    original_query: str = ""
    generated_sql: str = ""


class ActionApproveRequest(BaseModel):
    """Approve or reject a ReAct tool call (HITL for action pipeline)."""
    session_id: str
    approved: bool
    feedback: Optional[str] = None  # Optional rejection reason


class FeedbackRequest(BaseModel):
    """User feedback on query results (thumbs up/down)."""
    session_id: str
    query: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    generated_sql: Optional[str] = None
    comment: Optional[str] = None
    correction: Optional[str] = None  # User-provided correct SQL
    run_id: Optional[str] = None  # LangSmith run ID for trace linking


class ClarifyOption(BaseModel):
    index: int
    query: str
    reason: str


class QueryResponse(BaseModel):
    session_id: str
    status: str
    original_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    generated_sql: Optional[str] = None
    confidence: Optional[float] = None
    results: Optional[list] = None
    explanation: Optional[str] = None
    tables_used: Optional[list[str]] = None
    cache_hit: bool = False
    error: Optional[str] = None
    clarification_message: Optional[str] = None
    clarification_options: Optional[list[ClarifyOption]] = None
    duration_ms: Optional[int] = None
    # v6.0 fields
    structured_output: Optional[dict] = None
    decision_trace: Optional[list[dict]] = None
    estimated_cost: Optional[str] = None
    approval_explanation: Optional[str] = None
    # ReAct / Action pipeline fields
    intent: Optional[str] = None
    react_steps: Optional[list[dict]] = None
    react_result: Optional[str] = None
    pending_tool_call: Optional[dict] = None
    # Feedback integration
    run_id: Optional[str] = None  # LangSmith run/trace ID for feedback linking
