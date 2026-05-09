"""User Approval Agent (HITL) - Configurable human-in-the-loop gate.

Modes:
  - Dev Mode (require_approval=False): Auto-approve all queries
  - Prod Mode (require_approval=True): Human approval via LangGraph interrupt()
"""

import re
from langgraph.types import interrupt
from src.core.state import AgentState
from src.core.tracing import trace_agent_node
import structlog

log = structlog.get_logger()


def _generate_explanation(sql: str, tables_used: list[str], confidence: float, cost: str) -> str:
    """Generate a natural language explanation of what the SQL will do."""
    sql_upper = sql.upper()
    parts = []

    # Determine action type
    if "COUNT(" in sql_upper:
        parts.append("Count")
    elif "SUM(" in sql_upper:
        parts.append("Calculate the total of")
    elif "AVG(" in sql_upper:
        parts.append("Calculate the average of")
    elif "MAX(" in sql_upper or "MIN(" in sql_upper:
        parts.append("Find the extreme value of")
    else:
        parts.append("Retrieve")

    # Mention tables
    if tables_used:
        parts.append(f"data from {', '.join(tables_used)}")
    else:
        # Extract from SQL
        found_tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
        if found_tables:
            parts.append(f"data from {', '.join(found_tables)}")

    # Mention filtering
    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        parts.append(f"filtered by conditions")

    # Mention ordering
    if "ORDER BY" in sql_upper:
        if "DESC" in sql_upper:
            parts.append("sorted in descending order")
        else:
            parts.append("sorted in ascending order")

    # Mention limit
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        parts.append(f"limited to {limit_match.group(1)} rows")

    # Add cost/confidence context
    explanation = " ".join(parts) + "."
    if cost == "high":
        explanation += f" [Cost: HIGH - may take longer]"
    explanation += f" Confidence: {confidence:.0%}."

    return explanation


@trace_agent_node("approval_agent")
def approval_agent_node(state: AgentState) -> dict:
    """Configurable approval gate with NL explanation.

    Dev Mode: Auto-approve (require_approval=False)
    Prod Mode: Show SQL + NL explanation, wait for human approval
    """
    sql = state.get("generated_sql", "")
    confidence = state.get("sql_confidence", 0.7)
    require_approval = state.get("require_approval", False)
    tables_used = state.get("tables_used", [])
    cost = state.get("estimated_cost", "low")
    messages = state.get("messages", [])

    # Generate NL explanation regardless of mode (for tracing/logging)
    explanation = _generate_explanation(sql, tables_used, confidence, cost)

    # Dev Mode: auto-approve
    if not require_approval:
        log.info("approval_auto", mode="dev", explanation=explanation[:100])
        return {
            "messages": messages,
            "approved": True,
            "approval_explanation": explanation,
        }

    # Prod Mode: interrupt for human approval with explanation
    log.info("approval_requested", mode="prod", confidence=confidence)
    approval_response = interrupt({
        "type": "approval_request",
        "sql": sql,
        "confidence": confidence,
        "tables_used": tables_used,
        "estimated_cost": cost,
        "explanation": explanation,
        "message": f"{explanation}\n\nApprove executing this SQL? (confidence: {confidence:.0%})\n\n{sql}",
    })

    # Graph resumes here after user responds
    if not approval_response or not approval_response.get("approved", False):
        return {
            "messages": messages,
            "approved": False,
            "approval_explanation": explanation,
            "status": "failed",
            "error": "Query execution rejected by user",
        }

    return {
        "messages": messages,
        "approved": True,
        "approval_explanation": explanation,
    }
