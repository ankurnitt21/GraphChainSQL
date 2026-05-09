"""SQL Execution Agent - Executes validated SQL safely against the database."""

import json
from decimal import Decimal
from src.core.state import AgentState
from src.core.database import execute_query
from src.core.tracing import trace_agent_node
from src.services.guardrails_service import validate_sql
import structlog

log = structlog.get_logger()


@trace_agent_node("sql_executor")
def sql_executor_node(state: AgentState) -> dict:
    """Execute validated SQL query against the warehouse database.

    Uses function calling pattern - executes query with timeout and safety checks.
    Converts results to clean JSON-serializable format.
    """
    sql = state.get("generated_sql", "")
    messages = state.get("messages", [])

    if not sql:
        return {
            "messages": messages,
            "error": "No SQL to execute",
            "status": "failed",
        }

    # Final safety validation before execution
    is_safe, issues = validate_sql(sql)
    if not is_safe:
        return {
            "messages": messages,
            "error": f"SQL safety check failed: {'; '.join(issues)}",
            "status": "failed",
        }

    try:
        results = execute_query(sql, timeout=30)

        # Clean results (convert Decimal, datetime, etc.)
        clean_results = []
        for row in results:
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, Decimal):
                    clean_row[k] = float(v)
                else:
                    clean_row[k] = v
            clean_results.append(clean_row)

        log.info("sql_executed", row_count=len(clean_results), sql=sql[:100])
        return {
            "messages": messages,
            "results": clean_results,
        }

    except Exception as e:
        log.error("sql_execution_error", error=str(e), sql=sql[:100])
        return {
            "messages": messages,
            "error": f"SQL execution error: {str(e)}",
            "status": "failed",
        }
