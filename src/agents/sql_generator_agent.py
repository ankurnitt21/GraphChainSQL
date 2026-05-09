"""SQL Generation Agent - Creates SQL from natural language using schema context."""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from src.core import get_settings
from src.core.state import AgentState
from src.core.prompts import get_prompt
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter
import structlog

log = structlog.get_logger()
settings = get_settings()


def _get_llm():
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_fast_model,
        temperature=0,
    )


@trace_agent_node("sql_generator")
def sql_generator_node(state: AgentState) -> dict:
    """Generate SQL from the rewritten query and retrieved schema.

    Input: rewritten_query, schema_context, conversation history
    Output: generated_sql, sql_confidence, tables_used
    """
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    original_query = state.get("original_query", "")
    # Safety: if rewritten_query looks like SQL, use original_query instead
    if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("INSERT"):
        query = original_query
    schema = state.get("schema_context", "")
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", [])
    history = state.get("conversation_history", [])
    summary = state.get("conversation_summary", "")

    # Build conversation context
    context_parts = []
    if summary:
        context_parts.append(f"Previous conversation summary: {summary}")
    if history:
        recent = history[-3:]
        context_parts.append("Recent context:\n" + "\n".join(
            [f"  {h['role']}: {h['content']}" for h in recent]
        ))
    context = "\n".join(context_parts) if context_parts else ""

    # Error context for retries
    error_context = ""
    if validation_errors:
        error_context = f"\n\nPrevious attempt had these errors: {'; '.join(validation_errors)}. Fix them."

    # Function calling for structured output
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def generate_sql(sql: str, confidence: float, tables_used: list[str], reasoning: str) -> str:
        """Submit the generated SQL query with confidence and metadata."""
        return json.dumps({"sql": sql, "confidence": confidence, "tables_used": tables_used})

    llm = _get_llm().bind_tools([generate_sql], tool_choice="auto")

    # Fetch prompt from DB
    try:
        system_content = get_prompt("sql_generation").format(
            schema=schema,
            query=query,
            context=f"{context}{error_context}" if context or error_context else "",
        )
    except Exception as e:
        log.error("prompt_load_failed", prompt="sql_generation", error=str(e))
        return {
            "messages": messages,
            "status": "failed",
            "error": f"Prompt 'sql_generation' unavailable: {e}",
        }

    updates = {
        "messages": messages,
        "retry_count": retry_count + 1,
        "validation_errors": [],
        "sql_validated": False,
    }

    try:
        response = resilient_call(
            llm.invoke,
            [SystemMessage(content=system_content), HumanMessage(content=query)],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )

        if response.tool_calls:
            tc = response.tool_calls[0]["args"]
            updates["generated_sql"] = tc.get("sql", "")
            updates["sql_confidence"] = tc.get("confidence", 0.7)
            updates["tables_used"] = tc.get("tables_used", [])
        else:
            # Fallback: parse from text
            last_msg = response.content
            json_match = re.search(r'\{.*\}', last_msg, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                updates["generated_sql"] = parsed.get("sql", "")
                updates["sql_confidence"] = parsed.get("confidence", 0.7)
                updates["tables_used"] = parsed.get("tables_used", [])
            else:
                sql_match = re.search(
                    r'SELECT\s+.+?(?:LIMIT\s+\d+|$)', last_msg, re.DOTALL | re.IGNORECASE
                )
                if sql_match:
                    updates["generated_sql"] = sql_match.group().strip().rstrip(";")
                    updates["sql_confidence"] = 0.6

    except Exception as e:
        error_str = str(e)
        # Groq sometimes returns tool_use_failed with valid SQL in failed_generation
        if "tool_use_failed" in error_str and "failed_generation" in error_str:
            # Extract the JSON object from the failed_generation field
            # Look for the JSON after <function=generate_sql>
            json_in_error = re.search(r'\{"sql":\s*"(.+?)".*?"confidence":\s*([\d.]+).*?"tables_used":\s*\[([^\]]*)\]', error_str, re.DOTALL)
            if json_in_error:
                sql = json_in_error.group(1).replace('\\"', '"')
                confidence = float(json_in_error.group(2))
                tables_raw = json_in_error.group(3)
                tables = [t.strip().strip('"').strip("'") for t in tables_raw.split(",") if t.strip()]
                updates["generated_sql"] = sql
                updates["sql_confidence"] = confidence
                updates["tables_used"] = tables
                log.info("sql_parsed_from_failed_generation", sql=sql[:100])
                return updates
            # Simpler fallback: just grab the SQL directly
            sql_match = re.search(r'"sql":\s*"(SELECT\s+.+?LIMIT\s+\d+)"', error_str, re.DOTALL | re.IGNORECASE)
            if sql_match:
                updates["generated_sql"] = sql_match.group(1).replace('\\"', '"')
                updates["sql_confidence"] = 0.5
                log.info("sql_extracted_from_error", sql=updates["generated_sql"][:100])
                return updates

        log.error("sql_generation_error", error=error_str[:200])
        updates["error"] = f"SQL generation failed: {error_str}"
        updates["status"] = "failed"

    # Self-consistency check: verify generated SQL answers the question using LLM
    if updates.get("generated_sql") and not updates.get("error"):
        updates = _self_consistency_check(updates, query)

    return updates


def _self_consistency_check(updates: dict, query: str) -> dict:
    """LLM-based self-consistency verification.

    Asks the LLM whether the generated SQL structurally answers the query.
    Returns a confidence adjustment if misaligned.
    """
    sql = updates.get("generated_sql", "")

    try:
        system_content = get_prompt("sql_self_consistency")
    except Exception as e:
        log.warning("prompt_load_failed", prompt="sql_self_consistency", error=str(e))
        return updates  # Skip consistency check if prompt unavailable

    try:
        from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM
        llm = _get_llm()
        response = resilient_call(
            llm.invoke,
            [
                _SM(content=system_content),
                _HM(content=f"Question: {query}\nSQL: {sql}"),
            ],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )
        import json as _json
        data = _json.loads(response.content.strip())
        penalty = float(data.get("penalty", 0.0))
        if penalty > 0:
            original_confidence = updates.get("sql_confidence", 0.7)
            updates["sql_confidence"] = max(0.1, original_confidence - penalty)
            log.info("self_consistency_penalty", penalty=round(penalty, 2), new_confidence=updates["sql_confidence"])
    except Exception as e:
        log.debug("self_consistency_check_skipped", error=str(e))

    return updates
