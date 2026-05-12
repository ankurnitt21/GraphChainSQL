"""Response Resynthesis Agent - Smart response generation.

Strategy:
  - Simple results (single row, count, sum) → Template-based response (no LLM)
  - Complex results (multiple rows, joins) → LLM-powered explanation
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from src.core import get_settings
from src.core.state import AgentState
from src.core.prompts import get_prompt
from src.core.database import save_conversation
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter
from src.services.cache import semantic_cache_set
from src.services.guardrails_service import validate_output
import structlog

log = structlog.get_logger()
settings = get_settings()


def _is_simple_result(results: list[dict], query_complexity: str = "moderate") -> bool:
    """Determine if results are simple enough for a template response (no LLM).

    Template is used when:
      - No results (empty set)
      - Single row (any column count) — e.g. aggregate/count queries
      - query_complexity == "simple" and ≤ 10 rows — e.g. "top 5", "list all X"
      - Up to 3 rows with a single column — simple enumeration
    """
    if not results:
        return True
    # Single aggregate/count row — always template
    if len(results) == 1:
        return True
    # Small ranked/list result when the query is already classified simple
    if query_complexity == "simple" and len(results) <= 10:
        return True
    # Up to 3 single-column rows
    if len(results) <= 3 and all(len(r) == 1 for r in results):
        return True
    return False


def _template_response(query: str, results: list[dict]) -> str | None:
    """Generate a template-based response for simple results (no LLM needed).

    Returns None only if the result shape is unexpectedly complex so the caller
    can fall through to the LLM path.
    """
    if not results:
        return "The query returned no results."

    # Single aggregate row (e.g. COUNT, SUM)
    if len(results) == 1:
        row = results[0]
        if len(row) == 1:
            key, value = next(iter(row.items()))
            return f"The result is **{value}**."
        parts = [f"**{k}**: {v}" for k, v in row.items()]
        return "Result: " + ", ".join(parts) + "."

    # Single-column list
    if all(len(r) == 1 for r in results):
        key = next(iter(results[0].keys()))
        values = [str(r[key]) for r in results]
        return f"Results ({len(results)} items): {', '.join(values)}."

    # Multi-column ranked/list result — build a compact table summary
    cols = list(results[0].keys())
    rows_text = []
    for i, row in enumerate(results, 1):
        parts = [f"{v}" for v in row.values()]
        rows_text.append(f"{i}. " + " | ".join(parts))
    header = " | ".join(cols)
    return f"Top {len(results)} results ({header}):\n" + "\n".join(rows_text)


def _get_llm():
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )


@trace_agent_node("response_synthesizer", prompt_key="response_synthesis")
def response_synthesizer_node(state: AgentState) -> dict:
    """Convert raw SQL results into a natural language response.

    Uses template for simple results, LLM for complex ones.
    Also: caches result, saves history, validates for PII.
    """
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    original_query = state.get("original_query", "")
    sql = state.get("generated_sql", "")
    results = state.get("results", [])
    messages = state.get("messages", [])
    session_id = state.get("session_id", "")

    query_complexity = state.get("query_complexity", "moderate")

    # Strategy: Template vs LLM
    if _is_simple_result(results, query_complexity):
        explanation = _template_response(query, results)
        if explanation:
            log.info("response_template", rows=len(results), complexity=query_complexity)
        else:
            explanation = None  # Fall through to LLM
    else:
        explanation = None

    # LLM path for complex results
    if explanation is None:
        results_preview = json.dumps(results[:10], default=str) if results else "[]"
        llm = _get_llm()

        try:
            prompt_content = get_prompt("response_synthesis").format(
                query=query, sql=sql, results=results_preview, total_rows=len(results)
            )
        except Exception as e:
            log.error("prompt_load_failed", prompt="response_synthesis", error=str(e))
            prompt_content = f"Question: {query}\nSQL: {sql}\nResults ({len(results)} rows): {results_preview}"

        try:
            system_content = get_prompt("response_system")
        except Exception as e:
            log.error("prompt_load_failed", prompt="response_system", error=str(e))
            return {
                "messages": messages,
                "explanation": f"Query returned {len(results)} rows. (Prompt unavailable)",
                "status": "completed",
            }

        try:
            response = resilient_call(
                llm.invoke,
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=prompt_content),
                ],
                circuit=llm_circuit,
                rate_limiter=llm_rate_limiter,
            )
            explanation = response.content
        except Exception as e:
            explanation = f"Query returned {len(results)} rows."

    # Validate output for PII.
    # Skip the expensive LLM PII guard for SQL data results — warehouse output
    # (product names, prices, quantities) never contains personal data.
    # Use full LLM guard only for free-text explanations from complex queries.
    use_llm_pii = query_complexity == "complex"
    is_clean, pii_issues, cleaned_explanation = validate_output(explanation, use_llm_pii=use_llm_pii)
    if not is_clean:
        log.warning("pii_detected_in_response", issues=pii_issues)
        explanation = cleaned_explanation

    # Quality-controlled cache write with metadata
    try:
        cache_entry = {
            "sql": sql,
            "explanation": explanation,
            "results": results[:50],
        }
        cache_metadata = {
            "raw_query": original_query,
            "canonical_query": query,
            "confidence": state.get("sql_confidence", 0.0),
            "latency_ms": 0,  # Will be set by caller if needed
            "validated": state.get("sql_validated", False),
            "executed": True,  # We reached response_agent, so execution succeeded
            "ambiguous": state.get("is_ambiguous", False),
        }
        # Reuse precomputed embedding to avoid duplicate embed call
        precomputed_emb = state.get("query_embedding", [])
        semantic_cache_set(original_query, cache_entry,
                          precomputed_embedding=precomputed_emb,
                          metadata=cache_metadata)
        if query != original_query:
            semantic_cache_set(query, cache_entry,
                              precomputed_embedding=precomputed_emb,
                              metadata=cache_metadata)
    except Exception:
        pass

    # Save conversation history
    if session_id:
        try:
            save_conversation(session_id, "USER", original_query)
            save_conversation(session_id, "ASSISTANT", explanation, sql_query=sql)
        except Exception:
            pass

    # Build structured output for API consumers
    structured_output = {
        "answer": explanation,
        "sql": sql,
        "raw_data": results[:50],
        "total_rows": len(results),
        "tables_used": state.get("tables_used", []),
        "confidence": state.get("sql_confidence", 0.0),
        "estimated_cost": state.get("estimated_cost", "low"),
        "metadata": {
            "cache_hit": False,
            "ambiguity_score": state.get("ambiguity_score", 0.0),
            "rewrite_confidence": state.get("rewrite_confidence", 1.0),
            "validated": state.get("sql_validated", False),
            "query_complexity": state.get("query_complexity", "moderate"),
        },
    }

    return {
        "messages": messages,
        "explanation": explanation,
        "structured_output": structured_output,
        "status": "completed",
        "cache_hit": False,
    }
