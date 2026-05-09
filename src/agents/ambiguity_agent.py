"""Ambiguity Resolution Agent - LLM-based detection with dynamic DB prompts."""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from src.core import get_settings
from src.core.state import AgentState
from src.core.prompts import get_prompt
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter
from src.services.guardrails_service import validate_input
import structlog

log = structlog.get_logger()
settings = get_settings()


def _get_llm():
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_fast_model,
        temperature=0,
    )


@trace_agent_node("ambiguity_agent")
def ambiguity_agent_node(state: AgentState) -> dict:
    """Resolve ambiguity in user queries using LLM with dynamic DB prompt.

    Step 1: Guardrails injection check
    Step 2: LLM-based ambiguity detection (always, using DB prompt)

    Output: {is_ambiguous, rewritten_query, clarification_message, clarification_options}
    """
    query = state.get("original_query", "")
    messages = state.get("messages", [])
    history = state.get("conversation_history", [])
    summary = state.get("conversation_summary", "")

    # Step 1: Guardrails injection check
    is_safe, issues = validate_input(query)
    if not is_safe:
        return {
            "messages": messages,
            "status": "failed",
            "error": f"Query blocked: {'; '.join(issues)}",
            "is_ambiguous": False,
            "rewritten_query": "",
        }

    # Step 2: LLM-based ambiguity detection
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def resolve_ambiguity(
        is_ambiguous: bool,
        rewritten_query: str = "",
        clarification_message: str = "",
        clarification_options: list = [],
        ambiguity_score: float = 0.0,
        rewrite_confidence: float = 0.8,
    ) -> str:
        """Resolve query ambiguity. If clear, provide rewritten_query with confidence. If ambiguous, provide clarification options with score."""
        return json.dumps({
            "is_ambiguous": is_ambiguous,
            "rewritten_query": rewritten_query,
            "clarification_message": clarification_message,
            "clarification_options": clarification_options,
            "ambiguity_score": ambiguity_score,
            "rewrite_confidence": rewrite_confidence,
        })

    llm = _get_llm().bind_tools([resolve_ambiguity], tool_choice="auto")

    # Build context from history
    context_parts = []
    if summary:
        context_parts.append(f"Conversation summary: {summary}")
    if history:
        recent = history[-5:] if len(history) > 5 else history
        context_parts.append("Recent messages:\n" + "\n".join(
            [f"  {h['role']}: {h['content']}" for h in recent]
        ))
    context = "\n".join(context_parts) if context_parts else "No prior conversation context."

    try:
        system_prompt = get_prompt("ambiguity_resolution")
    except Exception as e:
        log.error("prompt_load_failed", prompt="ambiguity_resolution", error=str(e))
        # If prompt unavailable, treat query as clear and pass through
        return {
            "messages": state.get("messages", []),
            "is_ambiguous": False,
            "rewritten_query": query,
            "status": "processing",
        }

    user_content = f"Context:\n{context}\n\nUser query: {query}"

    try:
        response = resilient_call(
            llm.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )

        if response.tool_calls:
            tc = response.tool_calls[0]["args"]
            is_ambiguous = tc.get("is_ambiguous", False)
            rewritten = tc.get("rewritten_query", query)

            if is_ambiguous:
                options = tc.get("clarification_options", [])
                formatted_options = []
                for i, opt in enumerate(options):
                    if isinstance(opt, str):
                        formatted_options.append({"index": i + 1, "query": opt, "reason": ""})
                    elif isinstance(opt, dict):
                        formatted_options.append({
                            "index": opt.get("index", i + 1),
                            "query": opt.get("query", opt.get("option", str(opt))),
                            "reason": opt.get("reason", ""),
                        })

                return {
                    "messages": messages,
                    "is_ambiguous": True,
                    "rewritten_query": "",
                    "clarification_message": tc.get("clarification_message", "Could you clarify your question?"),
                    "clarification_options": formatted_options,
                    "status": "awaiting_clarification",
                    "ambiguity_score": tc.get("ambiguity_score", 0.8),
                    "rewrite_confidence": 0.0,
                }
            else:
                return {
                    "messages": messages,
                    "is_ambiguous": False,
                    "ambiguity_score": tc.get("ambiguity_score", 0.1),
                    "rewrite_confidence": tc.get("rewrite_confidence", 0.8),
                    "rewritten_query": rewritten if rewritten else query,
                }
        else:
            return {
                "messages": messages,
                "is_ambiguous": False,
                "ambiguity_score": 0.2,
                "rewrite_confidence": 0.7,
                "rewritten_query": query,
            }

    except Exception as e:
        error_str = str(e)
        if "tool_use_failed" in error_str and "failed_generation" in error_str:
            import re as _re
            is_amb_match = _re.search(r'"is_ambiguous":\s*(true|false)', error_str, _re.IGNORECASE)
            rewrite_match = _re.search(r'"rewritten_query":\s*"([^"]+)"', error_str)
            clarify_match = _re.search(r'"clarification_message":\s*"([^"]+)"', error_str)

            if is_amb_match:
                is_ambiguous_val = is_amb_match.group(1).lower() == "true"
                if is_ambiguous_val and clarify_match:
                    return {
                        "messages": messages,
                        "is_ambiguous": True,
                        "ambiguity_score": 0.7,
                        "rewrite_confidence": 0.0,
                        "rewritten_query": "",
                        "clarification_message": clarify_match.group(1),
                        "clarification_options": [],
                        "status": "awaiting_clarification",
                    }
                elif not is_ambiguous_val and rewrite_match:
                    return {
                        "messages": messages,
                        "is_ambiguous": False,
                        "ambiguity_score": 0.2,
                        "rewrite_confidence": 0.6,
                        "rewritten_query": rewrite_match.group(1),
                    }

        log.error("ambiguity_agent_error", error=error_str[:200])
        return {
            "messages": messages,
            "is_ambiguous": False,
            "ambiguity_score": 0.5,
            "rewrite_confidence": 0.5,
            "rewritten_query": query,
        }
