"""Intent Detector - Classifies user query as 'read' (SELECT) or 'action' (mutating/external) using LLM."""

import json as _json
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


def _get_fast_llm():
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_fast_model,
        temperature=0,
    )


@trace_agent_node("intent_detector")
def intent_detector_node(state: AgentState) -> dict:
    """Classify user intent as 'read' or 'action' using LLM with dynamic DB prompt."""
    query = state.get("original_query", "")
    messages = state.get("messages", [])

    try:
        system_content = get_prompt("intent_detection")
    except Exception as e:
        log.error("prompt_load_failed", prompt="intent_detection", error=str(e))
        return {
            "messages": messages,
            "intent": "read",
            "status": "processing",
            "error": f"Prompt 'intent_detection' unavailable: {e}",
        }

    try:
        llm = _get_fast_llm()
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
        intent = data.get("intent", "read")
        if intent not in ("read", "action"):
            intent = "read"
        log.info("intent_detected", intent=intent, query=query[:60])
        return {"messages": messages, "intent": intent}
    except Exception as e:
        log.warning("intent_detection_failed", error=str(e), fallback="read")
        return {"messages": messages, "intent": "read"}
