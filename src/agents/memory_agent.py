"""Memory & History Agent - Relevance-filtered, token-aware conversation memory.

Features:
  - Semantic relevance filtering (score history against current query)
  - Token-aware summarization (incremental, only when threshold exceeded)
  - Keeps high-relevance messages even if older
  - Drops irrelevant history even if recent
"""

import re
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from src.core import get_settings
from src.core.state import AgentState
from src.core.database import get_conversations, get_conversation_summary, save_conversation_summary
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter
from src.services.cache import _get_embeddings
import structlog

log = structlog.get_logger()
settings = get_settings()


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English)."""
    return len(text) // 4


def _get_llm():
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )


def _relevance_filter(query: str, history: list[dict], query_embedding: list[float] | None = None) -> list[dict]:
    """Filter conversation history by relevance to current query.

    Uses semantic similarity when embeddings are available,
    falls back to keyword overlap scoring.
    Returns only messages with relevance > threshold.
    """
    if not history:
        return []

    RELEVANCE_THRESHOLD = 0.3
    emb_model = _get_embeddings()

    # If we have embedding model and query embedding, use semantic scoring
    if emb_model and query_embedding and len(query_embedding) > 0:
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return history[-5:]  # Fallback to recent

            # Embed history messages
            history_texts = [h.get("content", "") for h in history]
            history_vectors = emb_model.embed_documents(history_texts)

            scored = []
            for i, h in enumerate(history):
                h_vec = np.array(history_vectors[i], dtype=np.float32)
                h_norm = np.linalg.norm(h_vec)
                if h_norm == 0:
                    continue
                similarity = float(np.dot(query_vec, h_vec) / (query_norm * h_norm))
                # Recency boost: more recent messages get slight bonus
                recency_boost = 0.05 * (i / max(len(history), 1))
                scored.append((similarity + recency_boost, h))

            # Keep messages above threshold, sorted by relevance
            relevant = [(s, h) for s, h in scored if s >= RELEVANCE_THRESHOLD]
            relevant.sort(key=lambda x: x[0], reverse=True)

            # Cap at reasonable size
            result = [h for _, h in relevant[:10]]
            if not result:
                return history[-3:]  # Always keep some context
            return result
        except Exception as e:
            log.debug("relevance_filter_embedding_failed", error=str(e))

    # Fallback: keyword overlap scoring
    query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    if not query_tokens:
        return history[-5:]

    scored = []
    for i, h in enumerate(history):
        content = h.get("content", "").lower()
        h_tokens = set(re.findall(r'\b\w{3,}\b', content))
        if not h_tokens:
            continue
        overlap = len(query_tokens & h_tokens) / max(len(query_tokens), 1)
        recency_boost = 0.1 * (i / max(len(history), 1))
        scored.append((overlap + recency_boost, h))

    # Keep relevant messages
    relevant = [(s, h) for s, h in scored if s >= RELEVANCE_THRESHOLD]
    relevant.sort(key=lambda x: x[0], reverse=True)

    result = [h for _, h in relevant[:10]]
    if not result:
        return history[-3:]
    return result


def _summarize_messages(messages: list[dict], existing_summary: str) -> str:
    """Perform incremental summarization of older messages using dynamic DB prompt."""
    from src.core.prompts import get_prompt

    llm = _get_llm()

    messages_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages]
    )

    try:
        system_content = get_prompt("memory_summarization")
    except Exception as e:
        log.error("prompt_load_failed", prompt="memory_summarization", error=str(e))
        # Return existing summary if we can't load the prompt
        return existing_summary or "Summary unavailable - prompt not loaded."

    user_content = f"Previous summary: {existing_summary or 'None'}\n\nNew messages to incorporate:\n{messages_text}"

    try:
        response = resilient_call(
            llm.invoke,
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )
        return response.content
    except Exception as e:
        log.warning("summarization_failed", error=str(e))
        return existing_summary or ""


@trace_agent_node("memory_agent", prompt_key="memory_summarization")
def memory_agent_node(state: AgentState) -> dict:
    """Load conversation history with relevance filtering and token-aware summarization.

    Logic:
    1. Load full conversation history
    2. Filter by relevance to current query (semantic or keyword)
    3. If token limit exceeded: summarize older messages
    4. Return only relevant, token-budgeted history

    Output: relevant_messages, summary, token_usage
    """
    session_id = state.get("session_id", "")
    messages = state.get("messages", [])
    query = state.get("original_query", "")
    query_embedding = state.get("query_embedding", [])

    if not session_id:
        return {
            "messages": messages,
            "conversation_history": [],
            "conversation_summary": "",
            "history_token_usage": 0,
        }

    # Load full conversation history
    history = get_conversations(session_id, limit=50)

    if not history:
        return {
            "messages": messages,
            "conversation_history": [],
            "conversation_summary": "",
            "history_token_usage": 0,
        }

    # Load existing summary
    existing_summary = get_conversation_summary(session_id) or ""

    # Relevance filtering: keep only messages relevant to current query
    relevant_history = _relevance_filter(query, history, query_embedding or None)

    log.info("memory_relevance_filtered",
             total=len(history), kept=len(relevant_history), query=query[:50])

    # Compute token usage
    total_text = " ".join([h["content"] for h in relevant_history])
    total_tokens = _estimate_tokens(total_text)
    summary_tokens = _estimate_tokens(existing_summary)
    total_tokens += summary_tokens

    # Check thresholds
    token_limit = settings.memory_token_limit
    max_messages = settings.memory_max_messages

    if total_tokens > token_limit or len(relevant_history) > max_messages:
        # Split: keep most relevant, summarize the rest
        keep_count = min(5, len(relevant_history))
        kept_messages = relevant_history[:keep_count]  # Already sorted by relevance
        overflow_messages = relevant_history[keep_count:]

        if overflow_messages:
            updated_summary = _summarize_messages(overflow_messages, existing_summary)
            summary_token_count = _estimate_tokens(updated_summary)
            kept_token_count = _estimate_tokens(" ".join([m["content"] for m in kept_messages]))
            save_conversation_summary(
                session_id, updated_summary, kept_token_count + summary_token_count
            )
        else:
            updated_summary = existing_summary

        final_token_usage = _estimate_tokens(updated_summary) + _estimate_tokens(
            " ".join([m["content"] for m in kept_messages])
        )

        return {
            "messages": messages,
            "conversation_history": kept_messages,
            "conversation_summary": updated_summary,
            "history_token_usage": final_token_usage,
        }

    # Under threshold - return relevance-filtered history
    return {
        "messages": messages,
        "conversation_history": relevant_history,
        "conversation_summary": existing_summary,
        "history_token_usage": total_tokens,
    }
