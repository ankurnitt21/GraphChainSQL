"""Dual-Layer Semantic Cache Agent - L1 (raw query) + L2 (canonical query).

L1 Cache: Pre-ambiguity check on raw user query (speed optimization).
L2 Cache: Post-ambiguity check on rewritten/canonical query (accuracy).

Uses precomputed embeddings when available to avoid duplicate embed calls.
"""

import json
from src.core.state import AgentState
from src.core.tracing import trace_agent_node
from src.services.cache import semantic_cache_get
import structlog

log = structlog.get_logger()


@trace_agent_node("cache_l1")
def cache_l1_node(state: AgentState) -> dict:
    """L1 Cache: Check raw user query for exact/semantic match.

    Purpose: Fast early exit before ambiguity resolution.
    Uses strict similarity threshold for high confidence.
    Passes precomputed embedding to avoid duplicate embed calls.
    """
    query = state.get("original_query", "")
    messages = state.get("messages", [])
    precomputed_emb = state.get("query_embedding", [])

    if not query:
        return {
            "messages": messages,
            "cache_hit": False,
            "l1_checked": True,
            "cached_response": {},
        }

    try:
        cached = semantic_cache_get(query, precomputed_embedding=precomputed_emb or None)
        if cached:
            log.info("cache_l1_hit", query=query[:50])
            return {
                "messages": messages,
                "cache_hit": True,
                "l1_checked": True,
                "cached_response": cached,
                "generated_sql": cached.get("sql", ""),
                "results": cached.get("results", []),
                "explanation": cached.get("explanation", ""),
                "status": "completed",
            }
    except Exception as e:
        log.warning("cache_l1_error", error=str(e))

    return {
        "messages": messages,
        "cache_hit": False,
        "l1_checked": True,
        "cached_response": {},
    }


@trace_agent_node("cache_l2")
def cache_l2_node(state: AgentState) -> dict:
    """L2 Cache: Check canonical/rewritten query after ambiguity resolution.

    Purpose: High-quality reuse of previously computed results.
    Uses the rewritten (canonical) query for lookup.
    Passes precomputed embedding to avoid duplicate embed calls.
    """
    rewritten = state.get("rewritten_query", "")
    original = state.get("original_query", "")
    messages = state.get("messages", [])
    precomputed_emb = state.get("query_embedding", [])

    # Use canonical query (rewritten) for L2 lookup
    lookup_query = rewritten if rewritten else original
    if not lookup_query:
        return {
            "messages": messages,
            "l2_hit": False,
            "cached_response": {},
        }

    try:
        # Check rewritten query first (canonical form)
        cached = semantic_cache_get(lookup_query, precomputed_embedding=precomputed_emb or None)
        # Also try original if different
        if not cached and lookup_query != original:
            cached = semantic_cache_get(original, precomputed_embedding=precomputed_emb or None)

        if cached:
            log.info("cache_l2_hit", query=lookup_query[:50])
            return {
                "messages": messages,
                "l2_hit": True,
                "cache_hit": True,
                "cached_response": cached,
                "generated_sql": cached.get("sql", ""),
                "results": cached.get("results", []),
                "explanation": cached.get("explanation", ""),
                "status": "completed",
            }
    except Exception as e:
        log.warning("cache_l2_error", error=str(e))

    return {
        "messages": messages,
        "l2_hit": False,
        "cached_response": {},
    }


# Keep backward-compatible alias
cache_agent_node = cache_l1_node
