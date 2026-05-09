"""Embedding Agent - Convert query to dense vector for schema retrieval."""

from src.core.state import AgentState
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, embedding_circuit
from src.services.cache import _get_embeddings
import structlog

log = structlog.get_logger()


@trace_agent_node("embedding_agent")
def embedding_agent_node(state: AgentState) -> dict:
    """Convert the final user query into a dense embedding vector.

    Uses the same embedding model as the cache (all-MiniLM-L6-v2).
    The embedding is used downstream by the Schema Retrieval Agent for
    similarity search over schema descriptions.

    Avoids duplicate embedding if already computed (e.g., by cache lookup).
    """
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    messages = state.get("messages", [])

    # Skip if embedding already computed (avoid duplicate calls)
    existing_embedding = state.get("query_embedding", [])
    if existing_embedding and len(existing_embedding) > 0:
        log.info("embedding_skipped_already_computed", dim=len(existing_embedding))
        return {
            "messages": messages,
            "query_embedding": existing_embedding,
            "embedding_done": True,
        }

    if not query:
        return {
            "messages": messages,
            "query_embedding": [],
            "embedding_done": True,
        }

    emb_model = _get_embeddings()
    if emb_model is None:
        log.warning("embedding_model_unavailable")
        return {
            "messages": messages,
            "query_embedding": [],
            "embedding_done": True,
        }

    try:
        embedding = resilient_call(
            emb_model.embed_query,
            query,
            circuit=embedding_circuit,
        )
        log.info("query_embedded", dim=len(embedding), query=query[:50])
        return {
            "messages": messages,
            "query_embedding": embedding,
            "embedding_done": True,
        }
    except Exception as e:
        log.error("embedding_error", error=str(e))
        return {
            "messages": messages,
            "query_embedding": [],
            "embedding_done": True,
        }
