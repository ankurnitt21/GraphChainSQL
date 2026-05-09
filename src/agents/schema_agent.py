"""Schema Retrieval Agent - Hybrid search (semantic + keyword) with RRF fusion."""

import re
import numpy as np
from src.core import get_settings
from src.core.state import AgentState
from src.core.database import get_full_schema_ddl, get_schema_descriptions
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, embedding_circuit
from src.services.cache import _get_embeddings
import structlog

log = structlog.get_logger()
settings = get_settings()

# Pre-computed schema embeddings cache (lazy init)
_schema_embeddings: list[dict] | None = None


def _get_schema_with_embeddings() -> list[dict]:
    """Get all schema descriptions with pre-computed embeddings (cached)."""
    global _schema_embeddings
    if _schema_embeddings is not None:
        return _schema_embeddings

    descriptions = get_schema_descriptions()
    emb_model = _get_embeddings()

    if emb_model is None:
        _schema_embeddings = descriptions
        return _schema_embeddings

    # Compute embeddings for each schema description
    texts = []
    for d in descriptions:
        text = f"{d['table_name']}"
        if d.get("column_name"):
            text += f".{d['column_name']}"
        text += f": {d['description']}"
        texts.append(text)

    try:
        vectors = resilient_call(
            emb_model.embed_documents,
            texts,
            circuit=embedding_circuit,
        )
        for i, d in enumerate(descriptions):
            d["embedding"] = vectors[i]
        _schema_embeddings = descriptions
    except Exception as e:
        log.warning("schema_embedding_failed", error=str(e))
        _schema_embeddings = descriptions

    return _schema_embeddings


def _semantic_search(query_embedding: list[float], top_k: int = 10) -> list[tuple[float, dict]]:
    """Find top-k most relevant schema descriptions by cosine similarity.

    Returns list of (score, item) tuples for RRF fusion.
    """
    schema_items = _get_schema_with_embeddings()

    if not query_embedding or not schema_items or "embedding" not in schema_items[0]:
        return [(1.0, item) for item in schema_items]

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return [(1.0, item) for item in schema_items]

    scored = []
    for item in schema_items:
        if "embedding" not in item:
            continue
        item_vec = np.array(item["embedding"], dtype=np.float32)
        item_norm = np.linalg.norm(item_vec)
        if item_norm == 0:
            continue
        similarity = float(np.dot(query_vec, item_vec) / (query_norm * item_norm))
        scored.append((similarity, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k * 2]  # Return more for RRF fusion


def _keyword_search(query: str, top_k: int = 10) -> list[tuple[float, dict]]:
    """Keyword-based schema search using token overlap scoring.

    Matches query tokens against table names, column names, descriptions, and domains.
    Returns list of (score, item) tuples for RRF fusion.
    """
    schema_items = _get_schema_with_embeddings()

    # Tokenize query - extract meaningful words
    query_tokens = set(re.findall(r'\b[a-z_]{2,}\b', query.lower()))
    # Add common synonyms/expansions
    synonym_map = {
        "product": {"product", "item", "goods", "sku"},
        "customer": {"customer", "client", "buyer"},
        "order": {"order", "purchase", "sale", "transaction"},
        "warehouse": {"warehouse", "storage", "facility", "location"},
        "inventory": {"inventory", "stock", "quantity"},
        "supplier": {"supplier", "vendor", "provider"},
        "price": {"price", "cost", "amount", "value"},
        "employee": {"employee", "staff", "worker"},
    }
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        for key, synonyms in synonym_map.items():
            if token in synonyms:
                expanded_tokens.update(synonyms)

    scored = []
    for item in schema_items:
        score = 0.0
        table = item.get("table_name", "").lower()
        column = item.get("column_name", "").lower() if item.get("column_name") else ""
        description = item.get("description", "").lower()
        domain = item.get("domain", "").lower()

        # Build item tokens
        item_text = f"{table} {column} {description} {domain}"
        item_tokens = set(re.findall(r'\b[a-z_]{2,}\b', item_text))

        # Token overlap scoring
        overlap = expanded_tokens & item_tokens
        if overlap:
            score = len(overlap) / max(len(expanded_tokens), 1)

            # Bonus for table name match (high signal)
            if table in expanded_tokens or any(t in table for t in expanded_tokens):
                score += 0.5

            # Bonus for column name match
            if column and (column in expanded_tokens or any(t in column for t in expanded_tokens)):
                score += 0.3

            # Bonus for domain match
            if domain in expanded_tokens:
                score += 0.2

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k * 2]


def _rrf_fusion(
    semantic_results: list[tuple[float, dict]],
    keyword_results: list[tuple[float, dict]],
    top_k: int = 10,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion (RRF) to combine semantic and keyword search results.

    RRF score = sum(1 / (k + rank_i)) for each ranking list.
    k=60 is the standard smoothing constant from the original RRF paper.
    """
    # Build RRF scores using item identity (table_name + column_name)
    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}

    def _item_key(item: dict) -> str:
        return f"{item.get('table_name', '')}::{item.get('column_name', '')}"

    # Score from semantic ranking
    for rank, (score, item) in enumerate(semantic_results):
        key = _item_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank + 1))
        item_map[key] = item

    # Score from keyword ranking
    for rank, (score, item) in enumerate(keyword_results):
        key = _item_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank + 1))
        item_map[key] = item

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    return [item_map[key] for key in sorted_keys[:top_k]]


@trace_agent_node("schema_retriever")
def schema_retriever_node(state: AgentState) -> dict:
    """Retrieve relevant database schema using hybrid search (semantic + keyword) with RRF.

    Combines vector similarity search with keyword matching using
    Reciprocal Rank Fusion for robust retrieval across query types.

    Output: schema_context, tables_used, schema_relationships
    """
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    query_embedding = state.get("query_embedding", [])
    messages = state.get("messages", [])

    # Get full DDL (always needed for SQL generation accuracy)
    schema_ddl = get_full_schema_ddl()

    # Hybrid search: semantic + keyword → RRF fusion
    top_k = settings.schema_top_k

    # Semantic search (uses pre-computed query embedding)
    if query_embedding:
        semantic_results = _semantic_search(query_embedding, top_k=top_k)
    else:
        semantic_results = []

    # Keyword search (always available, no embedding needed)
    keyword_results = _keyword_search(query, top_k=top_k)

    # RRF fusion of both result sets
    if semantic_results and keyword_results:
        relevant_descriptions = _rrf_fusion(semantic_results, keyword_results, top_k=top_k)
        log.info("schema_hybrid_rrf", semantic_count=len(semantic_results),
                 keyword_count=len(keyword_results), fused_count=len(relevant_descriptions))
    elif semantic_results:
        relevant_descriptions = [item for _, item in semantic_results[:top_k]]
    elif keyword_results:
        relevant_descriptions = [item for _, item in keyword_results[:top_k]]
    else:
        relevant_descriptions = get_schema_descriptions()

    # Build schema context with relevant descriptions
    desc_lines = []
    relevant_tables = set()
    for d in relevant_descriptions:
        table = d["table_name"]
        relevant_tables.add(table)
        col = d.get("column_name") or "(table)"
        desc_lines.append(f"  {table}.{col} ({d.get('domain', '')}): {d['description']}")

    # Extract relationships from DDL
    relationships = re.findall(r"(\w+\.\w+)\s*->\s*(\w+\.\w+)", schema_ddl)
    rel_lines = [f"{src} → {tgt}" for src, tgt in relationships]

    schema_context = (
        f"DATABASE SCHEMA (DDL):\n{schema_ddl}\n\n"
        f"RELEVANT DESCRIPTIONS (top-{top_k} by hybrid RRF search):\n" + "\n".join(desc_lines)
    )

    # Extract all table names from DDL
    all_tables = re.findall(r"TABLE (\w+) \(", schema_ddl)

    return {
        "messages": messages,
        "schema_context": schema_context,
        "tables_used": list(relevant_tables) if relevant_tables else all_tables,
        "schema_relationships": rel_lines,
    }
