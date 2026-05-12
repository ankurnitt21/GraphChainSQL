"""Semantic caching with Redis Vector Search (RediSearch) for similarity matching."""

import json
import hashlib
import time
import numpy as np
from decimal import Decimal
from typing import Optional
import redis
from src.core import get_settings
from src.core.resilience import redis_circuit, embedding_circuit, CircuitBreakerOpenError
import structlog

log = structlog.get_logger()
settings = get_settings()

_redis: Optional[redis.Redis] = None
_embeddings = None
_index_created = False

VECTOR_DIM = 1536  # OpenAI text-embedding-3-small dimension
INDEX_NAME = "idx:semantic_cache"
PREFIX = "graphchain:vec:"


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=False)
    return _redis


def _get_embeddings():
    """Get OpenAI embedding model for semantic similarity (lazy load)."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        except Exception as e:
            log.warning("embedding_model_unavailable", error=str(e))
            _embeddings = False
    return _embeddings if _embeddings is not False else None


def _ensure_index():
    """Create RediSearch vector index if not exists."""
    global _index_created
    if _index_created:
        return

    r = _get_redis()
    try:
        r.execute_command("FT.INFO", INDEX_NAME)
        _index_created = True
        return
    except redis.ResponseError:
        pass

    try:
        r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", PREFIX,
            "SCHEMA",
            "query", "TEXT",
            "result", "TEXT",
            "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32", "DIM", str(VECTOR_DIM), "DISTANCE_METRIC", "COSINE",
        )
        _index_created = True
        log.info("redis_vector_index_created", index=INDEX_NAME)
    except redis.ResponseError as e:
        if "Index already exists" in str(e):
            _index_created = True
        else:
            log.warning("redis_index_create_failed", error=str(e))


def _json_serializer(obj):
    """Handle Decimal and other non-standard types."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _hash_key(query: str) -> str:
    """Create a deterministic hash for exact match."""
    normalized = " ".join(query.lower().strip().split())
    return f"graphchain:exact:{hashlib.sha256(normalized.encode()).hexdigest()}"


def semantic_cache_get(query: str, precomputed_embedding: list[float] | None = None) -> Optional[dict]:
    """Try to find a cached query result.
    
    1. Exact hash match (O(1))
    2. Redis Vector Search KNN for semantic similarity (O(log n))
    3. Freshness validation (staleness check on cached_at)
    
    Args:
        query: The user query string
        precomputed_embedding: Optional pre-computed embedding to avoid duplicate embed calls
    """
    try:
        r = redis_circuit.call(_get_redis)
    except CircuitBreakerOpenError:
        log.warning("cache_get_circuit_open")
        return None

    # Exact match
    exact_key = _hash_key(query)
    try:
        exact = redis_circuit.call(r.get, exact_key)
        if exact:
            cached = json.loads(exact)
            if _is_fresh(cached):
                log.info("cache_hit_exact", query=query[:50])
                return cached
            else:
                log.info("cache_stale_exact", query=query[:50])
                return None
    except CircuitBreakerOpenError:
        return None
    except Exception as e:
        log.debug("cache_exact_get_error", error=str(e))

    # Semantic similarity via RediSearch KNN
    if precomputed_embedding and len(precomputed_embedding) > 0:
        query_emb = precomputed_embedding
    else:
        emb_model = _get_embeddings()
        if emb_model is None:
            return None
        try:
            query_emb = embedding_circuit.call(emb_model.embed_query, query)
        except Exception:
            return None

    try:
        _ensure_index()
        query_blob = np.array(query_emb, dtype=np.float32).tobytes()

        results = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            "*=>[KNN 1 @embedding $vec AS score]",
            "PARAMS", "2", "vec", query_blob,
            "SORTBY", "score",
            "RETURN", "2", "result", "score",
            "DIALECT", "2",
        )

        if results and results[0] > 0:
            fields = results[2]
            field_dict = {}
            for i in range(0, len(fields), 2):
                field_dict[fields[i].decode()] = fields[i + 1]

            score = float(field_dict.get("score", b"1.0"))
            similarity = 1.0 - score

            if similarity >= settings.semantic_cache_threshold:
                result_data = json.loads(field_dict["result"])
                if _is_fresh(result_data):
                    log.info("cache_hit_semantic", similarity=round(similarity, 3), query=query[:50])
                    return result_data
                else:
                    log.info("cache_stale_semantic", query=query[:50])
                    return None

    except CircuitBreakerOpenError:
        log.warning("cache_semantic_circuit_open")
    except Exception as e:
        log.debug("semantic_cache_search_error", error=str(e))

    return None


def _is_fresh(cached_entry: dict) -> bool:
    """Check if a cached entry is still fresh (not stale).

    Freshness rules:
    1. Basic TTL (handled by Redis expiry, but double-check)
    2. Table-specific freshness: queries on volatile tables expire faster
    3. Metadata age check: reject if older than freshness_threshold
    """
    metadata = cached_entry.get("metadata", {})
    cached_at = metadata.get("cached_at", 0)

    if cached_at == 0:
        # Legacy entry without timestamp - treat as fresh (TTL handles expiry)
        return True

    age_seconds = time.time() - cached_at

    # Base freshness threshold (same as TTL but acts as double-check)
    if age_seconds > settings.cache_ttl_seconds:
        return False

    # Table-specific freshness: volatile tables get shorter freshness
    sql = cached_entry.get("sql", "").upper()
    volatile_tables = {"inventory", "inventory_transaction", "sales_order", "sales_order_line", "shipment"}
    for table in volatile_tables:
        if table.upper() in sql:
            # Volatile tables: 60s freshness (vs 300s default)
            if age_seconds > 60:
                return False
            break

    return True


def invalidate_tables(table_names: list[str]):
    """Invalidate all cache entries that reference specific tables.

    Called when table data changes (e.g., after INSERT/UPDATE detected).
    Uses Redis SCAN to find and delete affected entries.
    """
    try:
        r = redis_circuit.call(_get_redis)
    except CircuitBreakerOpenError:
        return

    try:
        # Scan exact-match keys
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="graphchain:exact:*", count=100)
            for key in keys:
                try:
                    data = r.get(key)
                    if data:
                        entry = json.loads(data)
                        sql = entry.get("sql", "").upper()
                        if any(t.upper() in sql for t in table_names):
                            r.delete(key)
                except Exception:
                    pass
            if cursor == 0:
                break

        # Scan vector keys
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=f"{PREFIX}*", count=100)
            for key in keys:
                try:
                    result_data = r.hget(key, "result")
                    if result_data:
                        entry = json.loads(result_data)
                        sql = entry.get("sql", "").upper()
                        if any(t.upper() in sql for t in table_names):
                            r.delete(key)
                except Exception:
                    pass
            if cursor == 0:
                break

        log.info("cache_invalidated_tables", tables=table_names)
    except Exception as e:
        log.warning("cache_invalidation_error", error=str(e))


def semantic_cache_set(
    query: str,
    result: dict,
    precomputed_embedding: list[float] | None = None,
    metadata: dict | None = None,
):
    """Store query result with embedding for semantic matching.

    Quality-controlled cache write strategy:
    - Only caches if SQL was validated and execution was successful
    - Stores rich metadata (latency, confidence, canonical query)
    - Reuses precomputed embedding to avoid duplicate embed calls

    Args:
        query: The query string (raw or canonical)
        result: Must include {sql, explanation, results}
        precomputed_embedding: Optional pre-computed embedding vector
        metadata: Optional {latency_ms, confidence, canonical_query, validated, executed}
    """
    # Quality gate: only cache high-quality results
    meta = metadata or {}
    if not _should_cache(result, meta):
        log.info("cache_write_skipped_quality_gate", query=query[:50])
        return

    try:
        r = redis_circuit.call(_get_redis)
    except CircuitBreakerOpenError:
        log.warning("cache_set_circuit_open")
        return

    # Build enriched cache entry with metadata
    cache_entry = {
        "sql": result.get("sql", ""),
        "explanation": result.get("explanation", ""),
        "results": result.get("results", [])[:50],  # Cap stored results
        "metadata": {
            "raw_query": meta.get("raw_query", query),
            "canonical_query": meta.get("canonical_query", query),
            "confidence": meta.get("confidence", 0.0),
            "latency_ms": meta.get("latency_ms", 0),
            "cached_at": time.time(),
            "result_count": len(result.get("results", [])),
        },
    }

    # Always store exact match
    exact_key = _hash_key(query)
    try:
        redis_circuit.call(
            r.setex,
            exact_key,
            settings.cache_ttl_seconds,
            json.dumps(cache_entry, default=_json_serializer),
        )
    except Exception as e:
        log.warning("cache_exact_write_failed", error=str(e))

    # Store as vector for semantic search
    # Reuse precomputed embedding to avoid duplicate embed call
    if precomputed_embedding and len(precomputed_embedding) > 0:
        query_emb = precomputed_embedding
    else:
        emb_model = _get_embeddings()
        if emb_model is None:
            return
        try:
            query_emb = embedding_circuit.call(emb_model.embed_query, query)
        except Exception:
            return

    try:
        _ensure_index()
        emb_blob = np.array(query_emb, dtype=np.float32).tobytes()

        normalized = " ".join(query.lower().strip().split())
        vec_key = f"{PREFIX}{hashlib.sha256(normalized.encode()).hexdigest()}"

        redis_circuit.call(
            r.hset,
            vec_key,
            mapping={
                "query": query,
                "result": json.dumps(cache_entry, default=_json_serializer),
                "embedding": emb_blob,
            },
        )
        redis_circuit.call(r.expire, vec_key, settings.cache_ttl_seconds)
    except CircuitBreakerOpenError:
        log.warning("cache_vector_write_circuit_open")
    except Exception as e:
        log.warning("semantic_cache_write_failed", error=str(e))


def _should_cache(result: dict, metadata: dict) -> bool:
    """Quality control gate for cache writes.

    Only cache if:
      - SQL is present and non-empty
      - Execution was successful (has results or explicit success flag)
      - No ambiguity occurred
      - Result size is reasonable (not empty, not too large)
      - Confidence meets minimum threshold (if provided)
    """
    sql = result.get("sql", "")
    results = result.get("results", [])
    explanation = result.get("explanation", "")

    # Must have SQL
    if not sql or not sql.strip():
        return False

    # Must have either results or explanation
    if not results and not explanation:
        return False

    # Skip if explicitly marked as unvalidated
    if metadata.get("validated") is False:
        return False

    # Skip if execution failed
    if metadata.get("executed") is False:
        return False

    # Skip if ambiguity was flagged
    if metadata.get("ambiguous"):
        return False

    # Skip if result count is unreasonable (too large = likely accidental full table scan)
    if len(results) > 500:
        return False

    # Skip if confidence is too low
    confidence = metadata.get("confidence", 1.0)
    if confidence < 0.4:
        return False

    return True
