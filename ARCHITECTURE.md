# Architecture & Components Reference (v8.0)

Phase-based DAG Orchestrator | **4-Way Parallel Phase 1** | **Parallel Phase 4 (Schema + Complexity)** | **ReAct Action Pipeline** | **Human-in-the-Loop Tool Approval** | Adaptive Complexity Routing | Relevance-Filtered Memory | Dual-Layer Cache | Confidence Scoring | Circuit Breaker + Rate Limiter | Hybrid Schema Search (RRF) | **PostgresSaver Checkpointing** | **User Feedback → LangSmith** | **DB-Stored Prompts (zero hardcoding)** | **Pinecone Vector Store** | **Env-driven tiers + OTEL/LangSmith + optional RAGAS persistence**

**Operations:** Tier labels (`APP_ENV`), HTTP bind (`API_HOST` / `API_PORT`), trace export (`OTEL_EXPORTER_OTLP_ENDPOINT`), and optional RAGAS collection are configured via **`Settings`** (see `src/core/__init__.py`). See **Configuration and observability** in [README.md](README.md).

---

## LLM & Embedding Stack

| Component | Model | Dimension | Purpose |
|-----------|-------|-----------|---------|
| Chat LLM | `gpt-4o-mini` (OpenAI) | — | SQL gen, ambiguity, intent, response (complex only), guardrails (complex only) |
| Embeddings | `text-embedding-3-small` (OpenAI) | 1536 | Semantic cache (Redis) + schema retrieval |
| Vector Store | Pinecone Serverless (AWS us-east-1) | 1536 | Domain RAG (`rag-base`) + schema meta (`sql-base`) |

---

## Dual Pipeline Architecture

```
POST /api/query
      │
      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 1: parallel_init  (4-way ThreadPoolExecutor)                      │
│                                                                          │
│   ┌──────────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐      │
│   │ intent_detector  │  │memory_agent│  │cache_l1  │  │embedding│      │
│   │   (OpenAI LLM)   │  │  (DB)      │  │ (Redis)  │  │(OpenAI) │      │
│   └──────────────────┘  └────────────┘  └──────────┘  └─────────┘      │
│                                                                          │
│   + _fast_complexity(query)  — keyword heuristic, 0 ms, no LLM          │
│     sets query_complexity in state for routing & response decisions      │
│                                                                          │
│   OTEL context propagated into threads → spans nest in LangSmith        │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ├── intent == "action" ─────────────────────► ReAct Agent LOOP ──► END
         ├── L1 cache hit ─────────────────────────── ► respond_from_cache ► END
         ├── query_complexity == "simple" ───────────► cache_l2 (skip ambiguity)
         │
         └── moderate / complex ──► ambiguity_agent (LLM)
                                            │
                                         cache_l2
                                            │
                              ├── L2 hit ──► respond_from_cache ──► END
                              │
                              ▼
                       schema_retriever  (Hybrid RRF: semantic + keyword)
                              │
                       sql_generator    (OpenAI tool-calling)
                              │
                       sql_validator    (5 layers, no LLM, retry ×3)
                              │
                       approval_agent   (auto or HITL pause)
                              │
                       sql_executor     (PostgreSQL)
                              │
                       response_synthesizer
                         keyword=simple → template (0 ms)
                         moderate/complex → LLM explanation
                              │
                             END
```

---

## File Structure

| File | Role |
|------|------|
| `src/agents/pipeline.py` | Graph builder: parallel_init (4-way) + schema_and_complexity (2-way) + conditional routing |
| `src/agents/intent_detector.py` | Classifies "read" or "action" — runs inside parallel_init |
| `src/agents/schema_agent.py` | Hybrid schema retrieval (semantic + keyword, RRF fusion) |
| `src/agents/embedding_agent.py` | OpenAI text-embedding-3-small — runs inside parallel_init |
| `src/agents/sql_generator_agent.py` | SQL generation via OpenAI tool-calling + self-consistency |
| `src/agents/react_agent.py` | ReAct loop: think → interrupt(HITL) → execute |
| `src/agents/action_tools.py` | Tool registry: create_po, notify_supplier, update_shipment, call_erp_sync |
| `src/core/tracing.py` | OTEL → LangSmith; `run_in_context()` for thread span propagation; span attrs: `app.input.*`, `app.output.*`, `app.prompt.*`, `gen_ai.request.model`, `latency_ms` |
| `src/services/ragas_service.py` | Optional RAGAS eval + `ragas_eval_result` persistence; `GET /api/ragas/recent` |
| `src/core/resilience.py` | CircuitBreaker + RateLimiter (openai_llm, redis, database, embedding) |
| `src/core/prompts.py` | DB-backed prompt management with versioning |
| `src/services/cache.py` | Redis semantic cache (RediSearch KNN, 1536-dim OpenAI embeddings) |
| `src/services/guardrails_service.py` | Input/output validation using OpenAI LLM |
| `setup_pinecone.py` | One-time bootstrap: flush → recreate Pinecone indexes → seed domain docs + schema |
| `data/domains/` | Domain knowledge TXT files (inventory, sales, procurement) |
| `docker/docker-compose.yml` | PostgreSQL :5433 + Redis-Stack :6379 + pgAdmin :5050 (pre-registered server + pgpass) |

---

## Pinecone Architecture

```
setup_pinecone.py
  │
  ├─► rag-base (dim=1536, cosine)
  │     ├─ namespace: "inventory"    (5 vectors — inventory domain doc chunks)
  │     ├─ namespace: "sales"        (4 vectors — sales domain doc chunks)
  │     └─ namespace: "procurement"  (4 vectors — procurement domain doc chunks)
  │
  └─► sql-base (dim=1536, cosine)
        └─ namespace: "schema"       (29 vectors — PostgreSQL schema_description rows)
               metadata: {table_name, column_name, domain, description, data_type, text}
```

Domain docs live in `data/domains/*.txt`. Each file → chunked (500 chars, 100 overlap) → embedded → upserted under the domain namespace. Re-run `setup_pinecone.py` to refresh.

---

## LangSmith Span Hierarchy

```
sql_pipeline  (root span, created in routes.py before graph.invoke)
  │
  ├─ agent.parallel_init                 (Phase 1 — 4-way + 0ms keyword complexity)
  │    ├─ agent.intent_detector          (thread — OTEL context propagated via run_in_context)
  │    ├─ agent.memory_agent             (thread — OTEL context propagated via run_in_context)
  │    ├─ agent.cache_l1                 (thread — OTEL context propagated via run_in_context)
  │    └─ agent.embedding_agent          (thread — OTEL context propagated via run_in_context)
  │
  ├─ agent.ambiguity_agent               (only for moderate/complex — skipped for "simple")
  ├─ agent.cache_l2
  ├─ agent.schema_retriever              (DB only, no LLM)
  ├─ agent.sql_generator                 (LLM — always)
  ├─ agent.sql_validator                 (no LLM)
  ├─ agent.approval_agent
  ├─ agent.sql_executor                  (DB)
  └─ agent.response_synthesizer          (template for simple, LLM for complex only)
```

Root span opened in `routes.py::query()` before `graph.invoke()`. Each `@trace_agent_node` fetches parent context via `_get_parent_context(state)`. Parallel threads use `run_in_context(fn, state)` to explicitly `otel_context.attach(parent_ctx)` so all child spans nest under the root in LangSmith.

---

## Action Tools

| Tool | DB Operation | Args |
|------|-------------|------|
| `create_po` | INSERT purchase_order + purchase_order_line | product_id, qty, warehouse_id? |
| `notify_supplier` | Simulated HTTP POST (logged) | supplier_id, message? |
| `update_shipment` | UPDATE shipment SET status | shipment_id, status |
| `call_erp_sync` | Simulated microservice call | order_ids[], sync_type? |

---

## Human-in-the-Loop (HITL)

Two HITL gates:

1. **SQL Approval** (`require_approval=True`) — pauses before executing a SELECT.
   - Endpoint: `POST /api/approve`

2. **Tool Approval** (always on for action pipeline) — pauses before EACH tool.
   - Endpoint: `POST /api/action/approve`
   - Graph resumes via `Command(resume={approved, feedback})`

Both use LangGraph `interrupt()` + `PostgresSaver` checkpointing → survives restarts.

---

## Response Strategy

| Result type | Path | LLM call? | PII guard |
|-------------|------|-----------|-----------|
| Empty result | Template | No | Regex only |
| Single-row aggregate (COUNT, SUM) | Template | No | Regex only |
| `complexity=simple` AND ≤ 10 rows | Template | No | Regex only |
| Up to 3 single-column rows | Template | No | Regex only |
| `complexity=moderate` multi-row | LLM | Yes | Regex only |
| `complexity=complex` multi-row | LLM | Yes | Full LLM PII guard |

Template responses render a compact numbered table (no LLM call, ~0 ms).
Full LLM PII guard (guardrails-ai) only fires for `complex` queries where free-text
explanations with potentially sensitive data are more likely.

---

## Resilience Configuration

| Instance | Name | Failure Threshold | Recovery | Purpose |
|----------|------|------------------|----------|---------|
| `llm_circuit` | openai_llm | 5 failures | 30 s | OpenAI API calls |
| `llm_rate_limiter` | openai_llm | 100 burst tokens | 80/s refill | Match GPT-4o-mini tier limits |
| `redis_circuit` | redis | 3 failures | 10 s | Redis cache operations |
| `db_circuit` | database | 3 failures | 15 s | PostgreSQL queries |
| `embedding_circuit` | embedding | 3 failures | 20 s | OpenAI embed calls |

---

## Prompt Management

All 9 system prompts stored in `prompt_template` (PostgreSQL):

| Prompt Name | Used By | Purpose |
|------------|---------|---------|
| `intent_detection` | intent_detector.py | Classify read vs action |
| `complexity_detection` | pipeline.py | Classify query complexity |
| `sql_generation` | sql_generator_agent.py | Generate SQL from schema |
| `sql_self_consistency` | sql_generator_agent.py | Verify SQL matches intent |
| `ambiguity_resolution` | ambiguity_agent.py | Detect/resolve ambiguity |
| `memory_summarization` | memory_agent.py | Summarize conversation overflow |
| `response_synthesis` | response_agent.py | Explain results to user |
| `response_system` | response_agent.py | System message for response |
| `react_system` | react_agent.py | ReAct agent instructions |

Hot-swappable: update the DB row, no redeployment needed.

---

## Checkpointing & Scaling

| Component | Implementation |
|-----------|---------------|
| Checkpointer | `PostgresSaver` (langgraph-checkpoint-postgres) |
| State persistence | Survives crashes, supports horizontal scaling |
| HITL resume | Any instance can resume an interrupted graph |
| Session memory | Conversation history across requests |
