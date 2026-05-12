# GraphChainSQL (v8.0)

A **production-grade dual-pipeline multi-agent system** built with **LangGraph** (Python), featuring:

- **Parallel Phase 1** — Intent + Memory + Cache L1 + Embedding run concurrently (4-way ThreadPoolExecutor). Intent detection no longer blocks the pipeline.
- **Parallel Phase 4** — Schema Retrieval + Complexity Detection run concurrently after ambiguity resolution.
- **OpenAI GPT-4o-mini** — All LLM calls (SQL generation, ambiguity, intent, response, guardrails)
- **OpenAI text-embedding-3-small** — 1536-dim embeddings for semantic cache & schema retrieval
- **Pinecone Vector Store** — `rag-base` (domain knowledge, 3 namespaces) + `sql-base` (schema metadata)
- **ReAct Action Pipeline** — LLM reasons + acts in a loop using warehouse tools
- **Human-in-the-Loop** — every tool call requires approval BEFORE execution (`interrupt()`)
- **5-layer SQL validation** + circuit breaker + rate limiter
- **OpenTelemetry → LangSmith** — all agent spans nested under a single `sql_pipeline` root span, including spans created in parallel threads (explicit OTEL context propagation)
- **PostgresSaver** — distributed checkpointing (horizontal scaling ready)
- **Zero hardcoded prompts** — all prompts stored in PostgreSQL with versioning

---

## Quick Start

```bash
# 1. Start infrastructure
docker compose -f docker/docker-compose.yml up -d   # PostgreSQL :5433 · Redis :6379 · pgAdmin :5050

# 2. Install dependencies
pip install -r requirements.txt

# 3. Bootstrap Pinecone (flush old indexes, create domain docs + schema metadata)
python setup_pinecone.py

# 4. Run (port and tier come from .env — see .env.example)
python run.py   # default http://localhost:${API_PORT:-8085}
```

### Configuration and observability

All tier and bind settings are **environment-driven** (`APP_ENV`, `SERVICE_NAME`, `API_PORT`, `RAGAS_COLLECT_ON_COMPLETE`, etc.). There is no tier logic hardcoded in Python.

**Tiers:** set **`APP_ENV`** to `dev` | `test` | `stg` | `prod`. It is sent on the OTLP resource as `deployment.environment` for LangSmith filters.

**Service name:** **`SERVICE_NAME`** (or **`OTEL_SERVICE_NAME`**) becomes OTLP `service.name` and the `/health` payload.

**LangSmith (OTLP traces):** `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `OTEL_EXPORTER_OTLP_ENDPOINT` (default `https://api.smith.langchain.com/otel`). Spans are built in `src/core/tracing.py`.

| Span area | Attributes (LangSmith / OTLP) |
|-----------|-------------------------------|
| **Inputs / Outputs in LangSmith UI** | OpenInference mapping: **`input.value`** and **`output.value`** hold **JSON** (query, cache flags `cache_hit` / `l1_checked` / `l2_hit`, schema hints, SQL, summaries). GenAI message mapping: **`gen_ai.prompt.0.*`** / **`gen_ai.completion.0.*`** for readable chat-style rows. |
| Root `sql_pipeline` | Same as above + `gen_ai.system`, `gen_ai.operation.name`, `timing.pipeline_ms`, `session_id` |
| Each `agent.*` | Same as above + `app.*` mirrors, **`langsmith.metadata.prompt_name`** / **`prompt_version`** when a DB template is used, `gen_ai.request.model`, `latency_ms` |

**Optional RAGAS persistence:** `RAGAS_COLLECT_ON_COMPLETE` (default `false`). When `true`, completed non-cache answers enqueue a background eval; rows land in **`ragas_eval_result`**. List with **`GET /api/ragas/recent`**. The app also creates this table on startup if missing.

**Simulate prod locally (PowerShell example):**

```powershell
$env:APP_ENV = "prod"; $env:SERVICE_NAME = "graphchainsql-api"; python run.py
```

**Required environment variables** (already in `.env`):

```env
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=...
PINECONE_RAG_INDEX=rag-base
PINECONE_SQL_INDEX=sql-base
LANGSMITH_API_KEY=...
```

---

## Two Pipelines

### Read Pipeline (SQL DAG)

```
POST /api/query
  │
  └─► parallel_init  ←── 4-way concurrent ──────────────────────────────────┐
        ├─ intent_detector (LLM)                                             │
        ├─ memory_agent (DB)                                                 │
        ├─ cache_l1 (Redis)                                                  │
        └─ embedding_agent (OpenAI)                                          │
        + keyword complexity classifier (0 ms, no LLM)                      │
                │                                                            │
       ┌────────┼──────────────────────────────┐                            │
       │        │                              │                            │
  action      keyword=simple           moderate/complex                     │
       │        │                              │                            │
       ▼        ▼                              ▼                            │
  react_agent  cache_l2              ambiguity_agent (LLM)                  │
  (loop)          │                            │                            │
                  └─────────────┬──────────────┘                            │
                                │                                            │
                             cache_l2                                        │
                                │                                            │
                   ┌────────────┴─────────────┐                             │
                   │ cache hit                │ miss                        │
                   ▼                          ▼                              │
            respond_from_cache        schema_retriever (DB) ───────────────┘
                                               │
                                        sql_generator (LLM)
                                               │
                                        sql_validator (no LLM)
                                               │
                                       approval_agent
                                               │
                                        sql_executor (DB)
                                               │
                                     response_synthesizer
                                (template if simple, LLM if complex)
```

### Action Pipeline (ReAct Agent)

```
parallel_init → intent=="action" → react_agent loop:
  [LLM thinks → interrupt(HITL) → user approves → execute tool → repeat]
```

---

## Latency Profile (cold query, no cache)

| Phase | What runs | Saving |
|-------|-----------|--------|
| Phase 1 | intent + memory + cache + embed **in parallel** | ~1–5 s (intent was serial before) |
| Phase 1 | keyword complexity classifier (0 ms) replaces LLM complexity_detector | ~1.7 s |
| Phase 1 | keyword=simple → skip ambiguity LLM entirely | ~10 s |
| Phase 7 | template for simple/ranked results — **no LLM** | ~5 s |
| Guard | PII LLM call skipped for simple/moderate queries | ~2.5 s |

Typical "top N" / simple keyword query: **~11 s** cold (was ~14 s with complexity_detector, ~25 s without routing). **< 1 s** on cache hit.

### Complexity routing (no LLM)

| Keyword matches | complexity | Ambiguity? | Response |
|-----------------|------------|------------|---------|
| top, list, show, count, sum, total… | `simple` | Skip (→ cache_l2 directly) | Template (0 ms) |
| trend, forecast, compare, pivot, percentile… | `complex` | Always runs | LLM + full PII guard |
| (everything else) | `moderate` | Always runs | LLM + regex PII |

---

## Action Tools

| Tool | Operation | Args |
|------|-----------|------|
| `create_po` | INSERT purchase_order | product_id, qty, warehouse_id? |
| `notify_supplier` | HTTP POST (simulated) | supplier_id, message? |
| `update_shipment` | UPDATE shipment | shipment_id, status |
| `call_erp_sync` | Microservice call (sim) | order_ids[], sync_type? |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | Run query (auto-detects read vs action) |
| POST | `/api/query/stream` | SSE stream of pipeline steps |
| POST | `/api/approve` | Resume SQL approval (read pipeline) |
| POST | `/api/action/approve` | Resume tool approval (action pipeline) |
| POST | `/api/feedback` | Submit thumbs up/down on query result |
| GET | `/api/feedback/stats` | Feedback approval rate metrics |
| GET | `/api/feedback/negative` | Negative feedback for prompt improvement |
| GET | `/api/ragas/recent` | Recent RAGAS scores from Postgres (when collection is enabled) |
| GET | `/api/action/tools` | List action tools |
| GET | `/api/prompts` | List all DB-stored prompt templates |
| POST | `/api/clarify` | Submit clarification |
| GET | `/health` | Health check |

---

## Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5433 | Warehouse data, conversation history, prompts, checkpoints |
| Redis (redis-stack) | 6379 | Semantic cache (RediSearch vector index) |
| Redis Insight | 8001 | Redis UI |
| pgAdmin | 5050 | Web UI — login `admin@admin.com` / `admin`; server **GraphChainSQL Postgres** is pre-registered (connects to the `postgres` service on port 5432 inside Docker). |

---

## RAGAS (optional collection)

When **`RAGAS_COLLECT_ON_COMPLETE=true`**, completed non-cache queries trigger a **background** RAGAS evaluation; scores are stored in **`ragas_eval_result`** for dashboards, regression analysis, and future quality gates. Default is `false` to avoid surprise LLM cost. Details are in **Configuration and observability** above.

---

## Pinecone Indexes

| Index | Namespace | Dimension | Content |
|-------|-----------|-----------|---------|
| `rag-base` | `inventory` | 1536 | Inventory domain knowledge (5 chunks) |
| `rag-base` | `sales` | 1536 | Sales & fulfillment knowledge (4 chunks) |
| `rag-base` | `procurement` | 1536 | Procurement & supplier knowledge (4 chunks) |
| `sql-base` | `schema` | 1536 | Schema descriptions with rich metadata (29 vectors) |

Embeddings: **OpenAI text-embedding-3-small** (1536 dim, cosine metric).

To re-bootstrap: `python setup_pinecone.py` (flushes existing → recreates).

---

## Feedback Loop

Every query response includes a `run_id` (LangSmith trace ID):

```bash
curl -X POST http://localhost:8085/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","query":"top products","rating":1,"run_id":"abc123"}'
```

Feedback is stored in `query_feedback` table and synced to LangSmith.

---

## LangSmith observability (OpenTelemetry)

Spans export over OTLP with structured attributes: **`app.input.primary` / `app.input.sub`**, **`app.output.primary` / `app.output.sub`**, **`app.prompt.name` / `app.prompt.version` / `app.prompt.body_excerpt`** (for DB-backed prompts), **`gen_ai.request.model`**, and **`latency_ms`** / **`timing.*`**. All spans are nested under a single `sql_pipeline` root span:

```
sql_pipeline  (root)
  ├─ agent.parallel_init          ← 4-way concurrent + 0 ms keyword complexity
  │    ├─ agent.intent_detector   ← thread, OTEL context propagated
  │    ├─ agent.memory_agent      ← thread, OTEL context propagated
  │    ├─ agent.cache_l1          ← thread, OTEL context propagated
  │    └─ agent.embedding_agent   ← thread, OTEL context propagated
  ├─ agent.ambiguity_agent        ← only for moderate/complex (skipped for simple)
  ├─ agent.cache_l2
  ├─ agent.schema_retriever       ← DB only, no LLM
  ├─ agent.sql_generator          ← LLM always
  ├─ agent.sql_validator
  ├─ agent.approval_agent
  ├─ agent.sql_executor
  └─ agent.response_synthesizer   ← template for simple, LLM for complex only
```
