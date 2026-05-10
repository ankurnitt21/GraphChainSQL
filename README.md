# GraphChainSQLPython (v7.0)

A **production-grade dual-pipeline multi-agent system** built with **LangGraph** (Python), featuring:

- **Intent Detection** — automatically routes "read" queries to the SQL DAG or "action" commands to the ReAct agent
- **ReAct Action Pipeline** — LLM reasons + acts in a loop using warehouse tools (create_po, notify_supplier, update_shipment, call_erp_sync)
- **Human-in-the-Loop for tools** — every tool call requires user approval BEFORE execution (via `interrupt()`)
- **Adaptive SQL pipeline** (complexity routing, dual-layer cache, ambiguity resolution)
- **Parallel Phase 1** (Memory + Cache L1 + Embedding run concurrently)
- **5-layer SQL validation** + circuit breaker + rate limiter
- **OpenTelemetry → LangSmith** observability
- **PostgresSaver** — distributed checkpointing (horizontal scaling ready)
- **User Feedback Loop** — thumbs up/down synced to LangSmith traces
- **Zero hardcoded prompts** — all prompts stored in PostgreSQL with versioning

---

## Quick Start

```bash
cd GraphChainSQLPython
docker-compose -f docker/docker-compose.yml up -d   # PostgreSQL + Redis
pip install -r requirements.txt
cp .env.example .env                                 # add GROQ_API_KEY
python main.py                                       # http://localhost:8085
```

---

## Two Pipelines

### Read Pipeline (SQL DAG)

Ask natural language questions about warehouse data:

```
"Show top 5 products by unit price"
"What is inventory by zone?"
"Which suppliers have the best rating?"
```

Flow: `Intent → Parallel Init → Complexity → Ambiguity → Cache L2 → Schema → SQL Gen → Validate → [Approve] → Execute → Response`

### Action Pipeline (ReAct Agent)

Give action commands that mutate data or trigger external calls:

```
"Create a purchase order for product 1, quantity 50"
"Notify supplier 3 about the delay"
"Update shipment 5 to SHIPPED"
"Sync orders 1,2,3 with ERP, then create a PO for product 2"
```

Flow: `Intent → ReAct Loop: [LLM thinks → HITL interrupt → Execute tool → loop]`

---

## Action Tools

| Tool                           | Operation               | Example                 |
| ------------------------------ | ----------------------- | ----------------------- |
| `create_po(product_id, qty)`   | INSERT purchase_order   | Creates PO + line item  |
| `notify_supplier(supplier_id)` | HTTP POST (simulated)   | Sends notification      |
| `update_shipment(id, status)`  | UPDATE shipment         | Updates status          |
| `call_erp_sync(order_ids)`     | Microservice call (sim) | Triggers ERP sync batch |

---

## Human-in-the-Loop

**Tool Approval** (action pipeline, always on):

- Before each tool executes, the graph pauses with `interrupt()`
- API returns `status: "awaiting_tool_approval"` with `pending_tool_call`
- UI shows: tool name, arguments, LLM reasoning + approve/reject buttons
- User posts `POST /api/action/approve` → graph resumes

**SQL Approval** (read pipeline, opt-in via `require_approval=True`):

- Before executing SQL, graph pauses with `interrupt()`
- API returns `status: "awaiting_approval"` with generated SQL
- User posts `POST /api/approve` → graph resumes

---

## API Reference

| Method | Path                     | Description                             |
| ------ | ------------------------ | --------------------------------------- |
| POST   | `/api/query`             | Run query (auto-detects read vs action) |
| POST   | `/api/query/stream`      | SSE stream of pipeline steps            |
| POST   | `/api/approve`           | Resume SQL approval (read pipeline)     |
| POST   | `/api/action/approve`    | Resume tool approval (action pipeline)  |
| POST   | `/api/feedback`          | Submit thumbs up/down on query result   |
| GET    | `/api/feedback/stats`    | Get feedback approval rate metrics      |
| GET    | `/api/feedback/negative` | List negative feedback for analysis     |
| GET    | `/api/action/tools`      | List action tools                       |
| GET    | `/api/prompts`           | List all DB-stored prompt templates     |
| POST   | `/api/clarify`           | Submit clarification                    |
| GET    | `/health`                | Health check                            |

---

## Project Structure

```
src/
  agents/
    intent_detector.py   ← Route read vs action
    react_agent.py       ← ReAct loop with HITL
    action_tools.py      ← create_po, notify_supplier, update_shipment, call_erp_sync
    pipeline.py          ← LangGraph DAG (both pipelines) + PostgresSaver
    sql_generator_agent.py
    ... (other SQL pipeline agents)
  api/
    routes.py            ← FastAPI endpoints (query, feedback, approve)
    models.py            ← Pydantic models (incl. FeedbackRequest)
  core/
    state.py             ← AgentState (includes intent, react_steps, etc.)
    prompts.py           ← DB-backed prompt management (no hardcoding)
    database.py
    tracing.py           ← OTEL → LangSmith with run_id for feedback
    resilience.py        ← Circuit breaker + rate limiter
  services/
    feedback.py          ← Feedback collection + LangSmith sync
    cache.py             ← Dual-layer semantic caching
    guardrails_service.py
static/
  index.html             ← Single-page UI with dual pipeline support
docker/
  docker-compose.yml     ← PostgreSQL + Redis
```

---

## Environment Variables

```
GROQ_API_KEY=...
DATABASE_URL_SYNC=postgresql://warehouse_admin:warehouse_secret_2024@localhost:5433/warehouse_db
REDIS_URL=redis://localhost:6379
LANGSMITH_API_KEY=...   (for tracing + feedback sync)
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=GraphChainSQL
```

---

## Feedback Loop

Every query response includes a `run_id` (LangSmith trace ID). The UI can submit feedback:

```bash
# Thumbs up
curl -X POST http://localhost:8000/api/feedback -H 'Content-Type: application/json' \
  -d '{"session_id": "...", "query": "...", "rating": 1, "run_id": "..."}'

# Thumbs down with correction
curl -X POST http://localhost:8000/api/feedback -H 'Content-Type: application/json' \
  -d '{"session_id": "...", "query": "...", "rating": -1, "comment": "wrong table", "correction": "SELECT ...", "run_id": "..."}'
```

Feedback is:

1. Stored in `query_feedback` table (for local analytics)
2. Synced to LangSmith (visible on the trace in the UI)
3. Retrievable via `GET /api/feedback/negative` for prompt improvement

---

## Checkpointing (PostgresSaver)

State is persisted to PostgreSQL via `langgraph-checkpoint-postgres`. This enables:

- **Horizontal scaling** — multiple app instances share checkpoints
- **Crash recovery** — interrupted HITL flows survive restarts
- **Session persistence** — conversation memory across requests

2. PDF Ingestion Pipeline
PDF uploaded → stored in S3
Kafka sends lightweight EVENT (not PDF itself)
Python service consumes event
Downloads PDF from S3
Chunks → Embeds → Upserts to Pinecone

Kafka only carries metadata, not raw PDF bytes
Kafka has 1MB default limit


3. Deduplication

Generate MD5 hash of PDF on download
Check PostgreSQL before ingesting
If hash exists → skip
If new → ingest and store hash


4. PDF Versioning (hash changed)

Same filename + different hash = updated PDF
Store chunk IDs in PostgreSQL alongside hash
On update:

Delete old chunks from Pinecone using stored chunk IDs
Re-ingest fresh chunks
Update version record in PostgreSQL

5. Granular vs Full Chunk Replacement

Replacing only changed chunks sounds efficient but chunking is non-deterministic
Can't reliably map old chunks to new chunks
Delete all + re-ingest is safer and simpler
Cost is negligible (~0.2 cents per typical PDF)
Granular replacement only works for structured PDFs with fixed section boundaries

Use pdfplumber for pdf with tables
<img width="890" height="595" alt="image" src="https://github.com/user-attachments/assets/ed17d5f0-efad-4a1a-bc0a-d8becf69313c" />

