# Architecture & Components Reference (v7.0)

Phase-based DAG Orchestrator | **Intent Detection** | **ReAct Action Pipeline** | **Human-in-the-Loop Tool Approval** | Adaptive Complexity Routing | Relevance-Filtered Memory | Dual-Layer Cache | Confidence Scoring | Circuit Breaker + Rate Limiter | Hybrid Schema Search (RRF) | **PostgresSaver Checkpointing** | **User Feedback → LangSmith** | **DB-Stored Prompts (zero hardcoding)**

---

## Dual Pipeline Architecture

```
POST /api/query
      |
      v
[INTENT DETECTOR]  ← keyword fast path + LLM fallback
      |
      +── "read"   ──────────────────────────────────────────────────────►
      |                                                                    |
      |   [Phase 1: PARALLEL]                                             |
      |     Memory Agent | Cache L1 | Embedding Agent                    |
      |         |                                                         |
      |   [Complexity Detector] → simple/moderate/complex               |
      |         |                                                         |
      |   [Phase 2: Ambiguity] → rewrite or clarify                      |
      |         |                                                         |
      |   [Phase 3: Cache L2] → hit or miss                              |
      |         |                                                         |
      |   [Phase 4: Schema → SQL Gen → SQL Validate]                    |
      |         |                                                         |
      |   [Phase 5: HITL Approval] ← require_approval=true pauses       |
      |         |                                                         |
      |   [Phase 6: SQL Execute → Response]                             ◄
      |
      +── "action" ─────────────────────────────────────────────────────►
                                                                         |
          [ReAct Agent LOOP]  ← max 5 iterations                        |
            |                                                            |
            LLM THINKS: which tool? with what args? why?                |
            |                                                            |
            interrupt() ─── POST /api/action/approve ─── human approves|
            |                    or rejects                              |
            EXECUTE TOOL:                                                |
              create_po(product_id, qty)       → DB INSERT (purchase_order)
              notify_supplier(supplier_id)     → HTTP POST [simulated]  |
              update_shipment(id, status)      → DB UPDATE (shipment)   |
              call_erp_sync(order_ids)         → microservice [sim]    |
            |                                                            |
            loop back → LLM sees result → decides next tool or "done"  ◄
```

---

## File Structure

| File                            | Role                                                                      |
| ------------------------------- | ------------------------------------------------------------------------- |
| `src/agents/intent_detector.py` | Classifies query as "read" or "action" (keyword + LLM)                    |
| `src/agents/action_tools.py`    | Tool registry: create_po, notify_supplier, update_shipment, call_erp_sync |
| `src/agents/react_agent.py`     | ReAct loop node: think → interrupt(HITL) → execute                        |
| `src/agents/pipeline.py`        | Graph builder with PostgresSaver + intent routing + ReAct loop edges      |
| `src/api/routes.py`             | POST /api/query, /api/feedback, /api/action/approve, /api/approve         |
| `src/api/models.py`             | QueryResponse + FeedbackRequest + ActionApproveRequest models             |
| `src/core/state.py`             | AgentState with intent, react_steps, react_result, pending_tool_call      |
| `src/core/prompts.py`           | DB-backed prompt management with versioning (zero hardcoded prompts)      |
| `src/services/feedback.py`      | Feedback collection, DB storage, LangSmith sync                           |

---

## Action Tools

| Tool              | DB Operation                                | Args                           |
| ----------------- | ------------------------------------------- | ------------------------------ |
| `create_po`       | INSERT purchase_order + purchase_order_line | product_id, qty, warehouse_id? |
| `notify_supplier` | Simulated HTTP POST (logged)                | supplier_id, message?          |
| `update_shipment` | UPDATE shipment SET status                  | shipment_id, status            |
| `call_erp_sync`   | Simulated microservice call                 | order_ids[], sync_type?        |

---

## Human-in-the-Loop (HITL)

Two HITL gates in the system:

1. **SQL Approval** (`require_approval=True`) — Before executing a SELECT query.
   - Endpoint: `POST /api/approve`
   - Shows: generated SQL, confidence, NL explanation

2. **Tool Approval** (always on for action pipeline) — Before executing EACH tool.
   - Endpoint: `POST /api/action/approve`
   - Shows: tool name, arguments, LLM reasoning
   - User can provide rejection feedback
   - Graph resumes with `Command(resume={approved, feedback})`

---

## API Endpoints

| Method | Path                   | Description                                  |
| ------ | ---------------------- | -------------------------------------------- |
| POST   | /api/query             | Run query (read or action pipeline)          |
| POST   | /api/query/stream      | SSE stream of pipeline steps                 |
| POST   | /api/approve           | Resume after SQL approval (read pipeline)    |
| POST   | /api/action/approve    | Resume after tool approval (action pipeline) |
| POST   | /api/feedback          | Submit thumbs up/down (syncs to LangSmith)   |
| GET    | /api/feedback/stats    | Feedback approval rate metrics               |
| GET    | /api/feedback/negative | Negative feedback for prompt improvement     |
| GET    | /api/action/tools      | List available action tools                  |
| GET    | /api/prompts           | List all DB-stored prompt templates          |
| POST   | /api/clarify           | Submit clarification for ambiguous queries   |
| GET    | /health                | Health check                                 |

---

## State Fields (v7.0 additions)

| Field               | Type       | Description                                                          |
| ------------------- | ---------- | -------------------------------------------------------------------- |
| `intent`            | str        | "read" or "action"                                                   |
| `react_steps`       | list[dict] | All tool calls: step, tool, args, approved, success, message, result |
| `react_result`      | str        | Final ReAct agent summary                                            |
| `pending_tool_call` | dict       | Tool call currently awaiting human approval                          |

---

## Key Architectural Decisions (v7.0)

| Decision                         | Rationale                                         | Impact                                    |
| -------------------------------- | ------------------------------------------------- | ----------------------------------------- |
| **Intent detection as Phase 0**  | Single fast node before parallel init             | Action queries skip SQL pipeline entirely |
| **ReAct as loop with interrupt** | Each iteration = one node + one interrupt pause   | Clean HITL per tool, resumable state      |
| **Keyword fast path for intent** | No LLM needed for obvious queries                 | 0ms overhead for 80% of cases             |
| **Tool registry pattern**        | `TOOLS` dict + `execute_tool()` dispatcher        | Easy to add new tools                     |
| **Real DB mutations**            | create_po / update_shipment write to warehouse DB | End-to-end testable                       |

---

## Checkpointing & Scaling

| Component         | Before (v6)                                | After (v7)                                    |
| ----------------- | ------------------------------------------ | --------------------------------------------- |
| Checkpointer      | `MemorySaver` (in-memory, single instance) | `PostgresSaver` (distributed, multi-instance) |
| State persistence | Lost on restart                            | Survives crashes, supports horizontal scaling |
| HITL resume       | Single-process only                        | Any instance can resume an interrupted graph  |

---

## Prompt Management

All 9 system prompts are stored in `prompt_template` table (PostgreSQL):

| Prompt Name            | Used By                | Purpose                         |
| ---------------------- | ---------------------- | ------------------------------- |
| `intent_detection`     | intent_detector.py     | Classify read vs action         |
| `complexity_detection` | pipeline.py            | Classify query complexity       |
| `sql_generation`       | sql_generator_agent.py | Generate SQL from schema        |
| `sql_self_consistency` | sql_generator_agent.py | Verify SQL matches intent       |
| `ambiguity_resolution` | ambiguity_agent.py     | Detect/resolve ambiguity        |
| `memory_summarization` | memory_agent.py        | Summarize conversation overflow |
| `response_synthesis`   | response_agent.py      | Explain results to user         |
| `response_system`      | response_agent.py      | System message for response     |
| `react_system`         | react_agent.py         | ReAct agent instructions        |

**Zero hardcoded prompts** — if a prompt fails to load from DB, the agent logs an error and gracefully degrades (skip step or fail fast). No fallback strings in code.

Prompts are hot-swappable: update the DB row and it takes effect immediately (no redeployment).

---

## Feedback & Improvement Loop

```
User Query → Pipeline → Response (with run_id)
     |                              |
     |                    User clicks 👍 / 👎
     |                              |
     v                              v
query_feedback table        LangSmith Feedback API
     |                              |
     v                              v
GET /api/feedback/negative   Visible on trace in LangSmith UI
     |
     v
Analyze patterns → Update prompts in DB → Instant improvement
```

Feedback fields: `session_id`, `run_id`, `query`, `generated_sql`, `rating` (+1/-1), `comment`, `correction`

