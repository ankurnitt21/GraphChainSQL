# Flowchart - GraphChainSQL v8.0

> **Ops:** Environment tiers (`APP_ENV`), OpenTelemetry → LangSmith, and optional RAGAS storage are summarized under **Configuration and observability** in [README.md](README.md).

## Dual Pipeline Decision Flow

```
POST /api/query
       |
       v
+------------------------------------------------------------------------+
|  Phase 1: parallel_init  (4-way ThreadPoolExecutor)                   |
|                                                                        |
|  +----------------+  +--------+  +----------+  +----------+          |
|  |intent_detector |  | memory |  | cache_l1 |  | embed    |          |
|  |  (OpenAI LLM)  |  | (DB)   |  | (Redis)  |  | (OpenAI) |          |
|  +----------------+  +--------+  +----------+  +----------+          |
|                                                                        |
|  + _fast_complexity(query)  keyword heuristic, 0 ms, no LLM           |
|    -> sets query_complexity = "simple" | "moderate" | "complex"       |
|    OTEL context explicitly propagated into each thread                |
+-------------------------------------+----------------------------------+
                                      |
         +----------------------------+-------------------+
         |                            |                   |
    "action"                   keyword="simple"    moderate/complex
         |                            |                   |
         v                            v                   v
+---------------------------+      cache_l2     ambiguity_agent (LLM)
|   ReAct LOOP (max 5 steps)|         |                   |
|                           |         +--------+----------+
|  LLM THINKS               |                  |
|  interrupt() HITL PAUSE   |               cache_l2
|  EXECUTE TOOL             |                  |
|  loop / done              |         +--------+--------+
+---------------------------+       L2 hit           L2 miss
                                      |                   |
                                 from_cache       schema_retriever (DB)
                                                          |
                                                   sql_generator (LLM)
                                                          |
                                                   sql_validator (no LLM)
                                                   (retry loop x3)
                                                          |
                                                   approval_agent
                                                   (HITL optional)
                                                          |
                                                   sql_executor (DB)
                                                          |
                                               response_synthesizer
                                      (template if simple, LLM if moderate/complex)
```

---

## Response Path by Complexity

```
query_complexity = "simple" AND rows <= 10
       |
       v
  Template response (0 ms, no LLM)
  PII: regex only (~0 ms)

query_complexity = "moderate"
       |
       v
  LLM gpt-4o-mini (~3-5 s)
  PII: regex only (~0 ms)

query_complexity = "complex"
       |
       v
  LLM gpt-4o-mini (~3-5 s)
  PII: full LLM guard (~1.3 s per pass)
```

---

## State Machine (action pipeline)

```
status: processing -> react_agent (interrupt) -> awaiting_tool_approval
                                  |
                   user approves / rejects
                                  |
                   resume Command(approved=T/F)
                                  |
             approved=True  ->  execute tool  -> processing (loop)
                                  |                    |
             approved=False ->  action_rejected     "done" -> completed
```

---

## Response status values

| Status                   | Meaning                                         |
| ------------------------ | ----------------------------------------------- |
| `processing`             | Pipeline running                                |
| `completed`              | Read query answered OR action pipeline finished |
| `awaiting_approval`      | SQL HITL paused (read pipeline)                 |
| `awaiting_tool_approval` | ReAct tool HITL paused (action pipeline)        |
| `awaiting_clarification` | Ambiguity agent needs input                     |
| `action_rejected`        | User rejected a tool call                       |
| `failed`                 | Error in pipeline                               |

---

## Feedback Loop Flow

```
+--------------------------------------------------------------+
|                     QUERY RESPONSE                           |
|  { status: "completed", sql, result, run_id: "abc123..." }  |
+----------------------------+---------------------------------+
                             |
                   User clicks thumbs up or down
                             |
                             v
             +--------------------------+
             |  POST /api/feedback      |
             |  {                       |
             |    session_id,           |
             |    query,                |
             |    rating: +1 or -1,     |
             |    run_id: "abc123...",  |
             |    comment?,             |
             |    correction?           |
             |  }                       |
             +------------+-------------+
                          |
            +-------------+--------------+
            |                            |
            v                            v
 +------------------+       +-------------------------+
 | PostgreSQL        |       | LangSmith Feedback API  |
 | query_feedback    |       | POST /feedback          |
 | table (local)     |       | score: 0 or 1           |
 +------------------+       | run_id -> linked trace   |
                             +-------------------------+
```

---

## Checkpointing (PostgresSaver)

```
+------------+     +------------+     +------------+
| App Node 1 |     | App Node 2 |     | App Node 3 |
+-----+------+     +-----+------+     +-----+------+
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
              +----------------------+
              |     PostgreSQL       |
              |  checkpoint tables   |
              |  (langgraph managed) |
              +----------------------+

- Any instance can resume an interrupted HITL graph
- State survives crashes / restarts
- Replaces in-memory MemorySaver (v6)
```

---

## Prompt Loading (Zero Hardcoding)

```
Agent Node starts
       |
       v
+-------------------------+
| prompts.get_prompt(name) | <- loads from prompt_template table
+------------+------------+
             |
       +-----+------+
       |             |
    found          not found
       |             |
       v             v
  Use prompt    Log ERROR + graceful degrade
  from DB       (skip step / fail fast)
                NO fallback string in code
```
