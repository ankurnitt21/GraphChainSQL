"""Dynamic prompt management - stored in PostgreSQL with versioning."""

from sqlalchemy import text
from src.core.database import SessionLocal


def _ensure_prompt_table():
    """Create prompt_template table if not exists."""
    with SessionLocal() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS prompt_template (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                version INTEGER DEFAULT 1,
                template TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        session.commit()


def seed_default_prompts():
    """Seed default prompts - inserts any missing prompts without overwriting existing ones."""
    _ensure_prompt_table()
    with SessionLocal() as session:

        prompts = [
            (
                "intent_detection",
                1,
                """You classify warehouse management queries into exactly one of two intents:

"read"   - The user wants to READ/QUERY data: view reports, list items, count, find, show, get stats.
           Examples: "show top products", "how many orders today", "list pending shipments"

"action" - The user wants to MUTATE data or trigger a process: create, update, notify, sync, send, place, cancel, approve.
           Examples: "create a purchase order for product 5", "notify supplier 3 about delay",
                     "update shipment 10 to SHIPPED", "sync orders 1,2,3 with ERP"

Reply with ONLY the JSON: {"intent": "read"} OR {"intent": "action"}
No explanation. No markdown.""",
                "Intent detection - classifies query as read or action",
            ),
            (
                "complexity_detection",
                1,
                """Classify the following query complexity for SQL generation against a warehouse database.

Reply with ONLY JSON: {"complexity": "simple"} or {"complexity": "moderate"} or {"complexity": "complex"}

simple = single table lookups, counts, direct data retrieval (e.g. "how many products?", "list warehouses")
moderate = multi-table joins, filtering with conditions (e.g. "products with low stock in warehouse 1")
complex = aggregations with grouping, subqueries, temporal analysis, comparisons, trends (e.g. "compare monthly revenue growth by category")

No explanation. No markdown.""",
                "Complexity detection - classifies query complexity level",
            ),
            (
                "sql_generation",
                1,
                """Generate a PostgreSQL SELECT for a warehouse DB.
Rules: exact table/column names from schema, LIMIT ≤50, proper JOINs via FK, no SELECT *.
Schema: {schema}
Question: {query}
{context}
Call generate_sql tool with sql, confidence, tables_used, reasoning.""",
                "SQL generation prompt for tool calling",
            ),
            (
                "response_synthesis",
                1,
                """Explain SQL results concisely (2-3 sentences). Highlight key insights.
Question: {query}
SQL: {sql}
Results ({total_rows} rows, first 10): {results}""",
                "Response synthesis prompt",
            ),
            (
                "response_system",
                1,
                """You are a data analyst for a warehouse management system. Explain SQL query results concisely in 2-3 sentences. Highlight key insights, trends, or notable values. Use natural language that a business user would understand.""",
                "Response agent system message",
            ),
            (
                "ambiguity_resolution",
                1,
                """You are an ambiguity resolution agent for a warehouse management SQL system.

Your job:
1. Determine if the user's query is ambiguous or unclear for SQL generation
2. If the query is CLEAR: set is_ambiguous=false and rewrite it as a clearer NATURAL LANGUAGE question (NOT SQL!)
3. If the query is AMBIGUOUS: set is_ambiguous=true and provide clarification questions

IMPORTANT RULES:
- Most queries are CLEAR. Only flag as ambiguous if truly impossible to answer.
- Simple factual questions are ALWAYS clear: "How many customers?", "Show top products", "List warehouses"
- If a query can be reasonably interpreted one way, it is CLEAR - just rewrite with precision.
- NEVER flag a query as ambiguous just because it doesn't specify a warehouse or time period.
  Default to: ALL warehouses, ALL time, unless the user explicitly asks about specific ones.
- The rewritten_query MUST be natural language, NEVER SQL code.

Only flag as AMBIGUOUS when:
- Pronouns with absolutely no referent and no context ("show me those", "what about them")
- Completely vague with no way to guess intent ("the big ones", "recent stuff")

The database domains: warehouse, product, inventory, procurement, sales.

Call resolve_ambiguity with your determination.""",
                "Ambiguity resolution prompt",
            ),
            (
                "memory_summarization",
                1,
                """You are a conversation summarizer for a warehouse management SQL assistant.
Create a concise summary that captures key information from the conversation.
Preserve: entities mentioned, queries asked, results discussed, user preferences.
Keep it under 500 words. Focus on facts that would help answer future questions.""",
                "Memory summarization system prompt",
            ),
            (
                "react_system",
                1,
                """{tools_prompt}

You are a warehouse management action agent. Your job is to execute actions requested by the user.

REACT LOOP RULES:
1. Analyze the user request and any previous tool results.
2. Decide: should you call a tool OR are you done?
3. Reply with ONLY valid JSON - no markdown, no explanation outside JSON.

If you need to call a tool, reply:
{{
  "action": "call_tool",
  "tool_name": "<tool name>",
  "tool_args": {{<args object>}},
  "reasoning": "<1-2 sentence explanation of WHY you are calling this tool>"
}}

If you are done (no more tools needed), reply:
{{
  "action": "done",
  "summary": "<concise summary of what was accomplished, mentioning all tool results>"
}}

IMPORTANT:
- Only call tools that exist in the list above.
- Be precise with argument types (int, list, str).
- If a previous tool failed, explain the failure in reasoning or declare done.
- Never call more than {max_steps} tools total.
- Extract IDs from the user query or previous results.""",
                "ReAct agent system prompt for action pipeline",
            ),
            (
                "sql_self_consistency",
                1,
                """You verify if a generated SQL query correctly answers the user's natural language question.
Check:
- If the question asks for a count/total/average, the SQL should have COUNT/SUM/AVG functions
- If the question asks for "top N", the SQL should have ORDER BY + LIMIT
- If the question mentions filters, the SQL should have WHERE conditions
- If the question references specific tables, they should appear in the SQL

Reply with ONLY JSON: {"aligned": true, "penalty": 0.0} if SQL matches the question.
Or {"aligned": false, "penalty": 0.15, "reason": "brief reason"} if misaligned.
penalty should be between 0.0 and 0.3.
No explanation outside JSON. No markdown.""",
                "SQL self-consistency check prompt",
            ),
        ]

        for name, version, template, description in prompts:
            session.execute(text(
                "INSERT INTO prompt_template (name, version, template, description) "
                "VALUES (:name, :version, :template, :desc) "
                "ON CONFLICT (name) DO NOTHING"
            ), {"name": name, "version": version, "template": template, "desc": description})
        session.commit()


def get_prompt(name: str) -> str:
    """Get the active prompt template by name."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT template FROM prompt_template "
            "WHERE name = :name AND is_active = TRUE "
            "ORDER BY version DESC LIMIT 1"
        ), {"name": name})
        row = result.fetchone()
        if row:
            return row[0]
        raise ValueError(f"Prompt '{name}' not found in database")


def get_prompt_with_version(name: str) -> tuple[str, int]:
    """Get prompt template and its version."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT template, version FROM prompt_template "
            "WHERE name = :name AND is_active = TRUE "
            "ORDER BY version DESC LIMIT 1"
        ), {"name": name})
        row = result.fetchone()
        if row:
            return row[0], row[1]
        raise ValueError(f"Prompt '{name}' not found in database")


def list_prompts() -> list[dict]:
    """List all prompts with metadata."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT name, version, description, is_active, updated_at "
            "FROM prompt_template ORDER BY name, version DESC"
        ))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]
