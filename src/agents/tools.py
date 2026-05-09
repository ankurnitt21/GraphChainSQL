"""Agent tools - Shared utility functions used by agent nodes."""

import json
from langchain_core.tools import tool
from src.core.database import execute_query, get_full_schema_ddl, get_schema_descriptions
from src.core.prompts import get_prompt_with_version
from src.services.cache import semantic_cache_get, semantic_cache_set
from src.services.guardrails_service import validate_sql
import structlog

log = structlog.get_logger()


@tool
def get_database_schema() -> str:
    """Retrieve the complete database schema DDL including all tables, columns, data types, and foreign key relationships."""
    return get_full_schema_ddl()


@tool
def get_schema_context() -> str:
    """Get natural language descriptions of all database tables and columns."""
    descriptions = get_schema_descriptions()
    lines = []
    current_table = ""
    for d in descriptions:
        if d["table_name"] != current_table:
            current_table = d["table_name"]
            lines.append(f"\n{current_table} ({d['domain']}):")
        col = d.get("column_name") or "(table)"
        lines.append(f"  {col}: {d['description']}")
    return "\n".join(lines)


@tool
def validate_generated_sql(sql: str) -> str:
    """Validate a generated SQL query for safety, correctness, and security."""
    is_safe, issues = validate_sql(sql)
    return json.dumps({
        "is_valid": is_safe,
        "issues": issues,
    })


@tool
def get_prompt_template(prompt_name: str) -> str:
    """Retrieve a prompt template from the database by name."""
    try:
        template, version = get_prompt_with_version(prompt_name)
        return json.dumps({"template": template, "version": version, "name": prompt_name})
    except ValueError as e:
        return json.dumps({"error": str(e)})

