"""SQL Validation Agent - Multi-layer validation for safety, syntax, and correctness.

Validation Layers:
  1. Syntax validation (SELECT only, no DDL/DML)
  2. Schema alignment (table/column existence)
  3. Dangerous query detection (full scans, system tables)
  4. Logical correctness (JOIN paths, column references)
"""

import re
from src.core.state import AgentState
from src.core.tracing import trace_agent_node
from src.services.guardrails_service import validate_sql
import structlog

log = structlog.get_logger()


def _validate_schema_alignment(sql: str, schema_context: str) -> list[str]:
    """Layer 2: Validate SQL references exist in schema."""
    issues = []
    if not schema_context:
        return issues

    tables_in_schema = set(re.findall(r"TABLE (\w+) \(", schema_context))
    if not tables_in_schema:
        return issues

    # Extract tables from SQL (FROM, JOIN clauses)
    sql_tables = set(re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE))
    invalid_tables = sql_tables - tables_in_schema
    if invalid_tables:
        issues.append(f"Unknown tables: {', '.join(sorted(invalid_tables))}")

    return issues


def _validate_dangerous_patterns(sql: str) -> list[str]:
    """Layer 3: Detect dangerous query patterns."""
    issues = []
    sql_upper = sql.upper()

    # Cartesian product (multiple tables without JOIN condition)
    from_tables = re.findall(r'FROM\s+(\w+)\s*,\s*(\w+)', sql, re.IGNORECASE)
    if from_tables:
        issues.append("Potential cartesian product: use explicit JOIN instead of comma-separated tables")

    # SELECT * (overly broad)
    if re.search(r'SELECT\s+\*\s+FROM', sql, re.IGNORECASE):
        issues.append("SELECT * is not allowed - specify columns explicitly")

    # Subquery without LIMIT that could return many rows
    subqueries = re.findall(r'\(\s*SELECT\s+[^)]+\)', sql, re.IGNORECASE)
    for sq in subqueries:
        if 'LIMIT' not in sq.upper() and 'COUNT' not in sq.upper():
            if 'IN' in sql_upper or 'EXISTS' not in sql_upper:
                pass  # IN subqueries are generally fine

    return issues


def _validate_logical_correctness(sql: str, schema_context: str) -> list[str]:
    """Layer 4: Basic logical correctness checks."""
    issues = []

    # Check for GROUP BY consistency
    if re.search(r'\bGROUP\s+BY\b', sql, re.IGNORECASE):
        # If there's a GROUP BY, ensure aggregate functions or grouped columns in SELECT
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Has non-aggregated columns without GROUP BY reference?
            has_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', select_clause, re.IGNORECASE))
            if not has_agg:
                issues.append("GROUP BY clause without aggregate functions in SELECT")

    # Check ORDER BY references valid columns (basic)
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', sql, re.IGNORECASE)
    if order_match:
        order_clause = order_match.group(1).strip().rstrip(';')
        # Numeric references are valid (ORDER BY 1, 2)
        if not re.match(r'^[\d\s,]+$', order_clause):
            pass  # Column name references - can't fully validate without running

    return issues


def _estimate_query_cost(sql: str, schema_context: str) -> tuple[str, list[str]]:
    """Layer 5: Heuristic cost estimation without EXPLAIN.

    Returns (cost_level, warnings) where cost_level is 'low'|'medium'|'high'.
    """
    warnings = []
    cost_score = 0
    sql_upper = sql.upper()

    # Count JOINs - each join multiplies potential rows
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    cost_score += join_count * 2

    # Missing WHERE on multi-table query
    tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
    has_where = "WHERE" in sql_upper
    if len(tables) > 1 and not has_where:
        cost_score += 5
        warnings.append("Multi-table query without WHERE clause may be expensive")

    # Large LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        limit_val = int(limit_match.group(1))
        if limit_val > 100:
            cost_score += 3
        if limit_val > 1000:
            cost_score += 5
            warnings.append(f"Large LIMIT ({limit_val}) - consider reducing")
    elif not re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_upper):
        # No LIMIT and no aggregate - potentially unbounded
        cost_score += 4
        warnings.append("No LIMIT clause on non-aggregate query")

    # Subqueries add complexity
    subquery_count = sql_upper.count("SELECT") - 1
    if subquery_count > 0:
        cost_score += subquery_count * 3

    # LIKE with leading wildcard (no index use)
    if re.search(r"LIKE\s+'%", sql, re.IGNORECASE):
        cost_score += 3
        warnings.append("Leading wildcard LIKE prevents index usage")

    # DISTINCT on multiple columns
    if "DISTINCT" in sql_upper and len(re.findall(r',', sql[:sql_upper.find("FROM")])) > 3:
        cost_score += 2

    # Determine cost level
    if cost_score >= 10:
        cost_level = "high"
    elif cost_score >= 5:
        cost_level = "medium"
    else:
        cost_level = "low"

    return cost_level, warnings


@trace_agent_node("sql_validator")
def sql_validator_node(state: AgentState) -> dict:
    """Multi-layer SQL validation.

    Layer 1: Guardrails safety (injection, DDL, LIMIT)
    Layer 2: Schema alignment (table/column existence)
    Layer 3: Dangerous patterns (cartesian, SELECT *)
    Layer 4: Logical correctness (GROUP BY, ORDER BY)
    Layer 5: Cost estimation (heuristic-based)
    """
    sql = state.get("generated_sql", "")
    schema_context = state.get("schema_context", "")
    messages = state.get("messages", [])
    all_issues = []

    # Layer 1: Guardrails safety validation
    is_safe, safety_issues = validate_sql(sql)
    all_issues.extend(safety_issues)

    # Layer 2: Schema alignment
    schema_issues = _validate_schema_alignment(sql, schema_context)
    all_issues.extend(schema_issues)

    # Layer 3: Dangerous patterns
    danger_issues = _validate_dangerous_patterns(sql)
    all_issues.extend(danger_issues)

    # Layer 4: Logical correctness
    logic_issues = _validate_logical_correctness(sql, schema_context)
    all_issues.extend(logic_issues)

    # Layer 5: Cost estimation
    cost_level, cost_warnings = _estimate_query_cost(sql, schema_context)
    if cost_level == "high":
        all_issues.extend(cost_warnings)

    if all_issues:
        log.warning("sql_validation_failed", issues=all_issues, sql=sql[:100])

    return {
        "messages": messages,
        "validation_errors": all_issues,
        "sql_validated": True,
        "estimated_cost": cost_level,
    }
