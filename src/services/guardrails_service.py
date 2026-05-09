"""Guardrails service - input/output validation using guardrails-ai library."""

import structlog
import re

log = structlog.get_logger()

# Pre-built guards with guardrails-ai validators
_input_guard = None
_output_guard = None


def _get_input_guard():
    """Lazy-init input guard with toxic language detection."""
    global _input_guard
    if _input_guard is None:
        try:
            from guardrails import Guard
            from guardrails.hub import ToxicLanguage
            _input_guard = Guard(name="sql_input_guard").use(
                ToxicLanguage(threshold=0.8, on_fail="noop"),
            )
        except Exception as e:
            log.debug("guardrails_input_init_fallback", error=str(e))
            _input_guard = False
    return _input_guard if _input_guard is not False else None


def _get_output_guard():
    """Lazy-init output guard with PII detection."""
    global _output_guard
    if _output_guard is None:
        try:
            from guardrails import Guard
            from guardrails.hub import DetectPII
            _output_guard = Guard(name="sql_output_guard").use(
                DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"], on_fail="fix"),
            )
        except Exception as e:
            log.debug("guardrails_output_init_fallback", error=str(e))
            _output_guard = False
    return _output_guard if _output_guard is not False else None


# SQL injection patterns (fast regex pre-check before LLM)
SQL_INJECTION_PATTERNS = [
    r"\b(DROP|ALTER|TRUNCATE|CREATE|INSERT|UPDATE|DELETE|EXEC|EXECUTE)\b",
    r";\s*(DROP|ALTER|DELETE|UPDATE|INSERT)",
    r"--\s*$",
    r"/\*.*\*/",
    r"\bUNION\s+ALL\s+SELECT\b",
    r"\bOR\s+1\s*=\s*1\b",
]


def validate_input(query: str) -> tuple[bool, list[str]]:
    """Validate user input for safety using guardrails-ai + regex.
    
    Returns (is_safe, list_of_issues).
    """
    issues = []

    # Fast regex injection check
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            issues.append("Potential SQL injection detected")
            break

    if len(query) > 2000:
        issues.append("Query too long (max 2000 chars)")

    # Guardrails-ai toxic language check
    guard = _get_input_guard()
    if guard:
        try:
            result = guard.validate(query)
            if result.validation_passed is False:
                issues.append("Input failed guardrails validation (toxic content)")
        except Exception as e:
            log.debug("guardrails_input_validate_skip", error=str(e))

    return (len(issues) == 0, issues)


def validate_sql(sql: str) -> tuple[bool, list[str]]:
    """Validate generated SQL for safety.
    
    Returns (is_safe, list_of_issues).
    """
    issues = []

    if not sql or not sql.strip():
        issues.append("Empty SQL")
        return (False, issues)

    sql_upper = sql.upper().strip()

    # Must be SELECT only
    if not sql_upper.startswith("SELECT"):
        issues.append("Only SELECT queries are allowed")

    # No data modification
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "EXEC"]
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", sql_upper):
            issues.append(f"Dangerous keyword detected: {kw}")

    # No multiple statements
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) > 1:
        issues.append("Multiple SQL statements not allowed")

    # Must have LIMIT (unless it's a pure aggregate query like COUNT/SUM/AVG without GROUP BY)
    if "LIMIT" not in sql_upper:
        # Allow pure aggregate queries (no GROUP BY = single row result)
        is_aggregate_only = (
            re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_upper)
            and "GROUP BY" not in sql_upper
        )
        if not is_aggregate_only:
            issues.append("Missing LIMIT clause")

    # No system tables
    if "PG_CATALOG" in sql_upper or "INFORMATION_SCHEMA" in sql_upper:
        issues.append("System table access not allowed")

    return (len(issues) == 0, issues)


def validate_output(response_text: str) -> tuple[bool, list[str], str]:
    """Validate LLM output for PII using guardrails-ai DetectPII.
    
    Returns (is_safe, list_of_issues, cleaned_text).
    The cleaned_text has PII redacted when on_fail="fix" is active.
    """
    issues = []
    cleaned = response_text

    # Guardrails-ai PII detection
    guard = _get_output_guard()
    if guard:
        try:
            result = guard.validate(response_text)
            if result.validation_passed is False:
                issues.append("PII detected in output (redacted)")
                # Use the fixed/redacted output from guardrails
                if result.validated_output:
                    cleaned = result.validated_output
        except Exception as e:
            log.debug("guardrails_output_validate_skip", error=str(e))

    return (len(issues) == 0, issues, cleaned)
