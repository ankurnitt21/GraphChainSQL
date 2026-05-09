"""Database layer - sync engine for queries, prompt storage, schema DDL."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.core import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url_sync,
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(bind=engine)


def execute_query(sql: str, timeout: int = 30) -> list[dict]:
    """Execute a read-only SQL query with timeout."""
    with SessionLocal() as session:
        session.execute(text(f"SET statement_timeout = '{timeout}s'"))
        result = session.execute(text(sql))
        columns = list(result.keys())
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]


def get_full_schema_ddl() -> str:
    """Get complete DDL for all business tables including FK relationships."""
    with SessionLocal() as session:
        tables_result = session.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "AND table_name NOT IN ('conversation', 'conversation_summary', 'schema_description', 'prompt_template') "
            "ORDER BY table_name"
        ))
        tables = [r[0] for r in tables_result.fetchall()]

        lines = []
        for table in tables:
            cols_result = session.execute(text(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_name = :tbl AND table_schema = 'public' "
                "ORDER BY ordinal_position"
            ), {"tbl": table})
            cols = cols_result.fetchall()
            lines.append(f"TABLE {table} (")
            for col_name, data_type, nullable, default in cols:
                parts = [f"  {col_name} {data_type}"]
                if nullable == "NO":
                    parts.append("NOT NULL")
                if default and "nextval" in str(default):
                    parts.append("PRIMARY KEY")
                lines.append(" ".join(parts))
            lines.append(")")
            lines.append("")

        # Foreign keys
        fk_result = session.execute(text(
            "SELECT tc.table_name, kcu.column_name, "
            "ccu.table_name AS foreign_table, ccu.column_name AS foreign_column "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
            "JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name "
            "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'"
        ))
        fks = fk_result.fetchall()
        if fks:
            lines.append("FOREIGN KEYS:")
            for src, src_col, fk_tbl, fk_col in fks:
                lines.append(f"  {src}.{src_col} -> {fk_tbl}.{fk_col}")

        return "\n".join(lines)


def get_schema_descriptions() -> list[dict]:
    """Get all schema descriptions for LLM context."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT table_name, column_name, domain, description, data_type "
            "FROM schema_description ORDER BY domain, table_name"
        ))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]


def get_conversations(session_id: str, limit: int = 10) -> list[dict]:
    """Get recent conversation history."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT role, content, sql_query, created_at "
            "FROM conversation WHERE session_id = :sid "
            "ORDER BY created_at DESC LIMIT :lim"
        ), {"sid": session_id, "lim": limit})
        rows = [dict(zip(result.keys(), row)) for row in result.fetchall()]
        return list(reversed(rows))


def save_conversation(session_id: str, role: str, content: str, sql_query: str | None = None):
    """Save a conversation turn."""
    with SessionLocal() as session:
        session.execute(text(
            "INSERT INTO conversation (session_id, role, content, sql_query) "
            "VALUES (:sid, :role, :content, :sql)"
        ), {"sid": session_id, "role": role, "content": content, "sql": sql_query})
        session.commit()


def get_conversation_summary(session_id: str) -> str | None:
    """Get the stored conversation summary for a session."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT summary FROM conversation_summary WHERE session_id = :sid"
        ), {"sid": session_id})
        row = result.fetchone()
        return row[0] if row else None


def save_conversation_summary(session_id: str, summary: str, approximate_tokens: int = 0):
    """Upsert conversation summary."""
    with SessionLocal() as session:
        session.execute(text(
            "INSERT INTO conversation_summary (session_id, summary, approximate_tokens, updated_at) "
            "VALUES (:sid, :summary, :tokens, NOW()) "
            "ON CONFLICT (session_id) DO UPDATE SET summary = :summary, "
            "approximate_tokens = :tokens, updated_at = NOW()"
        ), {"sid": session_id, "summary": summary, "tokens": approximate_tokens})
        session.commit()


def get_schema_embeddings() -> list[dict]:
    """Get schema descriptions with embeddings for vector search."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT id, table_name, column_name, domain, description, data_type "
            "FROM schema_description ORDER BY domain, table_name"
        ))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]
