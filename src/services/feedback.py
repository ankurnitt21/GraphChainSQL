"""User feedback service - captures thumbs up/down and sends to LangSmith."""

from sqlalchemy import text
from src.core.database import SessionLocal
from src.core import get_settings
import structlog
import httpx

log = structlog.get_logger()
settings = get_settings()


def _ensure_feedback_table():
    """Create feedback table if not exists."""
    with SessionLocal() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS query_feedback (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                run_id VARCHAR(100),
                query TEXT NOT NULL,
                generated_sql TEXT,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                comment TEXT,
                correction TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feedback_session ON query_feedback(session_id)
        """))
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feedback_rating ON query_feedback(rating)
        """))
        session.commit()


def save_feedback(
    session_id: str,
    query: str,
    rating: int,
    generated_sql: str | None = None,
    comment: str | None = None,
    correction: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Save user feedback to DB and send to LangSmith.

    Args:
        session_id: The session that produced the result
        query: Original natural language query
        rating: 1 (thumbs up) or -1 (thumbs down)
        generated_sql: The SQL that was generated
        comment: Optional user comment
        correction: Optional correct SQL provided by user
        run_id: LangSmith run ID for linking feedback to trace
    """
    _ensure_feedback_table()

    with SessionLocal() as session:
        result = session.execute(text(
            "INSERT INTO query_feedback "
            "(session_id, run_id, query, generated_sql, rating, comment, correction) "
            "VALUES (:sid, :rid, :query, :sql, :rating, :comment, :correction) "
            "RETURNING id"
        ), {
            "sid": session_id,
            "rid": run_id,
            "query": query,
            "sql": generated_sql,
            "rating": rating,
            "comment": comment,
            "correction": correction,
        })
        feedback_id = result.fetchone()[0]
        session.commit()

    # Send feedback to LangSmith if API key is configured
    langsmith_result = None
    if settings.langsmith_api_key and run_id:
        langsmith_result = _send_to_langsmith(run_id, rating, comment, correction)

    log.info(
        "feedback_saved",
        feedback_id=feedback_id,
        session_id=session_id,
        rating=rating,
        langsmith_synced=langsmith_result is not None,
    )

    return {
        "feedback_id": feedback_id,
        "langsmith_synced": langsmith_result is not None,
    }


def _send_to_langsmith(run_id: str, rating: int, comment: str | None, correction: str | None) -> dict | None:
    """Send feedback to LangSmith as a run feedback annotation.

    This makes feedback visible in the LangSmith UI alongside the trace.
    """
    try:
        url = f"{settings.langsmith_endpoint}/feedback"
        headers = {
            "x-api-key": settings.langsmith_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "run_id": run_id,
            "key": "user_rating",
            "score": 1.0 if rating == 1 else 0.0,
            "value": "thumbs_up" if rating == 1 else "thumbs_down",
            "comment": comment or "",
        }
        if correction:
            payload["correction"] = {"desired_output": correction}

        response = httpx.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        log.info("langsmith_feedback_sent", run_id=run_id, status=response.status_code)
        return response.json()
    except Exception as e:
        log.warning("langsmith_feedback_failed", run_id=run_id, error=str(e))
        return None


def get_feedback_stats(session_id: str | None = None) -> dict:
    """Get feedback statistics, optionally filtered by session."""
    with SessionLocal() as session:
        if session_id:
            result = session.execute(text(
                "SELECT rating, COUNT(*) as cnt FROM query_feedback "
                "WHERE session_id = :sid GROUP BY rating"
            ), {"sid": session_id})
        else:
            result = session.execute(text(
                "SELECT rating, COUNT(*) as cnt FROM query_feedback GROUP BY rating"
            ))
        stats = {row[0]: row[1] for row in result.fetchall()}
        return {
            "thumbs_up": stats.get(1, 0),
            "thumbs_down": stats.get(-1, 0),
            "total": stats.get(1, 0) + stats.get(-1, 0),
            "approval_rate": (
                round(stats.get(1, 0) / (stats.get(1, 0) + stats.get(-1, 0)) * 100, 1)
                if (stats.get(1, 0) + stats.get(-1, 0)) > 0 else 0
            ),
        }


def get_negative_feedback(limit: int = 50) -> list[dict]:
    """Get recent negative feedback for prompt improvement analysis."""
    with SessionLocal() as session:
        result = session.execute(text(
            "SELECT session_id, query, generated_sql, comment, correction, created_at "
            "FROM query_feedback WHERE rating = -1 "
            "ORDER BY created_at DESC LIMIT :lim"
        ), {"lim": limit})
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]
