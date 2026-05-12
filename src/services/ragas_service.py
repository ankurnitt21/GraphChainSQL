"""RAGAS evaluation: optional LLM-as-judge scores persisted for offline analysis."""

from __future__ import annotations

import threading
from typing import Any

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from datasets import Dataset
from sqlalchemy import text

import structlog
from src.core import get_settings
from src.core.database import SessionLocal

log = structlog.get_logger()


def _ensure_ragas_table() -> None:
    with SessionLocal() as session:
        session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS ragas_eval_result (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(128) NOT NULL,
                run_id VARCHAR(128),
                question TEXT,
                answer_excerpt TEXT,
                faithfulness DOUBLE PRECISION,
                answer_relevancy DOUBLE PRECISION,
                eval_model VARCHAR(128),
                error TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
            )
        )
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_ragas_eval_session ON ragas_eval_result (session_id)"
            )
        )
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_ragas_eval_created ON ragas_eval_result (created_at DESC)"
            )
        )
        session.commit()


def build_ragas_contexts(state: dict[str, Any]) -> list[str]:
    """Turn pipeline state into RAGAS `contexts` (grounding text for faithfulness)."""
    chunks: list[str] = []
    schema = (state.get("schema_context") or "").strip()
    if schema:
        chunks.append(schema[:8000])
    sql = (state.get("generated_sql") or "").strip()
    if sql:
        chunks.append(f"Executed SQL:\n{sql}"[:4000])
    results = state.get("results")
    if results:
        chunks.append(f"Result rows (sample):\n{str(results)[:6000]}")
    if not chunks:
        chunks.append("(no schema/sql/results in state)")
    return chunks[:5]


def evaluate_response(question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    """Run RAGAS faithfulness + answer_relevancy on one Q/A pair."""
    try:
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        }
        log.info("ragas_evaluation", scores=scores)
        return scores
    except Exception as e:
        log.warning("ragas_evaluation_failed", error=str(e))
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "_error": str(e)}


def evaluate_sql_correctness(question: str, sql: str, schema: str) -> float:
    """Evaluate SQL string relevancy against schema (legacy helper)."""
    try:
        data = {
            "question": [question],
            "answer": [sql],
            "contexts": [[schema]],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[answer_relevancy])
        score = float(result["answer_relevancy"])
        log.info("ragas_sql_score", score=score)
        return score
    except Exception as e:
        log.warning("ragas_sql_eval_failed", error=str(e))
        return 0.5


def persist_ragas_eval(
    *,
    session_id: str,
    run_id: str | None,
    question: str,
    answer_excerpt: str,
    scores: dict[str, float],
    error: str | None = None,
) -> None:
    """Insert one RAGAS row (sync; call from background thread)."""
    _ensure_ragas_table()
    s = get_settings()
    model = (s.openai_chat_model or "").strip() or "unknown"
    ff = float(scores.get("faithfulness", 0.0))
    ar = float(scores.get("answer_relevancy", 0.0))
    with SessionLocal() as db:
        db.execute(
            text(
                """
                INSERT INTO ragas_eval_result
                (session_id, run_id, question, answer_excerpt, faithfulness, answer_relevancy, eval_model, error)
                VALUES (:session_id, :run_id, :question, :answer_excerpt, :faithfulness, :answer_relevancy, :eval_model, :error)
                """
            ),
            {
                "session_id": session_id[:128],
                "run_id": (run_id or "")[:128] or None,
                "question": question[:8000] if question else "",
                "answer_excerpt": (answer_excerpt or "")[:8000],
                "faithfulness": ff,
                "answer_relevancy": ar,
                "eval_model": model[:128],
                "error": error,
            },
        )
        db.commit()


def evaluate_and_persist(
    *,
    session_id: str,
    run_id: str | None,
    state: dict[str, Any],
) -> None:
    """Run RAGAS on a completed pipeline state and persist scores."""
    question = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    answer = (state.get("explanation") or "").strip()
    if not question or not answer:
        log.info("ragas_skip_empty_qa", session_id=session_id)
        return
    contexts = build_ragas_contexts(state)
    scores = evaluate_response(question, answer, contexts)
    err_msg: str | None = None
    if "_error" in scores:
        err_msg = str(scores.pop("_error"))
    persist_ragas_eval(
        session_id=session_id,
        run_id=run_id,
        question=question,
        answer_excerpt=answer[:4000],
        scores=scores,
        error=err_msg,
    )


def schedule_ragas_persist(session_id: str, run_id: str | None, state: dict[str, Any]) -> None:
    """Fire-and-forget background evaluation (extra LLM cost when enabled)."""
    if not get_settings().ragas_collect_on_complete:
        return
    if state.get("status") != "completed":
        return
    if state.get("cache_hit"):
        return
    if not (state.get("explanation") or "").strip():
        return

    def _job() -> None:
        try:
            evaluate_and_persist(session_id=session_id, run_id=run_id, state=state)
        except Exception as e:
            log.warning("ragas_persist_job_failed", session_id=session_id, error=str(e))

    threading.Thread(target=_job, daemon=True, name="ragas-eval").start()


def list_recent_ragas_evals(limit: int = 50) -> list[dict[str, Any]]:
    """Recent stored RAGAS rows for dashboards or debugging."""
    _ensure_ragas_table()
    lim = max(1, min(int(limit), 200))
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                SELECT id, session_id, run_id, question, answer_excerpt, faithfulness, answer_relevancy,
                       eval_model, error, created_at
                FROM ragas_eval_result
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"lim": lim},
        )
        rows = result.mappings().all()
        return [dict(r) for r in rows]
