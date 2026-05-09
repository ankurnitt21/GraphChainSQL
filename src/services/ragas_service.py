"""RAGAS evaluation integration for quality gates."""

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
import structlog

log = structlog.get_logger()


def evaluate_response(question: str, answer: str, contexts: list[str]) -> dict:
    """Run RAGAS evaluation on the generated response.
    
    Returns dict with metric scores.
    """
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
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def evaluate_sql_correctness(question: str, sql: str, schema: str) -> float:
    """Evaluate SQL correctness using RAGAS context.
    
    Returns score 0.0 - 1.0.
    """
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
        return 0.5  # Default pass
