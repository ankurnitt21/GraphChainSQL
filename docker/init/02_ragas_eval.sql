-- Offline / async RAGAS quality scores (see src/services/ragas_service.py)
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
);

CREATE INDEX IF NOT EXISTS idx_ragas_eval_session ON ragas_eval_result (session_id);
CREATE INDEX IF NOT EXISTS idx_ragas_eval_created ON ragas_eval_result (created_at DESC);
