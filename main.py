"""FastAPI application entry point."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import setup_langsmith, get_settings
from src.core.tracing import setup_otel

# Setup LangSmith BEFORE any langchain imports
setup_langsmith()
# Setup OTEL tracing to LangSmith
setup_otel()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import router
from src.core.prompts import seed_default_prompts
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

log = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="GraphChainSQL",
    description="Multi-Agent Text-to-SQL Pipeline with Memory, Guardrails, Resilience, and Observability",
    version="7.0.0",
)

app.include_router(router)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    """Serve the UI."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "GraphChainSQL API - Multi-Agent Text-to-SQL"}


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    log.info("starting_graphchainsql", version="7.0.0", langsmith_project=settings.langsmith_project)
    # Seed prompts
    try:
        seed_default_prompts()
        log.info("prompts_initialized")
    except Exception as e:
        log.error("prompt_seed_failed", error=str(e))

    # Ensure feedback table exists
    try:
        from src.services.feedback import _ensure_feedback_table
        _ensure_feedback_table()
        log.info("feedback_table_initialized")
    except Exception as e:
        log.warning("feedback_table_init_skipped", error=str(e))

    # Pre-initialize embedding model in a thread to avoid blocking startup
    import threading
    def _load_embeddings():
        try:
            from src.services.cache import _get_embeddings
            _get_embeddings()
            log.info("embeddings_loaded")
        except Exception as e:
            log.warning("embedding_preload_skipped", error=str(e))
    threading.Thread(target=_load_embeddings, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8085, reload=True)
