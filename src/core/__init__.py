"""Core configuration loaded from environment."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
        ),
        extra="ignore",
        env_nested_delimiter="__",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql+asyncpg://warehouse_admin:warehouse_secret_2024@localhost:5433/warehouse_db"
    database_url_sync: str = "postgresql://warehouse_admin:warehouse_secret_2024@localhost:5433/warehouse_db"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "GraphChainSQL"

    # Runtime tier (dev | test | stg | prod) — drives logging context, health payload, metric labels
    app_env: Literal["dev", "test", "stg", "prod"] = Field(
        default="dev",
        validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT", "NODE_ENV"),
    )

    # Logical service id for OTEL resource (set per deployment)
    service_name: str = Field(
        default="",
        validation_alias=AliasChoices("SERVICE_NAME", "OTEL_SERVICE_NAME"),
    )

    # Release / build id (e.g. CI SHA or semver)
    app_version: str = Field(default="", validation_alias=AliasChoices("APP_VERSION", "GIT_SHA", "BUILD_VERSION"))

    # HTTP server (uvicorn)
    api_host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("API_HOST", "UVICORN_HOST"))
    api_port: int = Field(default=8085, validation_alias=AliasChoices("API_PORT", "UVICORN_PORT", "PORT"))

    # RAGAS: after a completed (non-cache) query, run LLM-as-judge in a background thread and store scores
    ragas_collect_on_complete: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAGAS_COLLECT_ON_COMPLETE"),
    )

    # OTLP trace export (LangSmith-compatible)
    otel_exporter_otlp_endpoint: str = Field(
        default="https://api.smith.langchain.com/otel",
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    # App
    default_page_size: int = 50
    max_retries: int = 3
    cache_ttl_seconds: int = 300
    semantic_cache_threshold: float = 0.92

    # Memory Agent
    memory_token_limit: int = 4000
    memory_max_messages: int = 20

    # Embedding (OpenAI text-embedding-3-small = 1536 dimensions)
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Schema retrieval
    schema_top_k: int = 10

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_rag_index: str = "rag-base"
    pinecone_sql_index: str = "sql-base"

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalize_app_env(cls, v: object) -> str:
        if v is None:
            return "dev"
        s = str(v).strip().lower()
        aliases = {
            "development": "dev",
            "local": "dev",
            "testing": "test",
            "qa": "test",
            "staging": "stg",
            "stage": "stg",
            "preprod": "stg",
            "production": "prod",
            "live": "prod",
        }
        s = aliases.get(s, s)
        if s not in ("dev", "test", "stg", "prod"):
            raise ValueError(f"APP_ENV must be one of dev,test,stg,prod (got {v!r})")
        return s

    @field_validator("service_name", mode="before")
    @classmethod
    def _service_name(cls, v: object) -> str:
        if v is None or not str(v).strip():
            return "unspecified-service"
        return str(v).strip()


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def configure_process_environment() -> Settings:
    """Align os.environ with Settings so OTEL exporters and other libs see one consistent profile."""
    s = get_settings()
    os.environ.setdefault("OTEL_SERVICE_NAME", s.service_name)
    os.environ.setdefault("DEPLOYMENT_ENVIRONMENT", s.app_env)
    dep = f"deployment.environment={s.app_env}"
    cur = (os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or "").strip()
    if dep not in cur:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (f"{dep},{cur}" if cur else dep).strip(",").strip()
    return s


def setup_langsmith():
    """Configure LangSmith environment variables for tracing.
    
    Disables native LangChain tracing (403 on /runs/multipart) 
    and relies on OTEL export to LangSmith instead.
    """
    s = get_settings()
    # Disable native LangChain SDK tracing (use OTEL instead)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if s.langsmith_api_key:
        # OTEL will handle traces via setup_otel()
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_ENDPOINT"] = s.langsmith_endpoint
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
