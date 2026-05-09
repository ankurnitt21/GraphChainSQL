"""Agents package - Multi-agent SQL pipeline with DAG-based orchestration.

Agents:
  - memory_agent: Token-aware conversation summarization
  - ambiguity_agent: Query ambiguity detection and rewriting
  - cache_agent: Semantic cache lookup (Redis)
  - embedding_agent: Query → dense vector
  - schema_agent: pgvector-based schema retrieval
  - sql_generator_agent: SQL generation from NL + schema
  - sql_validator_agent: SQL safety/correctness validation
  - approval_agent: Human-in-the-loop approval gate
  - executor_agent: Safe SQL execution
  - response_agent: Result → natural language resynthesis
  - pipeline: Graph builder controlling the pipeline flow
"""
