"""Quick test - invoke graph directly to see errors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load env
from dotenv import load_dotenv
load_dotenv()

print("1. Building graph...")
from src.agents.pipeline import build_graph
graph = build_graph()
print("2. Graph built OK")

from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="Show me top 5 products by unit price")],
    "session_id": "test-debug",
    "original_query": "Show me top 5 products by unit price",
    # Memory & History (None = not yet loaded)
    "conversation_history": None,
    "conversation_summary": "",
    "history_token_usage": 0,
    # Ambiguity
    "rewritten_query": "",
    "is_ambiguous": None,
    "clarification_message": "",
    "clarification_options": [],
    # Dual-layer Cache
    "cache_hit": False,
    "l1_checked": False,
    "l2_hit": False,
    "cached_response": {},
    # Embedding
    "query_embedding": [],
    "embedding_done": False,
    # Schema
    "schema_context": "",
    "tables_used": [],
    "schema_relationships": [],
    # SQL
    "generated_sql": "",
    "sql_confidence": 0.0,
    "validation_errors": [],
    "retry_count": 0,
    "sql_validated": False,
    # Results
    "results": [],
    "explanation": "",
    # Control
    "status": "processing",
    "error": "",
    "next_agent": "",
    # HITL
    "require_approval": False,
    "approved": None,
    # v6.0 fields
    "ambiguity_score": 0.0,
    "rewrite_confidence": 0.0,
    "estimated_cost": "",
    "structured_output": {},
    "approval_explanation": "",
    "query_complexity": "",
    "decision_trace": [],
}

config = {"configurable": {"thread_id": "test-debug"}}

print("3. Invoking graph...")
try:
    result = graph.invoke(state, config=config)
    print("4. SUCCESS!")
    print(f"   Status: {result.get('status')}")
    print(f"   Complexity: {result.get('query_complexity')}")
    print(f"   Ambiguity Score: {result.get('ambiguity_score')}")
    print(f"   Rewrite Confidence: {result.get('rewrite_confidence')}")
    print(f"   Rewritten Query: {result.get('rewritten_query', '')[:100]}")
    print(f"   SQL: {result.get('generated_sql')}")
    print(f"   SQL Confidence: {result.get('sql_confidence')}")
    print(f"   Estimated Cost: {result.get('estimated_cost')}")
    print(f"   Results: {result.get('results', [])[:2]}")
    print(f"   Explanation: {result.get('explanation', '')[:100]}")
    print(f"   Approval Explanation: {result.get('approval_explanation', '')[:100]}")
    print(f"   Cache Hit: {result.get('cache_hit')}")
    print(f"   Decision Trace: {result.get('decision_trace', [])}")
    if result.get('structured_output'):
        print(f"   Structured Output keys: {list(result['structured_output'].keys())}")
except Exception as e:
    print(f"4. ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
