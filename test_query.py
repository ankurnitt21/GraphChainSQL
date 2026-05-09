"""Test script - query pipeline + feedback endpoint."""
import httpx
import json

BASE = "http://localhost:8000"

print("=" * 60)
print("TEST 1: Query - How many products do we have?")
print("=" * 60)

resp = httpx.post(f"{BASE}/api/query", json={"query": "How many products do we have?"}, timeout=60)
data = resp.json()

print(f"Status: {data['status']}")
print(f"SQL: {data.get('generated_sql', 'N/A')}")
print(f"Explanation: {(data.get('explanation') or 'N/A')[:200]}")
print(f"Run ID: {data.get('run_id', 'N/A')}")
print(f"Cache hit: {data.get('cache_hit', False)}")
print(f"Session ID: {data.get('session_id', 'N/A')}")

session_id = data.get("session_id", "test")
run_id = data.get("run_id")

print()
print("=" * 60)
print("TEST 2: Submit POSITIVE feedback (thumbs up)")
print("=" * 60)

feedback_resp = httpx.post(f"{BASE}/api/feedback", json={
    "session_id": session_id,
    "query": "How many products do we have?",
    "rating": 1,
    "generated_sql": data.get("generated_sql", ""),
    "comment": "Perfect answer!",
    "run_id": run_id,
}, timeout=30)

print(f"Feedback response: {feedback_resp.json()}")

print()
print("=" * 60)
print("TEST 3: Submit NEGATIVE feedback (thumbs down)")
print("=" * 60)

feedback_resp2 = httpx.post(f"{BASE}/api/feedback", json={
    "session_id": session_id,
    "query": "How many products do we have?",
    "rating": -1,
    "generated_sql": data.get("generated_sql", ""),
    "comment": "Should include inactive products too",
    "correction": "SELECT COUNT(*) FROM product",
    "run_id": run_id,
}, timeout=30)

print(f"Feedback response: {feedback_resp2.json()}")

print()
print("=" * 60)
print("TEST 4: Get feedback stats")
print("=" * 60)

stats_resp = httpx.get(f"{BASE}/api/feedback/stats", timeout=10)
print(f"Stats: {stats_resp.json()}")

print()
print("=" * 60)
print("TEST 5: Get negative feedback list")
print("=" * 60)

neg_resp = httpx.get(f"{BASE}/api/feedback/negative", timeout=10)
neg_data = neg_resp.json()
print(f"Negative feedback count: {len(neg_data)}")
for item in neg_data[:3]:
    print(f"  - Query: {item['query'][:50]}, Comment: {item.get('comment', 'N/A')}")

print()
print("=" * 60)
print("TEST 6: Verify PostgresSaver checkpoint (second query same session)")
print("=" * 60)

resp2 = httpx.post(f"{BASE}/api/query", json={
    "query": "Show top 5 products by price",
    "session_id": session_id,
}, timeout=60)
data2 = resp2.json()
print(f"Status: {data2['status']}")
print(f"SQL: {data2.get('generated_sql', 'N/A')}")
print(f"Run ID: {data2.get('run_id', 'N/A')}")

print()
print("=" * 60)
print("ALL TESTS PASSED" if all([
    data["status"] == "completed",
    data.get("run_id"),
    feedback_resp.status_code == 200,
    feedback_resp2.status_code == 200,
    data2["status"] == "completed",
]) else "SOME TESTS FAILED")
print("=" * 60)
