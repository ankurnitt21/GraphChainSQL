"""Setup script: Flush Pinecone indexes and populate with domain docs and schema metadata."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from src.core import get_settings

settings = get_settings()

print("=" * 60)
print("GraphChainSQL - Pinecone Setup")
print("=" * 60)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    openai_api_key=settings.openai_api_key,
)

RAG_INDEX = settings.pinecone_rag_index
SQL_INDEX = settings.pinecone_sql_index
DIMENSION = 1536  # OpenAI text-embedding-3-small

def ensure_index_exists(index_name: str, force_recreate: bool = False):
    """Create index if it doesn't exist or recreate if dimensions mismatch."""
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing:
        # Check dimension
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        current_dim = stats.dimension
        
        if current_dim != DIMENSION or force_recreate:
            print(f"Index {index_name} has wrong dimension ({current_dim} vs {DIMENSION}). Deleting...")
            pc.delete_index(index_name)
            time.sleep(5)
            existing = []
        else:
            print(f"Index {index_name} already exists with correct dimension ({DIMENSION})")
            return
    
    if index_name not in existing:
        print(f"Creating index: {index_name} with dimension {DIMENSION}")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Waiting for index {index_name} to be ready...")
        time.sleep(15)

def flush_index(index_name: str):
    """Delete all vectors from an index (all namespaces)."""
    print(f"\nFlushing index: {index_name}")
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else [""]
        
        for ns in namespaces:
            try:
                index.delete(delete_all=True, namespace=ns)
                print(f"  Deleted all vectors from namespace: '{ns or 'default'}'")
            except Exception as e:
                print(f"  Warning deleting namespace {ns}: {e}")
        
        print(f"  Flushed {index_name} successfully")
    except Exception as e:
        print(f"  Error flushing {index_name}: {e}")

def load_domain_docs():
    """Load domain documents from data/domains folder."""
    domains_dir = os.path.join(os.path.dirname(__file__), "data", "domains")
    docs = []
    
    for filename in os.listdir(domains_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(domains_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract domain name from filename (e.g., inventory_domain.txt -> inventory)
            domain = filename.replace("_domain.txt", "")
            docs.append({
                "domain": domain,
                "filename": filename,
                "content": content,
            })
            print(f"  Loaded: {filename} ({len(content)} chars)")
    
    return docs

def chunk_document(content: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split document into overlapping chunks."""
    chunks = []
    lines = content.split("\n")
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line)
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            # Keep last few lines for overlap
            overlap_lines = []
            overlap_size = 0
            for prev_line in reversed(current_chunk):
                if overlap_size + len(prev_line) < overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_size += len(prev_line)
                else:
                    break
            current_chunk = overlap_lines
            current_size = overlap_size
        
        current_chunk.append(line)
        current_size += line_size
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

def store_domain_docs_in_pinecone(docs: list[dict]):
    """Store domain documents in rag-base index with domain-specific namespaces."""
    print(f"\nStoring domain docs in {RAG_INDEX}...")
    index = pc.Index(RAG_INDEX)
    
    for doc in docs:
        domain = doc["domain"]
        content = doc["content"]
        chunks = chunk_document(content)
        
        print(f"  Processing {domain}: {len(chunks)} chunks")
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{domain}-chunk-{i:03d}"
            embedding = embeddings.embed_query(chunk)
            
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "domain": domain,
                    "chunk_index": i,
                    "text": chunk[:1000],  # Store first 1000 chars in metadata
                    "source": doc["filename"],
                }
            })
        
        # Upsert to domain-specific namespace
        index.upsert(vectors=vectors_to_upsert, namespace=domain)
        print(f"    Upserted {len(vectors_to_upsert)} vectors to namespace '{domain}'")

def get_schema_descriptions():
    """Get schema descriptions from PostgreSQL."""
    from src.core.database import SessionLocal
    from sqlalchemy import text
    
    with SessionLocal() as session:
        result = session.execute(text("""
            SELECT 
                id, table_name, column_name, domain, description, data_type
            FROM schema_description 
            ORDER BY domain, table_name, column_name
        """))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]

def store_schema_in_pinecone():
    """Store schema metadata in sql-base index."""
    print(f"\nStoring schema metadata in {SQL_INDEX}...")
    
    try:
        schemas = get_schema_descriptions()
        print(f"  Found {len(schemas)} schema descriptions")
    except Exception as e:
        print(f"  Error loading schema from DB: {e}")
        print("  Make sure Docker is running and database is initialized")
        return
    
    index = pc.Index(SQL_INDEX)
    vectors_to_upsert = []
    
    for schema in schemas:
        # Build rich text for embedding
        text_parts = [f"Table: {schema['table_name']}"]
        if schema.get("column_name"):
            text_parts.append(f"Column: {schema['column_name']}")
        if schema.get("data_type"):
            text_parts.append(f"Type: {schema['data_type']}")
        text_parts.append(f"Domain: {schema['domain']}")
        text_parts.append(f"Description: {schema['description']}")
        
        full_text = " | ".join(text_parts)
        embedding = embeddings.embed_query(full_text)
        
        vector_id = f"schema-{schema['id']}"
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "table_name": schema["table_name"],
                "column_name": schema.get("column_name") or "",
                "domain": schema["domain"],
                "description": schema["description"],
                "data_type": schema.get("data_type") or "",
                "text": full_text,
            }
        })
    
    # Upsert all schema vectors to sql-base (default namespace)
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace="schema")
        print(f"  Upserted {len(vectors_to_upsert)} schema vectors to namespace 'schema'")

def verify_setup():
    """Verify the setup by checking index stats."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    for idx_name in [RAG_INDEX, SQL_INDEX]:
        try:
            index = pc.Index(idx_name)
            stats = index.describe_index_stats()
            print(f"\n{idx_name}:")
            print(f"  Total vectors: {stats.total_vector_count}")
            if stats.namespaces:
                for ns, ns_stats in stats.namespaces.items():
                    print(f"  Namespace '{ns or 'default'}': {ns_stats.vector_count} vectors")
        except Exception as e:
            print(f"\n{idx_name}: Error - {e}")

def test_query():
    """Test a sample query against the indexes."""
    print("\n" + "=" * 60)
    print("Test Query")
    print("=" * 60)
    
    test_query = "How do I check inventory levels for a product?"
    print(f"\nQuery: {test_query}")
    
    query_embedding = embeddings.embed_query(test_query)
    
    # Test RAG index
    try:
        rag_index = pc.Index(RAG_INDEX)
        results = rag_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace="inventory"
        )
        print(f"\nRAG Results (inventory namespace):")
        for match in results.matches:
            print(f"  Score: {match.score:.3f} - {match.metadata.get('text', '')[:80]}...")
    except Exception as e:
        print(f"RAG query error: {e}")
    
    # Test SQL index
    try:
        sql_index = pc.Index(SQL_INDEX)
        results = sql_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace="schema"
        )
        print(f"\nSQL Schema Results:")
        for match in results.matches:
            print(f"  Score: {match.score:.3f} - {match.metadata.get('table_name')}.{match.metadata.get('column_name', 'TABLE')}")
    except Exception as e:
        print(f"SQL query error: {e}")

def main():
    print("\nStep 1: Ensure indexes exist")
    ensure_index_exists(RAG_INDEX)
    ensure_index_exists(SQL_INDEX)
    
    print("\nStep 2: Flush existing data")
    flush_index(RAG_INDEX)
    flush_index(SQL_INDEX)
    
    # Wait for flush to propagate
    time.sleep(3)
    
    print("\nStep 3: Load domain documents")
    docs = load_domain_docs()
    
    print("\nStep 4: Store domain docs in Pinecone (RAG)")
    store_domain_docs_in_pinecone(docs)
    
    print("\nStep 5: Store schema metadata in Pinecone (SQL)")
    store_schema_in_pinecone()
    
    # Wait for upserts to propagate
    time.sleep(3)
    
    verify_setup()
    test_query()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
