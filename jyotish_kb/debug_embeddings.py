"""Debug script to check embeddings in ChromaDB"""
import chromadb
import numpy as np

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="jyotish_kb")

print(f"Collection name: {collection.name}")
print(f"Collection count: {collection.count()}")

# Get a few samples
results = collection.get(limit=3, include=['embeddings', 'documents', 'metadatas'])

print(f"\nSample entries:")
for i in range(min(3, len(results['ids']))):
    print(f"\n[{i+1}] ID: {results['ids'][i]}")
    print(f"    Metadata: {results['metadatas'][i]}")
    print(f"    Document: {results['documents'][i][:100]}...")
    
    if results['embeddings'] and results['embeddings'][i]:
        emb = np.array(results['embeddings'][i])
        print(f"    Embedding shape: {emb.shape}")
        print(f"    Embedding norm: {np.linalg.norm(emb):.4f}")
        print(f"    First 5 values: {emb[:5]}")
    else:
        print(f"    Embedding: NONE/NULL")

# Test a query
print(f"\n\nTesting query...")
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://127.0.0.1:11434")
query_text = "Saturn aspects"
query_embedding = embed_model.get_text_embedding(query_text)

print(f"Query: {query_text}")
print(f"Query embedding shape: {len(query_embedding)}")
print(f"Query embedding norm: {np.linalg.norm(query_embedding):.4f}")
print(f"First 5 values: {query_embedding[:5]}")

# Try manual query
try:
    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['documents', 'distances', 'metadatas']
    )
    
    print(f"\n\nQuery results:")
    for i in range(len(query_results['ids'][0])):
        print(f"\n[{i+1}] Distance: {query_results['distances'][0][i]:.4f}")
        print(f"    Metadata: {query_results['metadatas'][0][i]}")
        print(f"    Document: {query_results['documents'][0][i][:100]}...")
except Exception as e:
    print(f"Query failed: {e}")
