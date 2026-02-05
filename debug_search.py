import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DB_DIR = "./chroma_jyotish"
COLLECTION = "jyotish_book"

# Use same embedding model as chat
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def debug_search():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    
    print(f"Collection stats: {collection.count()} documents")
    
    # Test different spellings
    queries = [
        "Mrigasira",
        "Mrigashirsha", 
        "Mrigashira",
        "mrigasira",
        "mrigashirsha",
        "nakshatra",
        "yoga"
    ]
    
    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Get embedding
        query_embedding = embed_model.get_text_embedding(query)
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                print(f"  Result {i+1} (dist={dist:.3f}, page={meta.get('page', '?')}):")
                # Show snippet around potential match
                text_lower = doc.lower()
                query_lower = query.lower()
                if query_lower in text_lower:
                    start = max(0, text_lower.find(query_lower) - 50)
                    end = min(len(doc), text_lower.find(query_lower) + 50 + len(query))
                    snippet = doc[start:end].replace('\n', ' ')
                    print(f"    MATCH: ...{snippet}...")
                else:
                    # Show first 100 chars
                    snippet = doc[:100].replace('\n', ' ')
                    print(f"    {snippet}...")
        else:
            print("  No results")

if __name__ == "__main__":
    debug_search()
