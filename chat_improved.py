import chromadb
import sys
import re

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DB_DIR = "./chroma_jyotish"
COLLECTION = "jyotish_book"
MODEL = "qwen2.5:3b"

Settings.llm = Ollama(
    model=MODEL, 
    request_timeout=180.0,
    context_window=4096,
    num_ctx=4096
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def hybrid_search(query, collection, embed_model, top_k=5):
    """Combine semantic search with keyword search for better results"""
    
    # 1. Semantic search
    query_embedding = embed_model.get_text_embedding(query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 2. Keyword search (case-insensitive)
    all_docs = collection.get(include=["documents", "metadatas"])
    keyword_matches = []
    
    query_lower = query.lower()
    for i, (doc, meta) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
        if query_lower in doc.lower():
            keyword_matches.append((doc, meta, 0.0))  # Perfect match score
    
    # 3. Combine results (keyword matches first, then semantic)
    combined_docs = []
    combined_metas = []
    
    # Add keyword matches
    for doc, meta, _ in keyword_matches:
        combined_docs.append(doc)
        combined_metas.append(meta)
    
    # Add semantic results that aren't duplicates
    seen_pages = {meta.get('page') for doc, meta, _ in keyword_matches}
    for doc, meta, dist in zip(
        semantic_results['documents'][0],
        semantic_results['metadatas'][0], 
        semantic_results['distances'][0]
    ):
        if meta.get('page') not in seen_pages:
            combined_docs.append(doc)
            combined_metas.append(meta)
            if len(combined_docs) >= top_k:
                break
    
    return combined_docs, combined_metas

def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    system_rules = (
        "You are a Jyotish assistant. Use ONLY the provided book excerpts.\n"
        "If the answer is not supported by the excerpts, say: Not found in the book.\n"
        "Always include page numbers from the excerpts when you answer.\n"
        "Keep answers concise.\n"
    )

    print(f"Using model: {MODEL}")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            q = input("Q> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
                
            print("Searching...")
            
            # Try hybrid search first
            docs, metas = hybrid_search(q, collection, Settings.embed_model)
            
            if docs:
                # Create a custom context from our results
                context = "\n\n".join([f"[Page {meta.get('page', '?')}] {doc}" for doc, meta in zip(docs, metas)])
                
                # Simple prompt with our context
                prompt = f"""{system_rules}

Context from book:
{context}

Question: {q}

Answer:"""
                
                resp = Settings.llm.complete(prompt)
                print(f"\nA> {resp.text}\n")
            else:
                print("\nA> Not found in the book.\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            error_msg = str(e).lower()
            if "exit status" in error_msg or "terminated" in error_msg:
                print(f"\n❌ Ollama crashed. Try one of these fixes:")
                print(f"   1. Use an even smaller model: ollama pull qwen2.5:1.5b")
                print(f"   2. Restart Ollama: ollama serve")
                print(f"   3. Check available memory\n")
                sys.exit(1)
            else:
                print(f"\n⚠️  Error: {e}\n")
                continue

if __name__ == "__main__":
    main()
