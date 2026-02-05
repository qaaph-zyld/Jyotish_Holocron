import chromadb
import sys

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

def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    qe = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )

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
            resp = qe.query(system_rules + "\nQuestion: " + q)
            print(f"\nA> {resp}\n")
            
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
