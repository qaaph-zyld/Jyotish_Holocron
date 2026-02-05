from pypdf import PdfReader
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PDF_PATH = "vedic_astro_textbook.pdf"
DB_DIR = "./chroma_jyotish"
COLLECTION = "jyotish_book"

Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.transformations = [SentenceSplitter(chunk_size=900, chunk_overlap=120)]

def load_pdf_as_documents(path: str) -> list[Document]:
    pdf = PdfReader(path)
    docs: list[Document] = []
    for page_num, page in enumerate(pdf.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, metadata={"source": path, "page": page_num}))
    return docs

def main():
    docs = load_pdf_as_documents(PDF_PATH)
    if not docs:
        raise RuntimeError("No text extracted. If this is a scanned PDF, OCR it first.")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print(f"✅ Indexed {len(docs)} pages into {DB_DIR} / {COLLECTION}")

if __name__ == "__main__":
    main()
