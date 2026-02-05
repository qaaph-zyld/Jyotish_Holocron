# Jyotish Holocron - Local RAG System

A local RAG (Retrieval-Augmented Generation) system for querying Vedic Astrology books with page citations. No cloud API needed—everything runs on your machine using Ollama.

## Features

* **Local LLM**: Uses Ollama (qwen2.5:7b or your choice)
* **Local Embeddings**: sentence-transformers/all-MiniLM-L6-v2
* **Vector Database**: ChromaDB (persistent, no server needed)
* **Page Citations**: Answers include source page numbers
* **Grounded Responses**: Only answers from book content, no hallucinations

## Setup

### 1. Install Ollama

Download and install [Ollama](https://ollama.ai/) for Windows.

Pull a model (choose based on your hardware):

```powershell
ollama pull qwen2.5:7b
```

For weaker hardware, try `qwen2.5:3b` or `qwen2.5:1.5b`.

Verify Ollama is running:

```powershell
ollama serve
```

### 2. Create Python Environment

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add Your PDF

Place your Vedic Astrology PDF in the project root and name it `vedic_astro_textbook.pdf` (or update `PDF_PATH` in `ingest.py`).

### 4. Ingest the Book

```powershell
py ingest.py
```

This creates a `chroma_jyotish/` folder with your vector database. Takes a few minutes for a 512-page book.

### 5. Start Chatting

```powershell
py chat.py
```

## Example Queries

```
Q> Explain arudha padas and how to use them.
Q> What are the 4 pillars of Vedic astrology in this book?
Q> Give the definition of nakshatra and its length.
Q> How do you calculate the ascendant?
```

Type `exit` or `quit` to stop.

## Adding More Books

To add additional PDFs:

1. Update `PDF_PATH` in `ingest.py` to point to the new PDF
2. Run `py ingest.py` again

The new content will be added to the same ChromaDB collection.

## Upgrades (Future)

1. **Better Embeddings**: Swap to `BAAI/bge-large-en-v1.5` for technical terms
2. **Reranking**: Add a reranker for laser-accurate retrieval
3. **Multi-Model**: Test different Ollama models for comparison

## Configuration

### Chunking Parameters

In `ingest.py`:

```python
SentenceSplitter(chunk_size=900, chunk_overlap=120)
```

Adjust for your content density.

### Retrieval Settings

In `chat.py`:

```python
similarity_top_k=6  # Number of chunks to retrieve
response_mode="compact"  # How chunks are combined
```

### Change Model

Both files use:

```python
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)
```

Change `model=` to any Ollama model you've pulled.

## Troubleshooting

**No text extracted**: Your PDF might be scanned. Use OCR first.

**Slow responses**: Try a smaller Ollama model or increase `request_timeout`.

**Wrong answers**: Increase `similarity_top_k` or improve embeddings model.

**Memory issues**: Reduce `chunk_size` or use a smaller LLM.

## Architecture

```
PDF → pypdf → Documents (page-level) 
    → SentenceSplitter → Chunks 
    → HuggingFace Embeddings → Vectors 
    → ChromaDB (persistent storage)

Query → Embed → ChromaDB Search → Top-K Chunks 
      → Ollama LLM → Answer with Citations
```

## License

Use responsibly. Ensure you have rights to the PDFs you ingest.
