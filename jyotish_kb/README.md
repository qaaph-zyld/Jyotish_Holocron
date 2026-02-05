# Jyotish Knowledge Base

A **Windows-friendly, fully local, open-source RAG system** for chatting with Vedic Astrology (Jyotish) textbooks. Built with LlamaIndex, ChromaDB, and Ollama - no paid APIs required.

## Features

- **100% Local**: Runs entirely on your machine with Ollama
- **Smart Chunking**: Markdown-aware splitting preserves header hierarchy
- **Incremental Updates**: Only re-processes changed files
- **CLI + Web UI**: Interactive REPL and optional Streamlit interface
- **Grounded Answers**: Citations with source tracking, no hallucinations
- **Windows Optimized**: Tested on Windows 10/11

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Ollama installed from https://ollama.com

### 2. Install Ollama Models

Open PowerShell/CMD and run:

```powershell
# Start Ollama server (keep this running)
ollama serve

# In another terminal, pull the models
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### 3. Setup Project

```powershell
# Clone or navigate to the project
cd jyotish_kb

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Or activate (CMD)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Your Data

Place your markdown file(s) in the `data/` folder:

```powershell
copy ..\vedic_astro_textbook.md data\
```

Or edit `config.yaml` to point to your file:

```yaml
data_path: "D:/Project/Tools/Jyotish_holocron/vedic_astro_textbook.md"
```

### 5. Run Ingestion

```powershell
python scripts/ingest.py
```

First run creates the vector database (~5-15 minutes for 500+ pages).

### 6. Start Chatting

**CLI Mode:**

```powershell
python scripts/chat_cli.py
```

**Web UI Mode:**

```powershell
streamlit run scripts/app_streamlit.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Ollama settings
ollama_base_url: "http://127.0.0.1:11434"

# Models
llm_model: "llama3.1:8b"       # Chat model
embed_model: "nomic-embed-text" # Embeddings model

# Paths
data_path: "./data"              # Markdown file or folder
chroma_dir: "./chroma_db"        # Vector DB storage
collection_name: "jyotish_kb"

# Chunking
chunk_size_tokens: 900
chunk_overlap_tokens: 100

# Retrieval
top_k: 6
answer_style: "detailed"  # or "concise"
reliability_threshold: 0.7
```

## CLI Commands

Inside the chat CLI (`chat_cli.py`):

```
> What are Saturn's aspects?
  (shows answer with citations)

> :sources
  (reprint last sources)

> :open 2
  (show full text of source #2)

> :set top_k=10
  (change retrieval count)

> :set style=concise
  (change answer style)

> :config
  (show current settings)

> :exit
  (quit)
```

## Ingestion Options

```powershell
# Normal (incremental) - only processes changed files
python scripts/ingest.py

# Rebuild entire database
python scripts/ingest.py --rebuild

# Verbose logging
python scripts/ingest.py --verbose

# Custom config
python scripts/ingest.py --config my_config.yaml
```

## Project Structure

```
jyotish_kb/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Your markdown files
│   └── vedic_astro_textbook.md
├── chroma_db/              # Persistent vector store (auto-created)
│   ├── chroma.sqlite3
│   └── manifest.json
└── scripts/
    ├── ingest.py           # Ingestion pipeline
    ├── chat_cli.py         # CLI chat interface
    ├── app_streamlit.py    # Web UI (optional)
    ├── utils.py            # Shared utilities
    └── split_md_by_h1.py   # Utility to split large MD files
```

## How It Works

### 1. Markdown-Aware Parsing

Uses `MarkdownNodeParser` to split by headers:

```
# Grahas                    → header_path: "Grahas"
## Saturn                   → header_path: "Grahas > Saturn"
### Aspects                 → header_path: "Grahas > Saturn > Aspects"
```

### 2. Smart Chunking

Two-stage process:
1. **MarkdownNodeParser**: Splits at headers, preserves structure
2. **SentenceSplitter**: Further splits oversized sections

All chunks keep their `header_path` metadata for precise retrieval.

### 3. Incremental Updates

Tracks file hashes in `manifest.json`:
- **Unchanged file**: Skipped (fast!)
- **Modified file**: Old chunks deleted, re-ingested
- **New file**: Fully processed

### 4. Retrieval & Answering

- Similarity search returns top-K chunks
- Metadata (header_path) shown in citations
- Reliability threshold prevents hallucinations

## Troubleshooting

### "Cannot connect to Ollama"

```powershell
# Check if Ollama is running
ollama serve

# Verify in another terminal
ollama list
```

### "Model not found"

```powershell
# Pull the required model
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### "No markdown files found"

Check your `config.yaml`:

```yaml
# For a single file
data_path: "./data/vedic_astro_textbook.md"

# For a folder
data_path: "./data"

# Use absolute paths if needed
data_path: "D:/Project/Tools/Jyotish_holocron/data"
```

### Permission Errors (Windows)

Run PowerShell as Administrator, or:

```powershell
# Fix execution policy (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or bypass for this session
powershell -ExecutionPolicy Bypass -Command "python scripts/ingest.py"
```

### Slow Ingestion

For very large files (500+ pages):
- Ensure SSD storage for `chroma_dir`
- Close other applications
- Use `--verbose` to see progress
- Consider splitting with `split_md_by_h1.py` first

### "Module not found" errors

```powershell
# Ensure you're in the virtual environment
.venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## Advanced Usage

### Split Large Files by H1 Headers

If your 500-page book has logical sections:

```powershell
python scripts/split_md_by_h1.py \
    --input data/vedic_astro_textbook.md \
    --output data/split/
```

This creates separate files per chapter, improving:
- Incremental updates (only changed chapters re-ingested)
- Source attribution (clearer file names)

### Custom Models

Edit `config.yaml` to use different Ollama models:

```yaml
# More capable (slower, more RAM)
llm_model: "llama3.1:70b"

# Faster (less accurate)
llm_model: "phi3:mini"

# Different embeddings
embed_model: "mxbai-embed-large"
```

Pull them first:

```powershell
ollama pull llama3.1:70b
ollama pull mxbai-embed-large
```

### Multiple Collections

Create separate knowledge bases:

```yaml
# config_classical.yaml
collection_name: "jyotish_classical"
data_path: "./data/classical_texts"
chroma_dir: "./chroma_db_classical"

# config_modern.yaml
collection_name: "jyotish_modern"
data_path: "./data/modern_texts"
chroma_dir: "./chroma_db_modern"
```

Run with:

```powershell
python scripts/ingest.py --config config_classical.yaml
python scripts/chat_cli.py --config config_classical.yaml
```

## Performance Tips

1. **SSD Storage**: ChromaDB performs much better on SSD vs HDD
2. **RAM**: 8GB+ recommended for 8B models, 16GB+ for larger
3. **Model Size**: 8B models work well; 70B for deeper analysis
4. **Chunk Size**: Reduce `chunk_size_tokens` if retrieval is slow
5. **Top-K**: Lower `top_k` (e.g., 4) for faster responses

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Markdown Files │────▶│  MarkdownNode    │────▶│  Text Chunks    │
│  (500 pages)    │     │  Parser          │     │  (w/ headers)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                           ┌──────────────────────────────┘
                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  Ollama Embed    │────▶│  ChromaDB       │
│  (CLI/Web)      │     │  (nomic-embed)   │     │  (vector store) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                           ┌──────────────────────────────┘
                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Cited Answer   │◄────│  Ollama LLM      │◄────│  Top-K Chunks   │
│  (with sources) │     │  (llama3.1:8b)   │     │  + metadata     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Secret Sauce 🎯

The quality of RAG depends on **consistent section templates** in your source material:

```markdown
# Graha Name
## Karakatva (Significations)
## Dignity (Strengths)
## Aspects (Drishti)
## Special Yogas
## Classical References
## Remedial Measures
```

When every section follows this pattern, `MarkdownNodeParser` automatically creates chunks with rich `header_path` metadata like `"Grahas > Saturn > Aspects"`, enabling precise retrieval.

## License

MIT License - Feel free to use, modify, and share.

## Credits

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) for RAG pipeline
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.com/) for local LLMs
- [Streamlit](https://streamlit.io/) for web UI (optional)

---

**Jaya Jyotish!** 🕉️
