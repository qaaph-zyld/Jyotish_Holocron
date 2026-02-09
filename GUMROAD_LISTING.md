# Gumroad Product Listing — Jyotish Holocron

## Product Name
**Jyotish Holocron — Local AI for Vedic Astrology Books**

## Price
$29 (one-time purchase)

## Short Description
Query any Vedic Astrology PDF with AI — get answers with page citations. 100% local, 100% private. No cloud APIs, no subscriptions. Uses Ollama + ChromaDB + LlamaIndex.

## Full Description

### What is Jyotish Holocron?

A **local RAG (Retrieval-Augmented Generation) system** purpose-built for Vedic Astrology practitioners and students. Feed it any Jyotish PDF textbook and ask questions in plain English — it responds with accurate answers citing the exact page numbers.

**Everything runs on your machine.** No OpenAI. No cloud. No data leaves your computer.

### Why You Need This

- You have Jyotish books but **can't quickly find specific topics**
- You want to **cross-reference concepts** across hundreds of pages
- You need **page-accurate citations** for study or client consultations
- You want AI assistance **without sending sacred texts to the cloud**

### How It Works

```
Your PDF → Ingestion → Vector Database (ChromaDB)
Your Question → AI Search → Top Matches → Ollama LLM → Answer + Page Numbers
```

1. **Ingest**: Drop your PDF in the folder, run `python ingest.py`
2. **Chat**: Run `python chat.py` and ask anything
3. **Get answers**: AI responds using ONLY your book content with page citations

### Example Queries

```
Q> Explain arudha padas and how to use them.
A> Arudha padas are the reflection of houses... (pages 142-145)

Q> How do you calculate the ascendant?
A> The ascendant (Lagna) is calculated by... (page 23)

Q> What are the significations of the 7th house?
A> The 7th house governs marriage, partnerships... (pages 67-69)
```

### What's Included

```
jyotish-holocron/
├── ingest.py              # PDF ingestion pipeline
├── chat.py                # Interactive chat interface
├── chat_improved.py       # Enhanced chat with better formatting
├── requirements.txt       # Python dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # 5-minute setup guide
└── LICENSE                # MIT License
```

### Technical Stack
- **LLM**: Ollama (qwen2.5:7b recommended, 3b for lighter hardware)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB (persistent, no server needed)
- **Framework**: LlamaIndex
- **Language**: Python 3.8+

### Requirements
- Python 3.8+
- Ollama installed ([ollama.ai](https://ollama.ai))
- 8GB+ RAM recommended (4GB minimum with smaller models)
- Your own Vedic Astrology PDF(s)

### Key Features
- **Page citations** in every answer
- **Grounded responses** — only answers from your book content
- **Multi-book support** — ingest multiple PDFs into the same knowledge base
- **Configurable** — change chunk size, overlap, model, retrieval count
- **No hallucinations** — system prompt forces book-only answers
- **Runs offline** — zero internet needed after initial setup

## Tags
vedic-astrology, jyotish, rag, ai, ollama, local-llm, knowledge-base, pdf-chat, chromadb, llama-index

## Category
Software / AI Tools / Education
