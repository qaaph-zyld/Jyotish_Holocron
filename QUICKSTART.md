# Jyotish Holocron — Quick Start (5 minutes)

## Step 1: Install Ollama

Download from [https://ollama.ai](https://ollama.ai) and install.

Then pull a model:

```powershell
# Recommended (needs ~8GB RAM)
ollama pull qwen2.5:7b

# Lighter alternative (needs ~4GB RAM)
ollama pull qwen2.5:3b
```

Make sure Ollama is running:

```powershell
ollama serve
```

## Step 2: Set Up Python Environment

```powershell
cd jyotish-holocron
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Add Your PDF

Place your Vedic Astrology PDF in this folder. Then open `ingest.py` and update the path:

```python
PDF_PATH = "your_book.pdf"  # Change this to your file name
```

## Step 4: Ingest the Book

```powershell
py ingest.py
```

This takes 2-5 minutes depending on book size. You'll see:

```
✅ Indexed 512 pages into ./chroma_jyotish / jyotish_book
```

## Step 5: Start Chatting

```powershell
py chat.py
```

Ask anything:

```
Q> What are the significations of Venus in the 7th house?
Q> Explain the concept of yogas in Vedic astrology.
Q> How is the Vimshottari dasha period calculated?
```

Type `exit` to quit.

## Adding More Books

Just change `PDF_PATH` in `ingest.py` and run it again. New content is added to the same knowledge base.

## Changing the Model

Edit the `MODEL` variable in `chat.py`:

```python
MODEL = "qwen2.5:3b"   # Lighter, faster
MODEL = "qwen2.5:7b"   # More accurate
MODEL = "llama3:8b"     # Alternative
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No text extracted" | Your PDF might be scanned. Use OCR software first. |
| Slow responses | Use a smaller model (`qwen2.5:3b`) |
| Ollama crashes | Reduce `similarity_top_k` in `chat.py` to 2 |
| Wrong answers | Increase `similarity_top_k` to 5-6 |
