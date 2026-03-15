# Private RAG System

100% local retrieval-augmented generation. No API keys, no data leaves your machine.

## What it does
- Upload any PDF
- Documents are chunked and embedded locally using sentence-transformers
- Embeddings stored in ChromaDB on your machine
- Questions answered by local Phi-2 (4-bit) using only your documents

## Stack
Sentence Transformers · ChromaDB · LangChain · Phi-2 · Gradio

## Run it
```bash
pip install -r requirements.txt
python demo/app.py
```
