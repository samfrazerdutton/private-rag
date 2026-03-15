from sentence_transformers import SentenceTransformer
import torch

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"[Embedder] Ready. No data leaves your machine.")

    def embed(self, texts: list[str]) -> list:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
