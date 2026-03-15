import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from core.embedder import LocalEmbedder
import os

class PrivateVectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.embedder = LocalEmbedder()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="private_docs",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] ChromaDB ready. Documents: {self.collection.count()}")

    def add_document(self, file_path: str):
        print(f"[VectorStore] Loading: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        texts = [c.page_content for c in chunks]
        ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.embed(texts)
        self.collection.upsert(documents=texts, embeddings=embeddings, ids=ids)
        print(f"[VectorStore] Added {len(chunks)} chunks from {os.path.basename(file_path)}")
        return len(chunks)

    def query(self, question: str, n_results: int = 4) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.embedder.embed([question])
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, count)
        )
        return results["documents"][0] if results["documents"] else []
