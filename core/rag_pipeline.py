from core.vector_store import PrivateVectorStore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class PrivateRAGPipeline:
    def __init__(self, model_name="microsoft/phi-2"):
        self.store = PrivateVectorStore()

        print(f"[RAG] Loading LLM: {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("[RAG] Pipeline ready. Fully private — no API calls.")

    def add_document(self, file_path: str) -> str:
        chunks = self.store.add_document(file_path)
        return f"Added {chunks} chunks from {file_path}"

    def answer(self, question: str) -> dict:
        # Retrieve relevant chunks
        context_chunks = self.store.query(question)
        if not context_chunks:
            return {"answer": "No documents loaded yet.", "context": []}

        context = "\n\n".join(context_chunks)

        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return {"answer": answer, "context": context_chunks}
