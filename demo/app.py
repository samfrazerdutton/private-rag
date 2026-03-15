import gradio as gr
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.rag_pipeline import PrivateRAGPipeline

pipeline = None

def load_pipeline():
    global pipeline
    pipeline = PrivateRAGPipeline()
    return "✓ Pipeline ready. Upload a PDF to get started."

def upload_doc(file):
    if pipeline is None:
        return "Click 'Initialize' first."
    result = pipeline.add_document(file.name)
    return f"✓ {result}"

def ask_question(question):
    if pipeline is None:
        return "Click 'Initialize' first.", ""
    if not question.strip():
        return "Please enter a question.", ""
    result = pipeline.answer(question)
    context_display = "\n\n---\n\n".join(result["context"])
    return result["answer"], context_display

with gr.Blocks(title="Private RAG") as demo:
    gr.Markdown("""
    # Private RAG System
    **100% local. No API keys. No data leaves your machine.**
    Upload any PDF and ask questions about it.
    """)

    init_btn = gr.Button("Initialize Pipeline", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Add Document")
            upload_status = gr.Textbox(label="Upload status", interactive=False)
            question = gr.Textbox(label="Ask a question", lines=2)
            ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            answer = gr.Textbox(label="Answer", lines=8)
            context = gr.Textbox(label="Retrieved context", lines=8)

    init_btn.click(load_pipeline, outputs=status)
    upload_btn.click(upload_doc, inputs=upload, outputs=upload_status)
    ask_btn.click(ask_question, inputs=question, outputs=[answer, context])

if __name__ == "__main__":
    demo.launch(share=True)
