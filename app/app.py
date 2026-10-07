"""Gradio UI for kalman-rag. Offline it shows retrieved+cited chunks; with API
keys set it shows Claude's grounded answer. Run: python app/app.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
import gradio as gr
from corpus import CORPUS
from kalman_rag.retrieve import chunk_documents, HybridRetriever, grounded_answer

retriever = HybridRetriever(chunk_documents(CORPUS))


def ask(question):
    retrieved = retriever.retrieve(question, k=5)
    sources = "\n".join(f"- [{c.source} p.{c.page}] (score {s:.3f})" for c, s in retrieved)
    if os.environ.get("ANTHROPIC_API_KEY"):
        from kalman_rag.answer import answer, to_candidates
        return answer(question, to_candidates(retrieved)), sources
    return grounded_answer(question, retrieved), sources


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask a control-theory question",
                      value="derive the Kalman gain"),
    outputs=[gr.Textbox(label="Grounded context / answer"),
             gr.Textbox(label="Sources")],
    title="kalman-rag",
    description="Hybrid RAG over control-theory notes. Every answer is cited.")

if __name__ == "__main__":
    demo.launch()
