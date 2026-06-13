"""
PharmaDocs AI — Pharmaceutical Document Intelligence
Author: Sravani Bhamidipaty
"""

import os
import time
import numpy as np
import pandas as pd
import fitz
import faiss
import gradio as gr

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K_RETRIEVE = 8
TOP_K_RERANK = 4

DOC_TYPES = [
    "cover_letter", "certificate_of_quality", "packaging_specification",
    "bse_tse_declaration", "material_description", "supplier_qualification",
    "chain_of_custody", "unknown",
]


@dataclass
class Chunk:
    text: str
    doc_type: str
    page_start: int
    page_end: int
    source_file: str
    chunk_index: int
    word_count: int = 0


@dataclass
class RetrievalResult:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    rerank_score: float
    final_score: float


@dataclass
class QueryResult:
    query: str
    answer: str
    sources: List[str]
    confidence: float
    chunks_used: int
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)


class DocumentProcessor:
    def __init__(self):
        self._ocr_available = self._check_ocr()

    def _check_ocr(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_pages(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            is_scanned = len(text) < 50
            if is_scanned and self._ocr_available:
                text = self._ocr_page(page)
            pages.append({"page_num": i, "text": text, "is_scanned": is_scanned, "char_count": len(text)})
        doc.close()
        return pages

    def _ocr_page(self, page) -> str:
        import pytesseract
        from PIL import Image
        import cv2
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return pytesseract.image_to_string(Image.fromarray(thresh), config="--psm 6")


class DocumentClassifier:
    RULES = {
        "certificate_of_quality": ["certificate of quality", "lot number", "expiration date", "certificate no", "lot no", "manufactured by", "product release"],
        "packaging_specification": ["packaging", "blister", "carton", "pkg-spec", "packaging component", "tray", "lid film", "secondary carton"],
        "bse_tse_declaration":     ["bse", "tse", "transmissible spongiform", "bovine", "spongiform encephalopathy", "animal origin"],
        "material_description":    ["material description", "raw material", "chemical name", "cas number", "molecular weight", "material spec"],
        "supplier_qualification":  ["supplier qualification", "vendor qualification", "supplier assessment", "approved supplier", "vendor audit"],
        "chain_of_custody":        ["chain of custody", "custody transfer", "traceability", "lot traceability", "chain of custody record"],
        "cover_letter":            ["dear", "sincerely", "regards", "to whom it may concern", "please find enclosed", "attached please"],
    }

    def classify(self, text: str) -> str:
        t = text.lower()
        scores = {dt: sum(1 for kw in kws if kw in t) for dt, kws in self.RULES.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"


class Chunker:
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict], source_file: str) -> List[Chunk]:
        chunks, idx = [], 0
        classifier = DocumentClassifier()
        for page in pages:
            if not page["text"].strip():
                continue
            doc_type = classifier.classify(page["text"])
            words = page["text"].split()
            step = max(1, self.chunk_size - self.overlap)
            for j in range(0, max(1, len(words)), step):
                sub = " ".join(words[j:j + self.chunk_size])
                if not sub.strip():
                    continue
                chunks.append(Chunk(text=sub, doc_type=doc_type, page_start=page["page_num"],
                                    page_end=page["page_num"], source_file=os.path.basename(source_file),
                                    chunk_index=idx, word_count=len(sub.split())))
                idx += 1
        return chunks


class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.chunks: List[Chunk] = []
        self.faiss_index = None
        self.bm25 = None

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=32).astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE, filter_doc_type: Optional[str] = None) -> List[RetrievalResult]:
        if not self.chunks or self.faiss_index is None:
            return []
        q_emb = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        vec_scores, vec_ids = self.faiss_index.search(q_emb, min(k * 2, len(self.chunks)))
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_ids = np.argsort(bm25_scores)[::-1][:k * 2]
        candidate_ids = set(vec_ids[0].tolist()) | set(bm25_top_ids.tolist())
        candidate_ids = {i for i in candidate_ids if 0 <= i < len(self.chunks)}
        if filter_doc_type and filter_doc_type != "Auto":
            ft = filter_doc_type.lower().replace(" ", "_")
            candidate_ids = {i for i in candidate_ids if self.chunks[i].doc_type == ft}
        if not candidate_ids:
            return []
        candidate_list = list(candidate_ids)
        rerank_scores = self.reranker.predict([(query, self.chunks[i].text) for i in candidate_list])
        vec_score_map = {int(vid): float(vs) for vid, vs in zip(vec_ids[0], vec_scores[0])}
        results = []
        for chunk_id, rerank_score in zip(candidate_list, rerank_scores):
            vs = vec_score_map.get(chunk_id, 0.0)
            bs = float(bm25_scores[chunk_id]) / (max(bm25_scores) + 1e-9)
            final = 0.4 * vs + 0.2 * bs + 0.4 * float(rerank_score) / 10.0
            results.append(RetrievalResult(chunk=self.chunks[chunk_id], vector_score=vs,
                                           bm25_score=bs, rerank_score=float(rerank_score), final_score=final))
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:TOP_K_RERANK]


class PharmaRAGEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = Chunker()
        self.retriever = HybridRetriever()
        self.chunks: List[Chunk] = []
        self.processing_stats = {}

    def ingest(self, pdf_path: str) -> Dict:
        t0 = time.time()
        pages = self.processor.extract_pages(pdf_path)
        chunks = self.chunker.chunk_pages(pages, pdf_path)
        self.chunks = chunks
        self.retriever.index(chunks)
        elapsed = round((time.time() - t0) * 1000, 1)
        type_counts = {}
        for c in chunks:
            type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1
        self.processing_stats = {"pages": len(pages), "chunks": len(chunks),
                                  "scanned_pages": sum(1 for p in pages if p["is_scanned"]),
                                  "doc_types": type_counts, "ingestion_ms": elapsed}
        return self.processing_stats

    def query(self, question: str, filter_doc_type: Optional[str] = None) -> QueryResult:
        t0 = time.time()
        t_ret = time.time()
        results = self.retriever.retrieve(question, filter_doc_type=filter_doc_type)
        retrieval_ms = round((time.time() - t_ret) * 1000, 1)
        if not results:
            return QueryResult(query=question, answer="No relevant information found in the uploaded documents.",
                               sources=[], confidence=0.0, chunks_used=0, latency_ms=0,
                               retrieval_latency_ms=retrieval_ms, generation_latency_ms=0)
        context_parts, sources = [], []
        for r in results:
            header = (f"[Source: {r.chunk.source_file} | Type: {r.chunk.doc_type.replace('_',' ').title()} | "
                      f"Page {r.chunk.page_start + 1} | Relevance: {r.final_score:.3f}]")
            context_parts.append(f"{header}\n{r.chunk.text}")
            src = (f"{r.chunk.source_file} p{r.chunk.page_start + 1} "
                   f"({r.chunk.doc_type.replace('_',' ').title()}) [score: {r.final_score:.3f}]")
            if src not in sources:
                sources.append(src)
        context = "\n\n---\n\n".join(context_parts)
        avg_confidence = sum(r.final_score for r in results) / len(results)
        prompt = f"""You are a precise pharmaceutical document assistant with expertise in regulatory compliance.

STRICT RULES:
1. Answer ONLY using the provided context. Do not use prior knowledge.
2. If the answer is not in the context, say "This information is not available in the provided documents."
3. Always cite which document type and page your answer comes from.
4. Be specific — include exact values, lot numbers, dates when present.

Context:
{context}

Question: {question}

Answer (with source citations):"""
        t_gen = time.time()
        response = client.chat.completions.create(model=LLM_MODEL,
                                                   messages=[{"role": "user", "content": prompt}],
                                                   temperature=0.1, max_tokens=512)
        generation_ms = round((time.time() - t_gen) * 1000, 1)
        answer = response.choices[0].message.content.strip()
        total_ms = round((time.time() - t0) * 1000, 1)
        return QueryResult(query=question, answer=answer, sources=sources, confidence=avg_confidence,
                           chunks_used=len(results), latency_ms=total_ms, retrieval_latency_ms=retrieval_ms,
                           generation_latency_ms=generation_ms, retrieved_chunks=results)


GROUND_TRUTH = [
    {"query": "What sterilization method was used for this product?",
     "expected_keywords": ["autoclave", "gamma irradiation", "sterilization"],
     "expected_doc_type": "certificate_of_quality", "relevant_pages": [1, 2]},
    {"query": "What are the storage conditions specified in the certificate?",
     "expected_keywords": ["+5", "storage", "temperature"],
     "expected_doc_type": "cover_letter", "relevant_pages": [0]},
    {"query": "What test methods were used for quality control?",
     "expected_keywords": ["flow rate", "pressure", "visual inspection", "package integrity"],
     "expected_doc_type": "certificate_of_quality", "relevant_pages": [1, 2]},
    {"query": "What are the part numbers listed in this document?",
     "expected_keywords": ["29477427", "pkg-", "part number"],
     "expected_doc_type": "packaging_specification", "relevant_pages": [3, 4]},
    {"query": "What packaging configuration changes were made?",
     "expected_keywords": ["ecn", "petg", "pvc", "revision", "change"],
     "expected_doc_type": "packaging_specification", "relevant_pages": [3, 4]},
]


class Evaluator:
    def __init__(self, engine: PharmaRAGEngine):
        self.engine = engine

    def recall_at_k(self, retrieved_pages, relevant_pages, k):
        return len(set(retrieved_pages[:k]) & set(relevant_pages)) / len(relevant_pages) if relevant_pages else 0.0

    def mrr(self, retrieved_pages, relevant_pages):
        relevant_set = set(relevant_pages)
        for rank, page in enumerate(retrieved_pages, 1):
            if page in relevant_set:
                return 1.0 / rank
        return 0.0

    def answer_contains_keywords(self, answer, keywords):
        a = answer.lower()
        return sum(1 for kw in keywords if kw.lower() in a) / len(keywords)

    def run(self) -> pd.DataFrame:
        records, r1s, r3s, mrrs, kws, lats = [], [], [], [], [], []
        for gt in GROUND_TRUTH:
            result = self.engine.query(gt["query"])
            pages = [r.chunk.page_start for r in result.retrieved_chunks]
            r1 = self.recall_at_k(pages, gt["relevant_pages"], 1)
            r3 = self.recall_at_k(pages, gt["relevant_pages"], 3)
            mrr = self.mrr(pages, gt["relevant_pages"])
            kw = self.answer_contains_keywords(result.answer, gt["expected_keywords"])
            r1s.append(r1); r3s.append(r3); mrrs.append(mrr); kws.append(kw); lats.append(result.latency_ms)
            records.append({"Query": gt["query"][:60] + "...", "Recall@1": f"{r1:.2f}",
                            "Recall@3": f"{r3:.2f}", "MRR": f"{mrr:.2f}", "Keyword Match": f"{kw:.2f}",
                            "Confidence": f"{result.confidence:.3f}", "Latency (ms)": result.latency_ms,
                            "Chunks Used": result.chunks_used})
        records.append({"Query": "AVERAGE", "Recall@1": f"{np.mean(r1s):.2f}", "Recall@3": f"{np.mean(r3s):.2f}",
                        "MRR": f"{np.mean(mrrs):.2f}", "Keyword Match": f"{np.mean(kws):.2f}",
                        "Confidence": "-", "Latency (ms)": round(np.mean(lats), 1), "Chunks Used": "-"})
        return pd.DataFrame(records)


engine = PharmaRAGEngine()
evaluator = Evaluator(engine)


def ingest_pdf(pdf_file):
    if pdf_file is None:
        return "*No file uploaded*", gr.update(choices=["Auto"])
    stats = engine.ingest(pdf_file.name)
    summary = f"### ✅ {os.path.basename(pdf_file.name)}\n"
    summary += f"- **Pages:** {stats['pages']} ({stats['scanned_pages']} scanned)\n"
    summary += f"- **Chunks:** {stats['chunks']}\n"
    summary += f"- **Ingestion time:** {stats['ingestion_ms']}ms\n"
    summary += "- **Document types detected:**\n"
    for dt, count in stats["doc_types"].items():
        summary += f"  • {dt.replace('_',' ').title()}: {count} chunks\n"
    types = ["Auto"] + [dt.replace("_", " ").title() for dt in stats["doc_types"].keys()]
    return summary, gr.update(choices=types, value="Auto")


def chat(query, history, doc_filter, show_chunks):
    if not engine.chunks:
        return history + [{"role": "user", "content": query},
                          {"role": "assistant", "content": "⚠️ Please upload a PDF first."}], ""
    result = engine.query(query, filter_doc_type=doc_filter)
    response = f"{result.answer}\n\n---\n"
    response += f"📎 **Sources:** {' | '.join(result.sources)}\n"
    response += f"🎯 **Confidence:** {result.confidence:.3f} | 📦 **Chunks:** {result.chunks_used} | "
    response += f"⏱ **Total:** {result.latency_ms}ms (retrieval: {result.retrieval_latency_ms}ms, generation: {result.generation_latency_ms}ms)"
    chunk_details = ""
    if show_chunks and result.retrieved_chunks:
        chunk_details = "### Retrieved Chunks\n"
        for i, r in enumerate(result.retrieved_chunks):
            chunk_details += f"\n**Chunk {i+1}** | {r.chunk.doc_type.replace('_',' ').title()} | p{r.chunk.page_start+1} | "
            chunk_details += f"Vector: {r.vector_score:.3f} | BM25: {r.bm25_score:.3f} | Rerank: {r.rerank_score:.2f} | **Final: {r.final_score:.3f}**\n"
            chunk_details += f"```\n{r.chunk.text[:300]}...\n```\n"
    return history + [{"role": "user", "content": query},
                      {"role": "assistant", "content": response}], chunk_details


def run_eval():
    if not engine.chunks:
        return "⚠️ Please upload a PDF first."
    return evaluator.run().to_markdown(index=False)


with gr.Blocks(title="PharmaDocs AI") as demo:
    gr.Markdown(
        "# 🧬 PharmaDocs AI — Pharmaceutical Document Intelligence\n"
        "*End-to-end RAG pipeline with hybrid retrieval, reranking, and evaluation suite*"
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📂 Document Ingestion")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            doc_info = gr.Markdown("*No document loaded*")
            gr.Markdown("### ⚙️ Retrieval Settings")
            doc_filter = gr.Dropdown(choices=["Auto"], value="Auto", label="Filter by Document Type",
                                     info="Auto routes query to best-matching type")
            show_chunks = gr.Checkbox(label="Show retrieved chunks + scores", value=False)
            gr.Markdown("### 📊 Evaluation")
            eval_btn = gr.Button("▶ Run Evaluation Suite", variant="secondary")
            eval_out = gr.Markdown("*Click to run evaluation*")
            pdf_input.change(fn=ingest_pdf, inputs=pdf_input, outputs=[doc_info, doc_filter])
            eval_btn.click(fn=run_eval, outputs=eval_out)
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Query Interface")
            chatbot = gr.Chatbot(height=500, label="PharmaDocs Chat", render_markdown=True)
            chunk_display = gr.Markdown("")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask about test methods, storage conditions, lot numbers...",
                                 label="", scale=4, container=False)
                submit = gr.Button("Send →", variant="primary", scale=1)
            gr.Markdown("**Quick queries:**")
            with gr.Row():
                gr.Button("🧪 Test methods?").click(
                    fn=lambda h, f, s: chat("What test methods were used for quality control?", h, f, s),
                    inputs=[chatbot, doc_filter, show_chunks], outputs=[chatbot, chunk_display])
                gr.Button("🌡 Storage conditions?").click(
                    fn=lambda h, f, s: chat("What are the storage conditions?", h, f, s),
                    inputs=[chatbot, doc_filter, show_chunks], outputs=[chatbot, chunk_display])
                gr.Button("🔢 Part numbers?").click(
                    fn=lambda h, f, s: chat("What are the part numbers?", h, f, s),
                    inputs=[chatbot, doc_filter, show_chunks], outputs=[chatbot, chunk_display])
            submit.click(fn=chat, inputs=[msg, chatbot, doc_filter, show_chunks], outputs=[chatbot, chunk_display])
            msg.submit(fn=chat, inputs=[msg, chatbot, doc_filter, show_chunks], outputs=[chatbot, chunk_display])
    gr.Markdown("---\n**Architecture:** BGE embeddings → FAISS + BM25 → Cross-encoder reranking → Llama 3.3 70B")

demo.launch()
