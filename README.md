# 🧬 PharmaDocs AI — Pharmaceutical Document Intelligence

> End-to-end RAG pipeline for extracting and querying pharmaceutical compliance documents (SDF blobs, certificates of quality, packaging specs, BSE/TSE declarations, and more).

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/sravanib/PharmaDocs-AI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What It Does

Pharmaceutical companies receive multi-document SDF blobs weekly — certificates, declarations, and specs bundled together with no structure. Manually reviewing them costs 3+ hours per submission.

**PharmaDocs AI** solves this with a 3-stage RAG pipeline:
1. **Ingest** — Extract text via PyMuPDF (with Tesseract OCR fallback for scanned pages), classify 7 document types using keyword heuristics, chunk with a sliding window
2. **Retrieve** — Hybrid FAISS dense search + BM25 keyword search, then cross-encoder reranking
3. **Answer** — Llama 3.3 70B generates grounded answers with source citations and confidence scores

---

## 🏗 Architecture

```
PDF Upload → PyMuPDF + Tesseract OCR → Sliding Window Chunking (200w, 40 overlap)
    → BGE Embeddings → FAISS IndexFlatIP + BM25Okapi
    → Hybrid Retrieval (top-8) → Cross-Encoder Reranking (top-4)
    → Llama 3.3 70B (temp=0.1) → Answer + Source Citations
```

| Component | Technology | Config |
|---|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 | 384-dim, cosine similarity |
| Vector Store | FAISS IndexFlatIP | L2-normalized, in-memory |
| Sparse Search | BM25Okapi | rank-bm25 |
| Reranker | ms-marco-MiniLM-L-6-v2 | CrossEncoder, top-4 |
| LLM | Llama 3.3 70B | Groq API, temp=0.1 |
| UI | Gradio Blocks | Deployed on HF Spaces |

---

## 📊 Evaluation Results

Tested on 5 pharmaceutical queries against `pharma-blob-sample.pdf`:

| Metric | Score |
|---|---|
| Recall@3 | 60% |
| MRR | 0.60 |
| Citation Accuracy | 100% |
| Hallucinations | 0 |
| Avg Response Time | ~6s |
| Retrieval Latency | ~1.8s |
| LLM Generation | ~0.8s |

---

## 🚀 Quick Start

```bash
pip install groq pymupdf sentence-transformers faiss-cpu rank-bm25 gradio numpy pandas
export GROQ_API_KEY=your_key_here
python app.py
```

Or use the [live demo](https://huggingface.co/spaces/sravanib/PharmaDocs-AI) — no setup needed.

---

## 📁 Repository Structure

```
pharma-document-intelligence/
├── app.py                          # Full RAG pipeline + Gradio UI
├── requirements.txt
├── .gitignore
├── README.md
└── notebooks/
    ├── 01_RAG_Pipeline.ipynb
    ├── 02_Embedding_Comparison.ipynb
    ├── 03_RAG_Configurations.ipynb
    ├── 04_Open_Source_LLMs.ipynb
    ├── 05_Page_Segmentation.ipynb
    ├── 06_Query_Routing.ipynb
    └── 07_Gradio_RAG_UI.ipynb
```

---

## 🔬 Document Types Supported

- Cover Letter
- Certificate of Quality
- Packaging Specification
- BSE/TSE Declaration
- Material Description
- Supplier Qualification
- Chain of Custody

---

## 🛠 Key Design Decisions

**Why hybrid retrieval?** BM25 catches exact lot numbers and part numbers that dense embeddings miss. The cross-encoder reranker then fixes ranking errors — +15% Recall@3 vs vector-only.

**Why keyword classifier?** Zero latency, zero API cost, 90%+ accuracy on structured pharma docs. An LLM classifier would add ~1s per ingested page.

**Why temperature=0.1?** Factual pharmaceutical data requires deterministic answers. Higher temperature increases hallucination risk on compliance documents.

---

## 📈 Future Improvements

- Fine-tune cross-encoder on pharmaceutical Q&A pairs
- Add pdfplumber for structured table extraction
- Swap FAISS for ChromaDB (persistent storage)
- HyDE (Hypothetical Document Embeddings) for vague queries

---

## 👤 Author

**Sravani Bhamidipaty** — AI Externship, Pharmaceutical Document Intelligence
