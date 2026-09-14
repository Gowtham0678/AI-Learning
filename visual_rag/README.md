# Visual RAG: Multimodal Document Retrieval System

A Visual Retrieval-Augmented Generation (Visual RAG) system that combines
traditional text retrieval with visual document retrieval to answer questions
from visually rich PDF documents.

The project demonstrates how modern AI systems can retrieve information from
both the textual content and visual structure of documents such as diagrams,
architectures, screenshots, and other visual elements.

---

## 🚀 Project Overview

Traditional RAG systems primarily work with extracted text:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Retrieval
 ↓
LLM
 ↓
Answer
```

This approach can miss important information when the meaning of a document
depends on its visual layout or diagrams.

This project extends RAG with visual retrieval:

```text
                    PDF
                     │
            ┌────────┴────────┐
            ↓                 ↓
          Text            Page Images
            ↓                 ↓
       Embeddings          ColPali
            ↓                 ↓
    Text Retrieval    Visual Retrieval
            │                 │
            └────────┬────────┘
                     ↓
              Hybrid Retrieval
                     ↓
               Visual Context
                     ↓
               Answer Prompt
                     ↓
                    LLM
                     ↓
                  Answer
                     ↓
                Evaluation
```

---

## 🎯 Objectives

The project focuses on:

- Text-based document retrieval
- Visual document retrieval
- Multimodal document understanding
- Diagram-aware retrieval
- Hybrid evidence construction
- Evidence-based answer generation
- Automated RAG evaluation
- Faithfulness and hallucination analysis

---

# 🧠 Technologies Used

## Text Retrieval

- Python
- PyMuPDF
- LangChain Text Splitters
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Scikit-learn cosine similarity

## Visual Retrieval

- ColPali
- `vidore/colpali-v1.3-hf`
- PyTorch
- Hugging Face Transformers
- PIL

## Visual Understanding

- Vision-Language Models (VLMs)
- Qwen2.5-VL

## Evaluation

- Semantic similarity
- Source-aware evidence matching
- Natural Language Inference (NLI)
- `cross-encoder/nli-deberta-v3-base`

---

# 📂 Project Structure

```text
visual_rag/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── mcp.pdf
│
├── notebooks/
│   └── visual_rag_demo.ipynb
│
├── rendered_pages/
│   └── generated PDF page images
│
└── src/
    ├── __init__.py
    ├── evaluation.py
    ├── hybrid_retrieval.py
    ├── inspect_page.py
    ├── pdf_processing.py
    ├── text_extraction.py
    ├── text_retrieval.py
    ├── visual_retrieval.py
    └── vlm_analysis.py
```

> `rendered_pages/` contains generated files and is excluded from Git using
> `.gitignore`.

---

# 🔎 Text Retrieval Pipeline

The text retrieval pipeline performs the following steps:

```text
PDF
 ↓
PyMuPDF
 ↓
Page Text
 ↓
Recursive Character Chunking
 ↓
Sentence Transformer Embeddings
 ↓
Cosine Similarity
 ↓
Top-K Results
```

The current implementation uses:

```text
all-MiniLM-L6-v2
```

Example query:

```text
How does MCP connect AI applications to external systems?
```

Example text retrieval results:

```text
Page 6  → 0.7586
Page 8  → 0.6538
Page 13 → 0.6224
Page 7  → 0.6128
Page 21 → 0.5820
```

---

# 👁️ Visual Retrieval with ColPali

Text extraction alone may not capture the meaning of diagrams and document
layouts.

The visual retrieval pipeline converts PDF pages into images and uses ColPali
to retrieve visually relevant pages.

```text
PDF Pages
    ↓
Page Images
    ↓
ColPali
    ↓
Multi-Vector Page Representations
    ↓
Query Representation
    ↓
Late Interaction Scoring
    ↓
Top-K Visual Pages
```

Example visual retrieval results for the MCP query:

```text
Page 8  → 19.3594
Page 6  → 19.2812
Page 7  → 18.4375
Page 21 → 18.2031
Page 10 → 18.0781
```

ColPali inference is intended to run in a GPU environment such as Google
Colab because the model is computationally expensive for CPU-only machines.

---

# 🤖 Vision-Language Model Analysis

After visually relevant pages are retrieved, a Vision-Language Model can
analyze the page image.

The VLM is prompted to identify:

- Topic
- Text evidence
- Visual elements
- Relationships
- Grounding information

Example visual interpretation:

```text
AI Application
      ↓
     MCP
      ↓
External Systems
 ├── Web APIs
 ├── Database
 ├── GitHub
 ├── Gmail
 ├── Slack
 └── Local Filesystem
```

This allows the system to understand relationships represented visually,
rather than treating the page as plain text.

---

# 🔗 Hybrid Retrieval

The system combines text and visual evidence.

```text
             User Query
                 │
        ┌────────┴────────┐
        ↓                 ↓
 Text Retrieval     Visual Retrieval
        │                 │
        ↓                 ↓
 Text Evidence      Visual Evidence
        │                 │
        └────────┬────────┘
                 ↓
          Hybrid Context
                 ↓
          Evidence Units
```

Each evidence unit contains:

```python
{
    "type": "text",
    "page": 6,
    "content": "..."
}
```

or:

```python
{
    "type": "visual",
    "page": 6,
    "content": "..."
}
```

This gives downstream components access to both textual and visual evidence.

---

# 🛡️ Evidence-Based Answering

The hybrid retrieval layer creates an answer prompt that instructs the
answering model to:

1. Use only retrieved evidence.
2. Use text and visual evidence when useful.
3. Avoid unsupported information.
4. Explain diagram relationships when relevant.
5. Cite source pages.
6. Explicitly state when evidence is insufficient.

This provides a grounding layer between retrieval and generation.

---

# 📊 RAG Evaluation

The project includes an automated evaluation pipeline.

The evaluator combines:

```text
Claim
 ↓
Source-Aware Evidence Filtering
 ↓
Semantic Similarity
 ↓
Best Evidence
 ↓
NLI Model
 ↓
SUPPORTED / UNSUPPORTED / UNCERTAIN
```

The NLI model used is:

```text
cross-encoder/nli-deberta-v3-base
```

The evaluator uses three NLI relationships:

```text
ENTAILMENT
NEUTRAL
CONTRADICTION
```

These are mapped to:

```text
ENTAILMENT    → SUPPORTED
NEUTRAL       → UNCERTAIN
CONTRADICTION → UNSUPPORTED
```

---

# 📈 Evaluation Results

Five example MCP claims were evaluated.

```text
Total Claims: 5

Supported:    3
Unsupported:  0
Uncertain:    2

Strict NLI Faithfulness: 60%
```

The `60%` score is intentionally treated as a conservative automated NLI
metric rather than a direct measure of factual truth.

Two important observations emerged.

## 1. Semantic similarity is not proof

One claim achieved:

```text
Semantic Score: 0.9819
NLI: neutral
```

A high semantic similarity score indicates that two pieces of text are
related, but it does not guarantee that one logically entails the other.

## 2. Visual evidence requires multimodal evaluation

A visual claim received:

```text
Semantic Score: 0.9419
NLI: neutral
```

The NLI model evaluates the VLM-generated textual description rather than the
original image.

Therefore, neutral classifications for visual claims can represent
limitations of the evaluator rather than incorrect evidence.

---

# 🧪 Local Testing

Create a Python virtual environment:

```cmd
python -m venv .venv
```

Activate it on Windows CMD:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```cmd
pip install -r requirements.txt
```

Run the main pipeline:

```cmd
python main.py
```

Run text retrieval independently:

```cmd
python src\text_retrieval.py
```

Run hybrid retrieval:

```cmd
python src\hybrid_retrieval.py
```

Run evaluation:

```cmd
python src\evaluation.py
```

---

# ☁️ GPU / Google Colab Workflow

ColPali and VLM inference are computationally expensive.

For GPU-based experimentation:

1. Open the notebook:

```text
notebooks/visual_rag_demo.ipynb
```

2. Open or upload the notebook in Google Colab.

3. Enable a GPU runtime.

4. Render the PDF pages.

5. Load the ColPali model.

6. Generate visual page embeddings.

7. Perform visual retrieval.

8. Run VLM analysis on retrieved pages.

9. Use the resulting visual evidence with the hybrid retrieval pipeline.

The notebook serves as the experimental GPU workflow while the `src/`
directory contains reusable project modules.

---

# ⚙️ CPU vs GPU

| Component | Local CPU | GPU Recommended |
|---|---:|---:|
| PDF processing | ✅ | Optional |
| Text retrieval | ✅ | Optional |
| Embeddings | ✅ | Optional |
| Hybrid context | ✅ | Optional |
| ColPali | ⚠️ | ✅ |
| VLM | ⚠️ | ✅ |
| NLI evaluation | ✅ | Optional |

---

# 💡 Key Engineering Lessons

This project demonstrates several important AI engineering concepts.

## RAG

Retrieval improves the grounding of LLM responses by supplying relevant
external evidence.

## Multimodal RAG

Documents can contain important information that exists outside their text
layer.

## Visual Retrieval

ColPali enables retrieval based on the visual representation of document
pages.

## Vision-Language Models

Vision-Language Models can interpret relationships between text, diagrams,
objects, and document layout.

## Evidence Grounding

Retrieving evidence and explicitly passing it to the answer generation stage
helps reduce unsupported responses.

## Evaluation

Different evaluation methods measure different properties:

```text
Semantic Similarity → Relatedness
NLI                 → Logical relationship
Source Awareness    → Correct evidence location
Human Review        → Final evidence validity
```

No single metric should be treated as a perfect measure of RAG quality.

---

# 🧪 Example End-to-End Query

Example question:

```text
How does MCP connect AI applications to external systems?
```

Text retrieval identifies relevant pages based on their textual content.

Visual retrieval identifies visually relevant pages using ColPali.

The VLM analyzes the retrieved page images and extracts visual relationships.

The hybrid layer combines both forms of evidence.

The final answer is then grounded in the retrieved evidence and evaluated
using semantic similarity and NLI.

---

# 🚧 Current Limitations

- ColPali inference requires significant computational resources.
- VLM analysis is currently designed for GPU environments.
- NLI evaluation can produce false negatives for visual evidence.
- Visual evidence is currently represented through VLM-generated textual
  descriptions for downstream NLI evaluation.
- The current project demonstrates the retrieval and evaluation pipeline;
  production deployment would require additional serving, caching,
  observability, and security layers.

---

# 🔮 Future Improvements

Potential improvements include:

- Multimodal answer generation directly from page images
- Better visual evidence verification
- Reranking across text and visual evidence
- Automatic claim extraction from generated answers
- Automated citation validation
- Retrieval quality metrics such as Recall@K and MRR
- Evaluation datasets with human-verified ground truth
- FastAPI serving layer
- Redis caching
- PostgreSQL metadata storage
- Docker deployment
- Cloud GPU inference
- Production monitoring and observability

---

# 👨‍💻 Learning Outcomes

Through this project, the following concepts were implemented and explored:

- Python-based AI pipelines
- PDF processing
- Document chunking
- Embeddings
- Cosine similarity
- Text retrieval
- Visual document retrieval
- ColPali
- Vision-Language Models
- Hybrid retrieval
- Evidence units
- RAG prompting
- Semantic evaluation
- NLI evaluation
- Faithfulness analysis
- GPU-based AI workflows

---

# 📌 Project Status

## Core Visual RAG Pipeline: Complete ✅

The project currently demonstrates:

```text
Text Retrieval         ✅
Visual Retrieval       ✅
VLM Analysis           ✅
Hybrid Retrieval       ✅
Evidence Construction  ✅
Automated Evaluation   ✅
```

Future work focuses on productionization, deployment, and more robust
multimodal evaluation.

---

# ⭐ Summary

This project demonstrates a multimodal approach to Retrieval-Augmented
Generation by combining:

```text
Text Retrieval
      +
Visual Retrieval
      +
Vision-Language Understanding
      +
Hybrid Evidence
      +
Automated Evaluation
```

The goal is to build RAG systems that can understand not only what a document
says, but also how information is represented visually within the document.