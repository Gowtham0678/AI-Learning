from pathlib import Path

from src.pdf_processing import render_pdf
from src.text_retrieval import (
    load_pdf,
    split_documents,
    create_embeddings,
    retrieve_documents,
)
from src.hybrid_retrieval import (
    build_hybrid_context,
    create_evidence_units,
    build_answer_prompt,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

PDF_PATH = PROJECT_ROOT / "data" / "mcp.pdf"
RENDERED_DIR = PROJECT_ROOT / "rendered_pages"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUERY = "How does MCP connect AI applications to external systems?"

TOP_K = 5


# ---------------------------------------------------------
# Text Retrieval
# ---------------------------------------------------------

def run_text_retrieval():

    print("\n" + "=" * 70)
    print("STEP 1: TEXT RETRIEVAL")
    print("=" * 70)

    pages = load_pdf(PDF_PATH)

    print(f"Loaded pages: {len(pages)}")

    chunks = split_documents(pages)

    print(f"Created chunks: {len(chunks)}")

    model, embeddings = create_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    results = retrieve_documents(
        query=QUERY,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        top_k=TOP_K,
    )

    print("\nTop text results:")

    for result in results:
        print(
            f"Page {result['page']} "
            f"| Score: {result['score']:.4f}"
        )

    return results


# ---------------------------------------------------------
# PDF Rendering
# ---------------------------------------------------------

def run_pdf_rendering():

    print("\n" + "=" * 70)
    print("STEP 2: PDF RENDERING")
    print("=" * 70)

    rendered_pages = render_pdf(
        PDF_PATH,
        RENDERED_DIR,
    )

    print(
        f"Rendered {len(rendered_pages)} pages."
    )

    return rendered_pages


# ---------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------

def run_hybrid_retrieval(
    text_results,
    visual_results,
):
    """
    Combine text and visual evidence.
    """

    print("\n" + "=" * 70)
    print("STEP 3: HYBRID RETRIEVAL")
    print("=" * 70)

    hybrid_context = build_hybrid_context(
        text_results,
        visual_results,
    )

    evidence_units = create_evidence_units(
        text_results,
        visual_results,
    )

    print(
        f"Created {len(evidence_units)} evidence units."
    )

    return hybrid_context, evidence_units


# ---------------------------------------------------------
# Answer Prompt
# ---------------------------------------------------------

def create_rag_prompt(
    query,
    hybrid_context,
):

    print("\n" + "=" * 70)
    print("STEP 4: ANSWER PROMPT")
    print("=" * 70)

    prompt = build_answer_prompt(
        query,
        hybrid_context,
    )

    print("Answer prompt created.")

    return prompt


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("VISUAL RAG PIPELINE")
    print("=" * 70)

    print(f"\nPDF: {PDF_PATH}")
    print(f"Query: {QUERY}")

    # -----------------------------------------------------
    # 1. Render PDF
    # -----------------------------------------------------

    run_pdf_rendering()

    # -----------------------------------------------------
    # 2. Text retrieval
    # -----------------------------------------------------

    text_results = run_text_retrieval()

    # -----------------------------------------------------
    # 3. Visual retrieval
    #
    # Heavy ColPali + VLM inference is intentionally not
    # executed here on the local CPU machine.
    #
    # These results come from GPU inference in Colab.
    # -----------------------------------------------------

    visual_results = [
        {
            "page": 8,
            "score": 19.3594,
            "visual_context": (
                "The page shows multiple AI models "
                "connecting to multiple external systems "
                "through a standard MCP interface."
            ),
        },
        {
            "page": 6,
            "score": 19.2812,
            "visual_context": (
                "The page shows an AI Application "
                "connected through MCP to external systems."
            ),
        },
        {
            "page": 7,
            "score": 18.4375,
            "visual_context": (
                "The diagram shows an AI application "
                "using an MCP client to communicate "
                "with an MCP server and external tools."
            ),
        },
    ]

    # -----------------------------------------------------
    # 4. Hybrid retrieval
    # -----------------------------------------------------

    hybrid_context, evidence_units = run_hybrid_retrieval(
        text_results,
        visual_results,
    )

    # -----------------------------------------------------
    # 5. Create final answer prompt
    # -----------------------------------------------------

    answer_prompt = create_rag_prompt(
        QUERY,
        hybrid_context,
    )

    # -----------------------------------------------------
    # Display evidence
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    print(hybrid_context)

    # -----------------------------------------------------
    # Display prompt
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ANSWER PROMPT")
    print("=" * 70)

    print(answer_prompt)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"Evidence units: {len(evidence_units)}")
    print("Text retrieval: OK")
    print("Visual retrieval: GPU/Colab results loaded")
    print("Hybrid retrieval: OK")
    print("Answer prompt: OK")


if __name__ == "__main__":
    main()