from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_pdf(pdf_path):
    """
    Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries containing page number and page text.
    """

    pdf_path = Path(pdf_path)

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


def split_documents(
    pages,
    chunk_size=800,
    chunk_overlap=100
):
    """
    Split PDF page text into smaller chunks
    while preserving the original page number.

    Args:
        pages: List of page dictionaries.
        chunk_size: Maximum chunk size.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of chunk dictionaries.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return chunks


def create_embeddings(
    chunks,
    model_name="all-MiniLM-L6-v2"
):
    """
    Create embeddings for all document chunks.

    Returns:
        model: SentenceTransformer model.
        embeddings: NumPy array containing embeddings.
    """

    model = SentenceTransformer(model_name)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return model, embeddings


def retrieve_documents(
    query,
    chunks,
    embeddings,
    model,
    top_k=5
):
    """
    Retrieve the most semantically similar
    document chunks for a query.

    Args:
        query: User question.
        chunks: Document chunks.
        embeddings: Chunk embeddings.
        model: SentenceTransformer model.
        top_k: Number of results to return.

    Returns:
        List of retrieved evidence.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "page": chunks[index]["page"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })

    return results


if __name__ == "__main__":

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    pdf_path = (
        project_root
        / "data"
        / "mcp.pdf"
    )

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    pages = load_pdf(pdf_path)

    print(
        f"Loaded pages: {len(pages)}"
    )

    # --------------------------------------------------
    # 2. Split into chunks
    # --------------------------------------------------

    chunks = split_documents(pages)

    print(
        f"Created chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    model, embeddings = create_embeddings(
        chunks
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------
    # 4. Query
    # --------------------------------------------------

    query = (
        "How does MCP connect AI applications "
        "to external systems?"
    )

    # --------------------------------------------------
    # 5. Retrieve
    # --------------------------------------------------

    results = retrieve_documents(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        top_k=5
    )

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    print("\nTop results:\n")

    for result in results:

        print("=" * 60)

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            result["text"][:500]
        )