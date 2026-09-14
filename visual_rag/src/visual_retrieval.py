from pathlib import Path

import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor


MODEL_NAME = "vidore/colpali-v1.3-hf"


def get_page_paths(image_dir):
    """
    Return sorted rendered PDF page paths.
    """

    image_dir = Path(image_dir)

    return sorted(
        image_dir.glob("page_*.png")
    )


def load_page_images(image_dir):
    """
    Load rendered PDF pages as RGB PIL images.
    """

    page_paths = get_page_paths(image_dir)

    pages = []

    for page_path in page_paths:

        image = Image.open(
            page_path
        ).convert("RGB")

        pages.append(image)

    print(
        f"Loaded {len(pages)} page images."
    )

    return pages


def load_colpali_model():
    """
    Load ColPali processor and retrieval model.
    """

    processor = ColPaliProcessor.from_pretrained(
        MODEL_NAME
    )

    model = ColPaliForRetrieval.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    return processor, model


def encode_images(
    pages,
    processor,
    model,
    batch_size=4
):
    """
    Encode PDF page images using ColPali.

    Embeddings are moved to CPU after each batch
    to reduce GPU memory usage.
    """

    all_embeddings = []

    for start in range(
        0,
        len(pages),
        batch_size
    ):

        batch = pages[
            start:start + batch_size
        ]

        image_inputs = (
            processor
            .process_images(batch)
            .to(model.device)
        )

        with torch.no_grad():

            embeddings = model(
                **image_inputs
            ).embeddings

        all_embeddings.append(
            embeddings.cpu()
        )

        print(
            f"Encoded pages "
            f"{start + 1}-{min(start + batch_size, len(pages))}"
        )

    return torch.cat(
        all_embeddings,
        dim=0
    )


def retrieve_visual_pages(
    query,
    image_embeddings,
    processor,
    model,
    top_k=5
):
    """
    Retrieve the most relevant PDF pages
    using ColPali visual retrieval.
    """

    query_inputs = (
        processor
        .process_queries([query])
        .to(model.device)
    )

    with torch.no_grad():

        query_embeddings = model(
            **query_inputs
        ).embeddings

    # Move image embeddings to the model device
    # only during scoring.
    image_embeddings = (
        image_embeddings
        .to(model.device)
    )

    scores = processor.score_retrieval(
        query_embeddings,
        image_embeddings
    )[0]

    top_scores, top_indices = scores.topk(
        top_k
    )

    results = []

    for score, index in zip(
        top_scores,
        top_indices
    ):

        page_number = (
            index.item() + 1
        )

        results.append({
            "page": page_number,
            "score": float(
                score.item()
            )
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

    image_dir = (
        project_root
        / "rendered_pages"
    )

    # --------------------------------------------------
    # Load pages
    # --------------------------------------------------

    pages = load_page_images(
        image_dir
    )

    # --------------------------------------------------
    # Load ColPali
    # --------------------------------------------------

    print("\nLoading ColPali...")

    processor, model = (
        load_colpali_model()
    )

    print("ColPali loaded.")

    # --------------------------------------------------
    # Create visual embeddings
    # --------------------------------------------------

    print("\nEncoding pages...")

    image_embeddings = encode_images(
        pages,
        processor,
        model,
        batch_size=4
    )

    print(
        "\nImage embedding shape:",
        image_embeddings.shape
    )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = (
        "How does MCP connect AI applications "
        "to external systems?"
    )

    # --------------------------------------------------
    # Visual retrieval
    # --------------------------------------------------

    results = retrieve_visual_pages(
        query=query,
        image_embeddings=image_embeddings,
        processor=processor,
        model=model,
        top_k=5
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nTop visual results:\n")

    for result in results:

        print("=" * 60)

        print(
            f"Page: {result['page']}"
        )

        print(
            f"ColPali score: "
            f"{result['score']:.4f}"
        )