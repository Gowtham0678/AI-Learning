from pathlib import Path
import pymupdf


# Project root = visual_rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "data" / "mcp.pdf"
OUTPUT_DIR = PROJECT_ROOT / "rendered_pages"


def render_pdf(pdf_path, output_dir, zoom=2):
    """
    Render every page of a PDF as a PNG image.

    Args:
        pdf_path: Path to the PDF.
        output_dir: Directory for rendered pages.
        zoom: Rendering scale.

    Returns:
        List of generated image paths.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    image_paths = []

    matrix = pymupdf.Matrix(zoom, zoom)

    for page_number, page in enumerate(document, start=1):

        pixmap = page.get_pixmap(matrix=matrix)

        image_path = (
            output_dir / f"page_{page_number:03d}.png"
        )

        pixmap.save(image_path)

        image_paths.append(image_path)

    document.close()

    return image_paths


if __name__ == "__main__":

    paths = render_pdf(
        PDF_PATH,
        OUTPUT_DIR
    )

    print(f"Rendered {len(paths)} pages.")