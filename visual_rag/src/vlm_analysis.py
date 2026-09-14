from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_vlm():
    """
    Load the Qwen vision-language model and processor.
    """

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    return processor, model


def analyze_page(
    page_number,
    page_image,
    processor,
    model,
    max_new_tokens=350
):
    """
    Analyze a PDF page using a vision-language model.

    The model is asked to identify:
    - topic
    - text evidence
    - visual elements
    - relationships
    - grounding information
    """

    prompt = """
Analyze this PDF page for a multimodal RAG system.

Return exactly these sections:

TOPIC:
Describe the main topic of the page.

TEXT_EVIDENCE:
List important readable text that supports understanding of the page.

VISUAL_ELEMENTS:
Describe diagrams, boxes, arrows, tables, icons, layouts,
or other important visual elements.

RELATIONSHIPS:
Explain relationships between the components shown on the page.

GROUNDING:
Explain what information from this page could be useful
for answering questions about the document.

IMPORTANT:
- Preserve terminology from the page.
- Do not invent information.
- If something is unclear, say so.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": page_image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[page_image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    generated_ids = (
        output_ids[
            :,
            inputs.input_ids.shape[1]:
        ]
    )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return {
        "page": page_number,
        "visual_context": result
    }


def analyze_pages(
    page_numbers,
    pages,
    processor,
    model
):
    """
    Analyze multiple selected pages sequentially.

    Sequential processing helps reduce GPU memory usage.
    """

    results = []

    for page_number in page_numbers:

        print(
            f"\nAnalyzing page {page_number}..."
        )

        page_image = pages[
            page_number - 1
        ]

        result = analyze_page(
            page_number=page_number,
            page_image=page_image,
            processor=processor,
            model=model
        )

        results.append(result)

        print(
            f"Completed page {page_number}."
        )

    return results


if __name__ == "__main__":

    print(
        "VLM module created successfully."
    )

    print(
        "Run VLM inference in a GPU environment."
    )