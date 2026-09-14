def build_hybrid_context(
    text_results,
    visual_results
):
    """
    Combine text and visual retrieval results
    into a source-aware context.

    Each piece of evidence keeps:
    - source type
    - page number
    - retrieval score
    - content
    """

    context_parts = []

    # --------------------------------------------------
    # Text evidence
    # --------------------------------------------------

    context_parts.append(
        "===== TEXT EVIDENCE ====="
    )

    for result in text_results:

        context_parts.append(
            f"""
[TEXT SOURCE: Page {result['page']}]
[TEXT SCORE: {result['score']:.4f}]

{result['text']}
"""
        )

    # --------------------------------------------------
    # Visual evidence
    # --------------------------------------------------

    context_parts.append(
        "===== VISUAL EVIDENCE ====="
    )

    for result in visual_results:

        context_parts.append(
            f"""
[VISUAL SOURCE: Page {result['page']}]
[COLPALI SCORE: {result['score']:.4f}]

{result['visual_context']}
"""
        )

    return "\n".join(
        context_parts
    )


def create_evidence_units(
    text_results,
    visual_results
):
    """
    Convert retrieval results into individual
    evidence units.

    These units are later used by the
    faithfulness evaluator.
    """

    evidence_units = []

    # --------------------------------------------------
    # Text evidence units
    # --------------------------------------------------

    for result in text_results:

        evidence_units.append({
            "type": "text",
            "page": result["page"],
            "content": result["text"]
        })

    # --------------------------------------------------
    # Visual evidence units
    # --------------------------------------------------

    for result in visual_results:

        evidence_units.append({
            "type": "visual",
            "page": result["page"],
            "content": result["visual_context"]
        })

    return evidence_units


def build_answer_prompt(
    query,
    hybrid_context
):
    """
    Create a grounded prompt for the final
    answer-generation model.
    """

    prompt = f"""
You are answering a question using evidence
retrieved from a PDF.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{hybrid_context}

INSTRUCTIONS:

1. Answer using only the retrieved evidence.
2. Use both text and visual evidence when useful.
3. Do not invent unsupported information.
4. Explain diagram relationships when relevant.
5. Cite the source page using [Page X].
6. If the retrieved evidence is insufficient,
   explicitly say so.
7. Keep the answer clear and concise.

ANSWER:
"""

    return prompt


if __name__ == "__main__":

    # Small demonstration showing how the
    # module combines evidence.

    example_text_results = [
        {
            "page": 6,
            "score": 0.7586,
            "text": (
                "MCP acts as a universal connector "
                "for AI systems."
            )
        }
    ]

    example_visual_results = [
        {
            "page": 6,
            "score": 19.2812,
            "visual_context": (
                "The page shows an AI Application "
                "connected through MCP to external "
                "systems."
            )
        }
    ]

    hybrid_context = build_hybrid_context(
        example_text_results,
        example_visual_results
    )

    print(hybrid_context)

    print("\nEvidence units:\n")

    evidence_units = create_evidence_units(
        example_text_results,
        example_visual_results
    )

    for unit in evidence_units:

        print(
            f"Page {unit['page']} "
            f"({unit['type']})"
        )