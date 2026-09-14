from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def load_models():
    """
    Load the embedding model and NLI model.
    """
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    nli_model = CrossEncoder(NLI_MODEL)

    return embedding_model, nli_model


# ---------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------

def semantic_similarity(claim, evidence, embedding_model):
    """
    Calculate cosine similarity between a claim and evidence.
    """

    claim_embedding = embedding_model.encode(
        [claim],
        convert_to_numpy=True
    )

    evidence_embedding = embedding_model.encode(
        [evidence],
        convert_to_numpy=True
    )

    score = cosine_similarity(
        claim_embedding,
        evidence_embedding
    )[0][0]

    return float(score)


# ---------------------------------------------------------
# Find Best Evidence
# ---------------------------------------------------------

def find_best_evidence(
    claim,
    evidence_units,
    embedding_model,
    expected_pages=None,
    expected_types=None
):
    """
    Find the evidence unit most semantically similar
    to the claim.

    Optional source constraints:
    - expected_pages
    - expected_types
    """

    candidates = evidence_units

    # Filter by expected page
    if expected_pages:
        candidates = [
            unit
            for unit in candidates
            if unit["page"] in expected_pages
        ]

    # Filter by expected evidence type
    if expected_types:
        candidates = [
            unit
            for unit in candidates
            if unit["type"] in expected_types
        ]

    if not candidates:
        return None

    results = []

    for unit in candidates:

        score = semantic_similarity(
            claim,
            unit["content"],
            embedding_model
        )

        results.append({
            "page": unit["page"],
            "type": unit["type"],
            "content": unit["content"],
            "semantic_score": score
        })

    results.sort(
        key=lambda x: x["semantic_score"],
        reverse=True
    )

    return results[0]


# ---------------------------------------------------------
# NLI Evaluation
# ---------------------------------------------------------

def nli_evaluate(claim, evidence, nli_model):
    """
    Determine whether the evidence supports the claim.

    NLI labels:
    contradiction
    neutral
    entailment
    """

    logits = nli_model.predict(
        [[claim, evidence]]
    )[0]

    labels = [
        "contradiction",
        "neutral",
        "entailment"
    ]

    predicted_index = int(np.argmax(logits))

    # Stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "label": labels[predicted_index],
        "confidence": confidence,
        "logits": logits.tolist()
    }


# ---------------------------------------------------------
# Final Claim Evaluation
# ---------------------------------------------------------

def evaluate_claim(
    claim,
    evidence_units,
    embedding_model,
    nli_model,
    expected_pages=None,
    expected_types=None
):
    """
    Evaluate one claim against the retrieved evidence.
    """

    best_evidence = find_best_evidence(
        claim=claim,
        evidence_units=evidence_units,
        embedding_model=embedding_model,
        expected_pages=expected_pages,
        expected_types=expected_types
    )

    if best_evidence is None:

        return {
            "claim": claim,
            "status": "UNCERTAIN",
            "page": None,
            "type": None,
            "semantic_score": 0.0,
            "nli_label": "neutral",
            "confidence": 0.0
        }

    nli_result = nli_evaluate(
        claim,
        best_evidence["content"],
        nli_model
    )

    # Final classification
    if nli_result["label"] == "entailment":
        status = "SUPPORTED"

    elif nli_result["label"] == "contradiction":
        status = "UNSUPPORTED"

    else:
        status = "UNCERTAIN"

    return {
        "claim": claim,
        "status": status,
        "page": best_evidence["page"],
        "type": best_evidence["type"],
        "semantic_score": best_evidence["semantic_score"],
        "nli_label": nli_result["label"],
        "confidence": nli_result["confidence"]
    }


# ---------------------------------------------------------
# Evaluate All Claims
# ---------------------------------------------------------

def evaluate_claims(
    expected_sources,
    evidence_units,
    embedding_model,
    nli_model
):
    """
    Evaluate all claims using source-aware constraints.
    """

    results = []

    for item in expected_sources:

        result = evaluate_claim(
            claim=item["claim"],
            evidence_units=evidence_units,
            embedding_model=embedding_model,
            nli_model=nli_model,
            expected_pages=item.get("expected_pages"),
            expected_types=item.get("expected_types")
        )

        results.append(result)

    return results


# ---------------------------------------------------------
# Calculate Summary
# ---------------------------------------------------------

def calculate_summary(results):
    """
    Calculate overall evaluation statistics.
    """

    total = len(results)

    supported = sum(
        1
        for result in results
        if result["status"] == "SUPPORTED"
    )

    unsupported = sum(
        1
        for result in results
        if result["status"] == "UNSUPPORTED"
    )

    uncertain = sum(
        1
        for result in results
        if result["status"] == "UNCERTAIN"
    )

    faithfulness = (
        supported / total * 100
        if total > 0
        else 0
    )

    return {
        "total": total,
        "supported": supported,
        "unsupported": unsupported,
        "uncertain": uncertain,
        "faithfulness": faithfulness
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Claims generated from our MCP example
    # -----------------------------------------------------

    claims = [
        "MCP provides a standard middle-layer interface between AI applications and external systems.",

        "MCP helps avoid M × N custom integrations.",

        "Page 6 shows an AI Application connected through MCP to Web APIs, Database, GitHub, Slack, Gmail and Local Filesystem.",

        "MCP simplifies integration by using a standard interface.",

        "An MCP client embedded in an AI app establishes a connection to an MCP server."
    ]

    # -----------------------------------------------------
    # Expected sources
    # -----------------------------------------------------

    expected_sources = [
        {
            "claim": claims[0],
            "expected_pages": [6, 8],
            "expected_types": ["text", "visual"]
        },

        {
            "claim": claims[1],
            "expected_pages": [8],
            "expected_types": ["text", "visual"]
        },

        {
            "claim": claims[2],
            "expected_pages": [6],
            "expected_types": ["visual"]
        },

        {
            "claim": claims[3],
            "expected_pages": [8],
            "expected_types": ["text", "visual"]
        },

        {
            "claim": claims[4],
            "expected_pages": [21],
            "expected_types": ["text"]
        }
    ]

    # -----------------------------------------------------
    # Evidence units
    #
    # These are representative examples so this file can
    # also be tested independently.
    # -----------------------------------------------------

    evidence_units = [

        {
            "type": "text",
            "page": 6,
            "content": (
                "MCP acts as a universal connector for AI systems "
                "and allows AI models to interact with external "
                "tools, resources, and environments."
            )
        },

        {
            "type": "visual",
            "page": 6,
            "content": (
                "The page shows an AI Application connected "
                "through MCP to external systems including "
                "Web APIs, Database, GitHub, Gmail, Slack "
                "and Local Filesystem."
            )
        },

        {
            "type": "text",
            "page": 8,
            "content": (
                "MCP provides a standard interface for connecting "
                "multiple AI applications with multiple tools, "
                "avoiding the M × N integration problem."
            )
        },

        {
            "type": "visual",
            "page": 8,
            "content": (
                "The diagram illustrates multiple AI models "
                "connecting to multiple external tools through "
                "a standard MCP interface."
            )
        },

        {
            "type": "text",
            "page": 21,
            "content": (
                "An MCP client embedded in an AI application "
                "establishes a connection to an MCP server."
            )
        }
    ]

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print("\nLoading models...")

    embedding_model, nli_model = load_models()

    print("Models loaded successfully.")

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = evaluate_claims(
        expected_sources=expected_sources,
        evidence_units=evidence_units,
        embedding_model=embedding_model,
        nli_model=nli_model
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CLAIM RESULTS")
    print("=" * 70)

    for index, result in enumerate(results, start=1):

        print(f"\nClaim {index}")
        print("-" * 70)

        print(f"Claim: {result['claim']}")
        print(f"Status: {result['status']}")
        print(f"Evidence Page: {result['page']}")
        print(f"Evidence Type: {result['type']}")
        print(f"Semantic Score: {result['semantic_score']:.4f}")
        print(f"NLI Label: {result['nli_label']}")
        print(f"NLI Confidence: {result['confidence']:.4f}")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = calculate_summary(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total Claims: {summary['total']}")
    print(f"Supported: {summary['supported']}")
    print(f"Unsupported: {summary['unsupported']}")
    print(f"Uncertain: {summary['uncertain']}")
    print(f"Strict NLI Faithfulness: {summary['faithfulness']:.2f}%")