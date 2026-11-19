"""
Test the complete live workflow with at least one article and one standalone claim.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.claim_extractor import extract_claims
from evidence.pipeline import investigate_claim
from ml.predict import (
    VerityPredictor,
    PREDICTION_MAPPING,
    build_model_evidence,
)


def run_verification(input_text, input_description):
    print("\n" + "=" * 60)
    print(f"RUNNING WORKFLOW FOR: {input_description}")
    print("=" * 60)
    print("Input text:\n", input_text)

    # 1. Gemini Claim Extraction
    print("\n[Step 1] Extracting claims with Gemini...")
    claims = extract_claims(input_text)
    print(f"Extracted {len(claims)} claim(s):")
    for i, c in enumerate(claims, 1):
        print(f"  Claim {i}: {c.get('claim')}")
        print(f"    Type: {c.get('type')}, Checkable: {c.get('checkable')}, Attribution: {c.get('attribution')}")

    predictor = VerityPredictor()

    for i, claim in enumerate(claims, 1):
        claim_text = claim.get("claim")
        is_checkable = claim.get("checkable", False)

        print(f"\n--- Processing Claim {i} ---")
        print("Claim:", claim_text)

        if not is_checkable:
            print("Claim is not checkable. Skipping retrieval and ML prediction.")
            continue

        # 2. Evidence Retrieval
        print("[Step 2] Investigating claim using evidence pipeline...")
        investigation = investigate_claim(claim_text, max_sources=2)
        assessment = investigation.get("assessment", {})
        print(f"  Sources found: {investigation.get('sources_found')}")
        print(f"  Pages extracted: {investigation.get('pages_extracted')}")
        print(f"  Coverage Level: {assessment.get('coverage_level')}")
        print(f"  Reasons: {assessment.get('reasons')}")

        # 3. Format Evidence for Model
        print("[Step 3] Formatting evidence for ML model...")
        evidence_text = build_model_evidence(investigation, max_passages=3)

        if not evidence_text.strip():
            print("  ⚠️ Missing evidence: No relevant passages extracted. Prediction skipped.")
            continue

        print(f"  Formed evidence string ({len(evidence_text)} chars):")
        print("  " + "\n  ".join(evidence_text.splitlines()[:4]) + "...")

        # 4. Predict with ML Model
        print("[Step 4] Running ML Model inference...")
        ml_result = predictor.predict(claim_text, evidence_text)
        raw_pred = ml_result["prediction"]
        raw_score = ml_result["model_score"]
        mapped_verdict = PREDICTION_MAPPING.get(raw_pred, raw_pred)

        print(f"\n  [ML Verdict]: {mapped_verdict}")
        print(f"  [Raw Prediction]: {raw_pred}")
        print(f"  [Raw Model Score]: {raw_score:.4f} (Inference confidence score, not truth probability)")
        print(f"  [Class Scores]: {ml_result['all_scores']}")

        # 5. Retrieved Sources
        print("\n  [Retrieved Sources & Passages]:")
        for s_idx, src in enumerate(investigation.get("evidence", []), 1):
            title = src.get("title")
            url = src.get("url")
            pub_date = src.get("publication_date")
            passages = src.get("relevant_passages", [])
            print(f"    Source {s_idx}: {title} | {pub_date} | {url}")
            for p_idx, p in enumerate(passages[:2], 1):
                print(f"      Passage {p_idx}: {p.get('passage')[:100]}... (Relevance: {p.get('relevance_score')})")

    print("\nCompleted verification for:", input_description)


if __name__ == "__main__":
    # Test 1: Full News Article
    article_input = (
        "The Reserve Bank of India reduced the repo rate by 25 basis points on November 5, 2026. "
        "The decision was taken unanimously by the Monetary Policy Committee to support economic growth. "
        "Some analysts believe this move will boost lending."
    )
    run_verification(article_input, "TEST 1: NEWS ARTICLE WITH MULTIPLE CLAIMS")

    # Test 2: Standalone Claim
    standalone_input = "The Reserve Bank of India reduced the repo rate by 25 basis points on November 5, 2026."
    run_verification(standalone_input, "TEST 2: STANDALONE CLAIM")
