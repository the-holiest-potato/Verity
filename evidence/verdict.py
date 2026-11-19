"""
evidence/verdict.py
Evidence-Aware Overall Verdict Engine for Verity 2.0.

Synthesizes the Evidence Pipeline Assessment (retrieval coverage, direct
supporting/contradicting passages, source tiers, independent corroboration)
with the Machine Learning Model Prediction.

Possible outcomes:
- Verified: sufficiently strong direct supporting evidence.
- Unverified: sufficiently strong direct contradicting evidence.
- Ambiguous: insufficient or genuinely conflicting evidence.
- Review Required: significant disagreement between ML prediction and evidence assessment.
"""


def determine_overall_verdict(claim, assessment, ml_result):
    """
    Synthesize Evidence Assessment and ML Model Prediction into an overall verdict.

    Parameters:
        claim (str): The claim text.
        assessment (dict): Output from assess_evidence().
        ml_result (dict or None): Output from VerityPredictor.predict() or None if missing.

    Returns:
        dict: {
            "overall_verdict": str,
            "verdict_reason": str,
            "has_disagreement": bool,
            "evidence_stance": str,
            "ml_prediction": str or None,
            "ml_raw_score": float or None,
        }
    """
    evidence_summary = assessment.get("evidence_summary", {})
    supporting_info = evidence_summary.get("supporting", {})
    contradicting_info = evidence_summary.get("contradicting", {})

    supporting_passages = supporting_info.get("passage_count", 0)
    supporting_sources = supporting_info.get("source_count", 0)
    primary_supporting_sources = supporting_info.get("primary_source_count", 0)

    contradicting_passages = contradicting_info.get("passage_count", 0)
    contradicting_sources = contradicting_info.get("source_count", 0)
    primary_contradicting_sources = contradicting_info.get("primary_source_count", 0)

    coverage_level = assessment.get("coverage_level", "Insufficient")
    total_extracted = assessment.get("pages_extracted", 0)

    raw_pred = ml_result.get("prediction") if ml_result else None
    raw_score = (
        ml_result.get("model_score")
        if ml_result and ml_result.get("model_score") is not None
        else (ml_result.get("raw_score") if ml_result else None)
    )
    score_display = f"{raw_score:.4f}" if raw_score is not None else "N/A"

    # Determine evidentiary stance from retrieved sources
    if total_extracted == 0 or (supporting_passages == 0 and contradicting_passages == 0):
        evidence_stance = "insufficient"
    elif supporting_passages > 0 and contradicting_passages > 0:
        evidence_stance = "conflicting"
    elif supporting_passages > 0:
        evidence_stance = "supporting"
    elif contradicting_passages > 0:
        evidence_stance = "contradicting"
    else:
        evidence_stance = "insufficient"

    # 1. INSUFFICIENT EVIDENCE
    # Never mark a claim Verified solely because the model predicts Supported.
    # If evidence is insufficient, outcome is Ambiguous regardless of ML confidence.
    if evidence_stance == "insufficient":
        return {
            "overall_verdict": "Ambiguous",
            "verdict_reason": (
                "Insufficient direct evidence was retrieved from credible sources to corroborate or "
                "refute this claim. Claims cannot be marked Verified without credible direct evidence, "
                "regardless of model confidence."
            ),
            "has_disagreement": (raw_pred in ("Supported", "Refuted")),
            "evidence_stance": evidence_stance,
            "ml_prediction": raw_pred,
            "ml_raw_score": raw_score,
        }

    # 2. GENUINELY CONFLICTING EVIDENCE AMONG SOURCES
    # Credible sources present opposing accounts.
    if evidence_stance == "conflicting":
        return {
            "overall_verdict": "Ambiguous",
            "verdict_reason": (
                f"Retrieved sources present conflicting factual accounts: {supporting_passages} passage(s) "
                f"support the claim while {contradicting_passages} passage(s) contradict it. "
                "Flagged as Ambiguous pending authoritative resolution."
            ),
            "has_disagreement": (raw_pred in ("Supported", "Refuted")),
            "evidence_stance": evidence_stance,
            "ml_prediction": raw_pred,
            "ml_raw_score": raw_score,
        }

    # 3. EVIDENCE DIRECTLY SUPPORTS THE CLAIM
    if evidence_stance == "supporting":
        # Case 3A: Evidence supports + ML Supported -> potentially Verified
        if raw_pred == "Supported":
            primary_note = " (including official primary sources)" if primary_supporting_sources > 0 else ""
            return {
                "overall_verdict": "Verified",
                "verdict_reason": (
                    f"Direct supporting evidence was confirmed across {supporting_sources} source(s){primary_note}, "
                    f"and the sequence-classification model concurs with the evidence (score: {score_display})."
                ),
                "has_disagreement": False,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }

        # Case 3B: Evidence supports + ML Refuted -> Review Required (Disagreement detected!)
        if raw_pred == "Refuted":
            return {
                "overall_verdict": "Review Required",
                "verdict_reason": (
                    f"Significant disagreement detected: Direct passages retrieved from {supporting_sources} "
                    f"source(s) support the claim, but the ML model predicted Refuted (raw score: {score_display}). "
                    "Flagged for human review because primary/credible evidence contradicts the model classification."
                ),
                "has_disagreement": True,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }

        # Case 3C: Evidence supports + ML NEI or Conflicting
        if primary_supporting_sources > 0 or supporting_sources >= 2:
            return {
                "overall_verdict": "Verified",
                "verdict_reason": (
                    f"Direct supporting evidence verified across {supporting_sources} credible source(s) "
                    f"(including official primary sources), taking precedence over conservative classifier output."
                ),
                "has_disagreement": False,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }
        else:
            classifier_text = f"returned {raw_pred}" if raw_pred else "was unavailable in this deployment"
            return {
                "overall_verdict": "Review Required",
                "verdict_reason": (
                    f"Supporting evidence was retrieved from {supporting_sources} source(s), but classifier "
                    f"{classifier_text}. Additional verification is recommended."
                ),
                "has_disagreement": True if raw_pred else False,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }

    # 4. EVIDENCE DIRECTLY CONTRADICTS THE CLAIM
    if evidence_stance == "contradicting":
        # Case 4A: Evidence contradicts + ML Refuted -> Unverified (evidence contradicts claim)
        if raw_pred in ("Refuted", "Not Enough Evidence"):
            return {
                "overall_verdict": "Unverified",
                "verdict_reason": (
                    f"Direct contradicting evidence was identified across {contradicting_sources} credible "
                    f"source(s), refuting the claim. The evidence demonstrates that the claim is contrary to fact."
                ),
                "has_disagreement": False,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }

        # Case 4B: Evidence contradicts + ML Supported -> Review Required (Disagreement detected!)
        if raw_pred == "Supported":
            return {
                "overall_verdict": "Review Required",
                "verdict_reason": (
                    f"Significant disagreement detected: Direct passages retrieved from {contradicting_sources} "
                    f"source(s) contradict the claim, but the ML model predicted Supported (raw score: {score_display}, "
                    "likely due to lexical overlap with the subject). Evidence contradicts the claim."
                ),
                "has_disagreement": True,
                "evidence_stance": evidence_stance,
                "ml_prediction": raw_pred,
                "ml_raw_score": raw_score,
            }

        return {
            "overall_verdict": "Unverified",
            "verdict_reason": (
                f"Contradicting evidence was identified across {contradicting_sources} source(s), refuting the claim."
            ),
            "has_disagreement": False,
            "evidence_stance": evidence_stance,
            "ml_prediction": raw_pred,
            "ml_raw_score": raw_score,
        }

    return {
        "overall_verdict": "Ambiguous",
        "verdict_reason": "Evidence assessment remains inconclusive.",
        "has_disagreement": False,
        "evidence_stance": "insufficient",
        "ml_prediction": raw_pred,
        "ml_raw_score": raw_score,
    }
