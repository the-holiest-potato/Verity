import pytest
from evidence.verdict import determine_overall_verdict


def test_verdict_verified_when_evidence_and_ml_concur():
    assessment = {
        "coverage_level": "Moderate",
        "pages_extracted": 2,
        "evidence_summary": {
            "supporting": {"passage_count": 2, "source_count": 2, "primary_source_count": 1},
            "contradicting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
        },
    }
    ml_result = {
        "prediction": "Supported",
        "model_score": 0.92,
        "all_scores": {"Supported": 0.92, "Refuted": 0.05, "Not Enough Evidence": 0.01, "Conflicting Evidence/Cherrypicking": 0.02},
    }

    result = determine_overall_verdict("Claim text", assessment, ml_result)
    assert result["overall_verdict"] == "Verified"
    assert result["has_disagreement"] is False


def test_verdict_review_required_when_evidence_supports_but_ml_refutes():
    # Jensen Huang regression case
    assessment = {
        "coverage_level": "High",
        "pages_extracted": 3,
        "evidence_summary": {
            "supporting": {"passage_count": 3, "source_count": 3, "primary_source_count": 1},
            "contradicting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
        },
    }
    ml_result = {
        "prediction": "Refuted",
        "model_score": 0.8541,
        "all_scores": {"Supported": 0.08, "Refuted": 0.8541, "Not Enough Evidence": 0.04, "Conflicting Evidence/Cherrypicking": 0.02},
    }

    result = determine_overall_verdict("Co-founder Jensen Huang has stepped down as CEO.", assessment, ml_result)
    assert result["overall_verdict"] == "Review Required"
    assert result["has_disagreement"] is True
    assert "disagreement" in result["verdict_reason"].lower()


def test_verdict_review_required_when_evidence_contradicts_but_ml_supports():
    # RBI rate unchanged vs cut claim
    assessment = {
        "coverage_level": "Moderate",
        "pages_extracted": 2,
        "evidence_summary": {
            "supporting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
            "contradicting": {"passage_count": 2, "source_count": 2, "primary_source_count": 1},
        },
    }
    ml_result = {
        "prediction": "Supported",
        "model_score": 0.91,
        "all_scores": {"Supported": 0.91, "Refuted": 0.05, "Not Enough Evidence": 0.02, "Conflicting Evidence/Cherrypicking": 0.02},
    }

    result = determine_overall_verdict("The RBI reduced the repo rate.", assessment, ml_result)
    assert result["overall_verdict"] == "Review Required"
    assert result["has_disagreement"] is True


def test_verdict_unverified_when_evidence_contradicts_and_ml_refutes():
    assessment = {
        "coverage_level": "Moderate",
        "pages_extracted": 2,
        "evidence_summary": {
            "supporting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
            "contradicting": {"passage_count": 2, "source_count": 2, "primary_source_count": 0},
        },
    }
    ml_result = {
        "prediction": "Refuted",
        "model_score": 0.95,
        "all_scores": {"Supported": 0.02, "Refuted": 0.95, "Not Enough Evidence": 0.01, "Conflicting Evidence/Cherrypicking": 0.02},
    }

    result = determine_overall_verdict("Claim text", assessment, ml_result)
    assert result["overall_verdict"] == "Unverified"
    assert result["has_disagreement"] is False


def test_verdict_ambiguous_when_evidence_insufficient_even_if_ml_is_confident():
    # Constraint: Never mark Verified solely because model predicts Supported
    assessment = {
        "coverage_level": "Insufficient",
        "pages_extracted": 0,
        "evidence_summary": {
            "supporting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
            "contradicting": {"passage_count": 0, "source_count": 0, "primary_source_count": 0},
        },
    }
    ml_result = {
        "prediction": "Supported",
        "model_score": 0.98,
        "all_scores": {"Supported": 0.98, "Refuted": 0.01, "Not Enough Evidence": 0.005, "Conflicting Evidence/Cherrypicking": 0.005},
    }

    result = determine_overall_verdict("Claim text", assessment, ml_result)
    assert result["overall_verdict"] == "Ambiguous"
    assert "insufficient" in result["verdict_reason"].lower()


def test_verdict_ambiguous_when_evidence_is_conflicting():
    assessment = {
        "coverage_level": "Moderate",
        "pages_extracted": 4,
        "evidence_summary": {
            "supporting": {"passage_count": 2, "source_count": 2, "primary_source_count": 0},
            "contradicting": {"passage_count": 2, "source_count": 2, "primary_source_count": 0},
        },
    }
    ml_result = {
        "prediction": "Conflicting Evidence/Cherrypicking",
        "model_score": 0.65,
        "all_scores": {"Supported": 0.15, "Refuted": 0.15, "Not Enough Evidence": 0.05, "Conflicting Evidence/Cherrypicking": 0.65},
    }

    result = determine_overall_verdict("Claim text", assessment, ml_result)
    assert result["overall_verdict"] == "Ambiguous"
    assert "conflicting" in result["verdict_reason"].lower()
