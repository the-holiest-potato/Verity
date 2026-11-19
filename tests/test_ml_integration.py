import pytest
from ml.predict import (
    VerityPredictor,
    PREDICTION_MAPPING,
    build_model_evidence,
)


def test_prediction_mapping_labels():
    assert PREDICTION_MAPPING["Supported"] == "Verified"
    assert (
        PREDICTION_MAPPING["Refuted"]
        == "Unverified (evidence contradicts the claim)"
    )
    assert PREDICTION_MAPPING["Not Enough Evidence"] == "Ambiguous"
    assert (
        PREDICTION_MAPPING["Conflicting Evidence/Cherrypicking"]
        == "Ambiguous / Conflicting"
    )


def test_build_model_evidence_empty():
    assert build_model_evidence({}) == ""
    assert build_model_evidence({"evidence": []}) == ""
    assert build_model_evidence(None) == ""


def test_build_model_evidence_formats_and_sorts_passages():
    investigation = {
        "evidence": [
            {
                "domain": "sourceA.com",
                "publication_date": "2026-11-05",
                "relevant_passages": [
                    {
                        "passage": "Lower score passage.",
                        "relevance_score": 0.35,
                    },
                    {
                        "passage": "Higher score passage.",
                        "relevance_score": 0.88,
                    },
                ],
            },
            {
                "domain": "sourceB.com",
                "publication_date": None,
                "relevant_passages": [
                    {
                        "passage": "Medium score passage.",
                        "relevance_score": 0.65,
                    },
                ],
            },
        ]
    }

    evidence_text = build_model_evidence(investigation, max_passages=2)
    assert evidence_text != ""

    # Higher score passage should appear first
    assert evidence_text.startswith("Source: sourceA.com (2026-11-05)\nPassage: Higher score passage.")
    # Medium score passage should appear second
    assert "Medium score passage." in evidence_text
    # Lower score passage should not be present because max_passages=2
    assert "Lower score passage." not in evidence_text


def test_predictor_validates_empty_inputs():
    predictor = VerityPredictor()

    with pytest.raises(ValueError, match="Claim cannot be empty"):
        predictor.predict("", "Some evidence text.")

    with pytest.raises(ValueError, match="Evidence cannot be empty"):
        predictor.predict("Some claim.", "")


def test_predictor_inference():
    predictor = VerityPredictor()
    claim = "The RBI reduced the repo rate on November 5, 2026."
    evidence = "The Reserve Bank of India reduced the policy repo rate on November 5, 2026."

    result = predictor.predict(claim, evidence)

    assert "prediction" in result
    assert result["prediction"] in PREDICTION_MAPPING
    assert "model_score" in result
    assert isinstance(result["model_score"], float)
    assert "all_scores" in result
    assert len(result["all_scores"]) == 4
