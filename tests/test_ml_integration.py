import pytest
from pathlib import Path
from unittest.mock import patch
from ml.predict import (
    VerityPredictor,
    PREDICTION_MAPPING,
    build_model_evidence,
    is_valid_local_model_dir,
    resolve_model_target,
    ModelNotFoundError,
    DEFAULT_LOCAL_MODEL_PATH,
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


def test_is_valid_local_model_dir():
    # Real local model directory exists and has config.json + model.safetensors
    assert is_valid_local_model_dir(DEFAULT_LOCAL_MODEL_PATH) is True

    # Non-existent directory
    assert is_valid_local_model_dir("/tmp/non_existent_model_dir_xyz") is False
    assert is_valid_local_model_dir(None) is False


def test_resolve_model_target_local_when_present():
    target, is_local, token = resolve_model_target(None)
    assert is_local is True
    assert Path(target).resolve() == DEFAULT_LOCAL_MODEL_PATH.resolve()


def test_resolve_model_target_missing_local_raises_model_not_found():
    with patch("ml.predict.is_valid_local_model_dir", return_value=False), \
         patch("ml.predict.get_configured_hf_repo_id", return_value=None):
        with pytest.raises(ModelNotFoundError) as exc_info:
            resolve_model_target(None)
        # Verify that the error message contains actionable instructions
        assert "Verity ML model could not be loaded" in str(exc_info.value)
        assert "VERITY_MODEL_REPO" in str(exc_info.value)


def test_resolve_model_target_env_repo_id_when_local_missing():
    with patch("ml.predict.is_valid_local_model_dir", return_value=False), \
         patch("ml.predict.get_configured_hf_repo_id", return_value="custom-org/verity-model"), \
         patch("ml.predict.get_hf_auth_token", return_value="test_token_123"):
        target, is_local, token = resolve_model_target(None)
        assert is_local is False
        assert target == "custom-org/verity-model"
        assert token == "test_token_123"


def test_resolve_model_target_explicit_hf_repo():
    target, is_local, token = resolve_model_target("my-user/verity-roberta")
    assert is_local is False
    assert target == "my-user/verity-roberta"


def test_resolve_model_target_invalid_fs_path():
    with pytest.raises(ModelNotFoundError):
        resolve_model_target("/mount/src/verity/models/verity-distilroberta")

