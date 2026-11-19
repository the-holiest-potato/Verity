import json
from unittest.mock import patch, MagicMock
from core.claim_extractor import extract_claims
from evidence.pipeline import investigate_claim
from ml.predict import (
    VerityPredictor,
    PREDICTION_MAPPING,
    build_model_evidence,
)
from evidence.verdict import determine_overall_verdict


def test_full_workflow_with_article():
    # 1. Given a news article
    article_text = (
        "The Reserve Bank of India announced a monetary policy decision. "
        "The RBI reduced the policy repo rate by 25 basis points on November 5, 2026. "
        "Governor Shaktikanta Das announced the decision."
    )

    mock_gemini_data = {
        "claims": [
            {
                "claim": "The Reserve Bank of India announced a monetary policy decision.",
                "type": "factual",
                "checkable": True,
                "attribution": None,
            },
            {
                "claim": "The RBI reduced the policy repo rate by 25 basis points on November 5, 2026.",
                "type": "factual",
                "checkable": True,
                "attribution": None,
            },
            {
                "claim": "Governor Shaktikanta Das announced the decision.",
                "type": "factual",
                "checkable": True,
                "attribution": None,
            },
        ]
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(mock_gemini_data)

    mock_search_results = [
        {
            "title": "RBI cuts repo rate by 25 bps",
            "url": "https://www.reuters.com/business/rbi-rate-decision",
            "content": "The Reserve Bank of India reduced the policy repo rate by 25 basis points on November 5, 2026.",
            "published_date": "2026-11-05",
            "relevance_score": 0.95,
        }
    ]

    mock_page = {
        "title": "RBI cuts repo rate by 25 bps",
        "content": "The Reserve Bank of India announced a monetary policy decision on November 5, 2026. The RBI reduced the policy repo rate by 25 basis points. Analysts welcome the decision.",
        "success": True,
        "published_date": "2026-11-05",
        "date_source": "page_metadata",
        "author": "Reuters Staff",
        "sitename": "Reuters",
    }

    with patch("core.claim_extractor.client.models.generate_content", return_value=mock_response):
        claims = extract_claims(article_text)

    assert len(claims) == 3
    target_claim = claims[1]
    assert target_claim["checkable"] is True

    # 2. Investigate claim
    with patch("evidence.pipeline.search_claim", return_value=mock_search_results), \
         patch("evidence.pipeline.extract_page", return_value=mock_page):
        investigation = investigate_claim(target_claim["claim"])

    assert investigation["sources_found"] == 1
    assert investigation["pages_extracted"] == 1
    assert "assessment" in investigation

    # 3. Format evidence for ML model
    evidence_text = build_model_evidence(investigation)
    assert len(evidence_text) > 0
    assert "Passage:" in evidence_text

    # 4. Predict with real trained VerityPredictor
    predictor = VerityPredictor()
    ml_result = predictor.predict(target_claim["claim"], evidence_text)

    assert ml_result["prediction"] in PREDICTION_MAPPING
    mapped_verdict = PREDICTION_MAPPING[ml_result["prediction"]]
    assert mapped_verdict in [
        "Verified",
        "Unverified (evidence contradicts the claim)",
        "Ambiguous",
        "Ambiguous / Conflicting",
    ]
    assert "model_score" in ml_result
    assert isinstance(ml_result["model_score"], float)
    assert "all_scores" in ml_result
    assert len(ml_result["all_scores"]) == 4

    # 5. Evidence-aware overall verdict
    overall_verdict = determine_overall_verdict(
        target_claim["claim"], investigation["assessment"], ml_result
    )
    assert overall_verdict["overall_verdict"] in (
        "Verified",
        "Unverified",
        "Ambiguous",
        "Review Required",
    )


def test_full_workflow_with_standalone_claim():
    standalone_claim = "The Reserve Bank of India reduced the repo rate by 25 basis points on November 5, 2026."

    mock_gemini_data = {
        "claims": [
            {
                "claim": standalone_claim,
                "type": "factual",
                "checkable": True,
                "attribution": None,
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(mock_gemini_data)

    mock_search_results = [
        {
            "title": "RBI monetary policy statement",
            "url": "https://rbi.org.in/press/repo-rate-decision",
            "content": "The Reserve Bank of India reduced the policy repo rate by 25 basis points on November 5, 2026.",
            "published_date": "2026-11-05",
            "relevance_score": 0.99,
        }
    ]

    mock_page = {
        "title": "RBI monetary policy statement",
        "content": "The Reserve Bank of India reduced the repo rate by 25 basis points on November 5, 2026.",
        "success": True,
        "published_date": "2026-11-05",
        "date_source": "page_metadata",
        "author": "RBI",
        "sitename": "Reserve Bank of India",
    }

    with patch("core.claim_extractor.client.models.generate_content", return_value=mock_response):
        claims = extract_claims(standalone_claim)

    assert len(claims) == 1
    claim = claims[0]
    assert claim["checkable"] is True

    with patch("evidence.pipeline.search_claim", return_value=mock_search_results), \
         patch("evidence.pipeline.extract_page", return_value=mock_page):
        investigation = investigate_claim(claim["claim"])

    evidence_text = build_model_evidence(investigation)
    assert evidence_text != ""

    predictor = VerityPredictor()
    ml_result = predictor.predict(claim["claim"], evidence_text)

    assert ml_result["prediction"] in PREDICTION_MAPPING
    mapped_verdict = PREDICTION_MAPPING[ml_result["prediction"]]
    assert mapped_verdict in [
        "Verified",
        "Unverified (evidence contradicts the claim)",
        "Ambiguous",
        "Ambiguous / Conflicting",
    ]
    assert ml_result["model_score"] > 0.0

    # Evidence-aware overall verdict
    overall_verdict = determine_overall_verdict(
        claim["claim"], investigation["assessment"], ml_result
    )
    assert overall_verdict["overall_verdict"] in (
        "Verified",
        "Unverified",
        "Ambiguous",
        "Review Required",
    )
