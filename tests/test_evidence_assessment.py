import pytest
from evidence.temporal_stance import analyse_temporal_stance
from evidence.assessment import assess_evidence, classify_evidence_relation


def test_jensen_huang_stepping_down_supported():
    claim = "Co-founder Jensen Huang has stepped down as CEO."
    passage = "Nvidia co-founder Jensen Huang has stepped down as chief executive officer, the company announced today."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "supports_action"
    assert classify_evidence_relation(result) == "supporting"


def test_colette_kress_interim_ceo_supported():
    claim = "Colette Kress became interim CEO of NVIDIA."
    passage = "Chief Financial Officer Colette Kress will serve as interim CEO following the resignation of Jensen Huang."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "supports_action"
    assert classify_evidence_relation(result) == "supporting"


def test_jensen_huang_remains_ceo_contradiction():
    claim = "Co-founder Jensen Huang has stepped down as CEO."
    passage = "Jensen Huang denied resignation rumors and confirmed he remains as CEO of Nvidia."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "contradicts_action"
    assert classify_evidence_relation(result) == "contradicting"


def test_negation_contradiction_detected():
    claim = "New Zealand's new Food Bill bans gardening"
    passage = "The New Zealand Food Bill does not ban gardening or home-grown vegetables, according to food safety officials."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "contradicts_action"
    assert classify_evidence_relation(result) == "contradicting"


def test_satire_debunking_contradiction_detected():
    claim = "In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial."
    passage = "The claim originated on Scoopertino, a satirical fake news website that writes fictional stories about Apple."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "contradicts_action"
    assert classify_evidence_relation(result) == "contradicting"


def test_rate_cut_contradicted_by_hold():
    claim = "The Reserve Bank of India reduced the repo rate by 25 basis points on November 5, 2026."
    passage = "The Reserve Bank of India decided to keep the repo rate unchanged at 6.50 percent on November 5, 2026."

    result = analyse_temporal_stance(claim, passage)
    assert result["stance"] == "contradicts_action"
    assert classify_evidence_relation(result) == "contradicting"


def test_assess_evidence_syndication_and_counts():
    records = [
        {
            "domain": "reuters.com",
            "source_class": "established_news",
            "extraction_success": True,
            "publication_date": "2026-11-15",
            "relevant_passages": [
                {
                    "passage": "Nvidia co-founder Jensen Huang has stepped down as chief executive officer today.",
                    "evidence_relation": "potential_support",
                }
            ],
        },
        {
            "domain": "cnbc.com",
            "source_class": "established_news",
            "extraction_success": True,
            "publication_date": "2026-11-15",
            "relevant_passages": [
                {
                    "passage": "Nvidia co-founder Jensen Huang has stepped down as chief executive officer today.",
                    "evidence_relation": "potential_support",
                }
            ],
        },
    ]

    assessment = assess_evidence(records)
    assert assessment["evidence_summary"]["supporting"]["passage_count"] == 2
    assert assessment["evidence_summary"]["supporting"]["source_count"] == 2
    assert any("syndicated" in r.lower() for r in assessment["reasons"])
