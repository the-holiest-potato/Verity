"""
Regression test for Jensen Huang resignation and Colette Kress interim CEO claims.
Verifies:
1. Dynamic recognition of official NVIDIA sources & SEC filings.
2. Demotion of social media (Facebook CBS Mornings video).
3. Passage relevance scoring filtering unrelated articles (unrelated Business Insider article).
4. Compatible token representation feeding ML predictor.
5. Evidence-aware overall verdict preventing confident Refuted verdict when direct evidence supports.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evidence.source_classifier import classify_source
from evidence.passage_extractor import extract_relevant_passages
from evidence.assessment import assess_evidence
from evidence.verdict import determine_overall_verdict
from ml.predict import VerityPredictor, build_model_evidence


def test_jensen_regression():
    print("=" * 70)
    print("RUNNING JENSEN HUANG & COLETTE KRESS REGRESSION TEST")
    print("=" * 70)

    # 1. Official source classification
    claim_jensen = "Co-founder Jensen Huang has stepped down as CEO of NVIDIA."
    sources_to_test = [
        ("https://nvidianews.nvidia.com/news/leadership-transition", "official_primary"),
        ("https://www.sec.gov/ix?doc=/Archives/edgar/data/1045810/000104581026000001/nvda-8k.htm", "official_primary"),
        ("https://www.cnbc.com/2026/02/10/jensen-huang-resigns-nvidia.html", "established_news"),
        ("https://www.nytimes.com/2026/02/10/technology/nvidia-huang-resignation.html", "established_news"),
        ("https://finance.yahoo.com/news/nvidia-ceo-transition-123456.html", "established_news"),
        ("https://www.facebook.com/cbsmornings/videos/123456789", "social_media"),
        ("https://www.businessinsider.com/unrelated-ai-chips-market-trend", "other_secondary"),
        ("https://some-random-blog-12345.com/tech-rumors", "other_secondary")
    ]

    print("\n--- 1. Source Classification Test ---")
    for url, expected_class in sources_to_test:
        info = classify_source(url, claim=claim_jensen)
        print(f"URL: {url[:55]}... -> Tier: {info['tier']} | Class: {info['source_class']} | Primary: {info['is_primary']}")
        assert info["source_class"] == expected_class, f"Expected {expected_class}, got {info['source_class']}"
        if expected_class == "official_primary":
            assert info["tier"] == 1
            assert info["is_primary"] is True
        elif expected_class == "social_media":
            assert info["tier"] == 6
            assert info["is_social_media"] is True

    print("Source classification: ALL PASSED")

    # 2. Passage extraction & filtering unrelated Business Insider content
    print("\n--- 2. Passage Extraction & Relevance Scoring ---")
    cnbc_article = (
        "Nvidia co-founder Jensen Huang has stepped down as chief executive officer, "
        "the company announced on Tuesday. Huang, who led the graphics chipmaker for over 30 years, "
        "will transition to an advisory role. The move comes as a major shock to Wall Street."
    )
    unrelated_bi_article = (
        "Tech stocks experienced mixed trading today as investors digested broader macroeconomic data. "
        "Semiconductor demand remains robust across European and Asian supply chains. "
        "Analysts suggest consumer hardware upgrade cycles may accelerate in late 2026."
    )

    cnbc_passages = extract_relevant_passages(cnbc_article, claim_jensen)
    bi_passages = extract_relevant_passages(unrelated_bi_article, claim_jensen)

    print(f"CNBC extracted passages count: {len(cnbc_passages)}")
    assert len(cnbc_passages) > 0, "CNBC passage should be extracted"
    print(f"CNBC top passage: '{cnbc_passages[0]['passage'][:80]}...' (Score: {cnbc_passages[0]['relevance_score']})")

    print(f"Unrelated BI extracted passages count: {len(bi_passages)}")
    if bi_passages:
        print(f"BI top passage score: {bi_passages[0]['relevance_score']}")
        assert bi_passages[0]['relevance_score'] < cnbc_passages[0]['relevance_score'], "Unrelated article must score lower"

    # 3. Model Prediction with Compatible Context Token Format
    print("\n--- 3. ML Model Prediction with Compatible Context Token Format ---")
    predictor = VerityPredictor()

    investigation_records = [
        {
            "url": "https://nvidianews.nvidia.com/news/leadership-transition",
            "domain": "nvidianews.nvidia.com",
            "source_class": "corporate_official",
            "tier": 1,
            "extraction_success": True,
            "publication_date": "2026-02-10",
            "relevant_passages": [
                {
                    "passage": "NVIDIA today announced that founder Jensen Huang has stepped down as President and Chief Executive Officer.",
                    "evidence_relation": "potential_support",
                    "relevance_score": 18
                }
            ]
        },
        {
            "url": "https://www.cnbc.com/2026/02/10/jensen-huang-resigns-nvidia.html",
            "domain": "cnbc.com",
            "source_class": "established_news",
            "tier": 2,
            "extraction_success": True,
            "publication_date": "2026-02-10",
            "relevant_passages": [
                {
                    "passage": "Nvidia co-founder Jensen Huang has stepped down as chief executive officer, the company announced on Tuesday.",
                    "evidence_relation": "potential_support",
                    "relevance_score": 15
                }
            ]
        }
    ]

    formatted_evidence = build_model_evidence(investigation_records, claim=claim_jensen)
    print(f"Formatted Evidence for ML:\n{formatted_evidence}\n")

    ml_prediction = predictor.predict(claim_jensen, formatted_evidence)
    print(f"ML Model Output for Jensen Huang claim:")
    print(f"  Prediction: {ml_prediction['prediction']}")
    print(f"  Raw Confidence Score: {ml_prediction['raw_score']:.4f}")
    print(f"  All Probabilities: {ml_prediction['probabilities']}")

    # 4. Overall Verdict Determination
    print("\n--- 4. Overall Verdict Determination ---")
    assessment = assess_evidence(investigation_records)
    print(f"Coverage level: {assessment['coverage_level']}")
    print(f"Direct supporting sources: {assessment['supporting_sources_count']}")
    print(f"Direct contradicting sources: {assessment['contradicting_sources_count']}")

    overall_verdict = determine_overall_verdict(claim_jensen, assessment, ml_prediction)
    print(f"FINAL OVERALL VERDICT: {overall_verdict['overall_verdict']}")
    print(f"Reason: {overall_verdict['verdict_reason']}")
    print(f"Disagreement detected: {overall_verdict['has_disagreement']}")

    # Crucial check: never display Refuted as unquestionable verdict
    assert overall_verdict["overall_verdict"] in ("Verified", "Review Required"), (
        f"Verdict must be Verified or Review Required, got {overall_verdict['overall_verdict']}"
    )

    # 5. Colette Kress interim CEO claim test
    print("\n--- 5. Colette Kress Interim CEO Claim Test ---")
    claim_kress = "Colette Kress became interim CEO of NVIDIA."
    kress_records = [
        {
            "url": "https://nvidianews.nvidia.com/news/leadership-transition",
            "domain": "nvidianews.nvidia.com",
            "source_class": "corporate_official",
            "tier": 1,
            "extraction_success": True,
            "publication_date": "2026-02-10",
            "relevant_passages": [
                {
                    "passage": "Executive Vice President and Chief Financial Officer Colette Kress will serve as interim CEO following the resignation of Jensen Huang.",
                    "evidence_relation": "potential_support",
                    "relevance_score": 16
                }
            ]
        }
    ]
    formatted_kress_evidence = build_model_evidence(kress_records, claim=claim_kress)
    ml_kress = predictor.predict(claim_kress, formatted_kress_evidence)
    kress_assessment = assess_evidence(kress_records)
    kress_verdict = determine_overall_verdict(claim_kress, kress_assessment, ml_kress)

    print(f"Colette Kress ML Prediction: {ml_kress['prediction']} (Score: {ml_kress['raw_score']:.4f})")
    print(f"Colette Kress Overall Verdict: {kress_verdict['overall_verdict']}")
    assert kress_verdict["overall_verdict"] in ("Verified", "Review Required")

    # 6. Negative case: What if ML says Refuted (old behavior)?
    print("\n--- 6. Negative Case Simulation (ML disagrees with Evidence) ---")
    simulated_ml_refuted = {
        "prediction": "Refuted",
        "raw_score": 0.8541,
        "probabilities": {"Supported": 0.05, "Refuted": 0.8541, "Not Enough Evidence": 0.05, "Conflicting Evidence/Cherrypicking": 0.0459}
    }
    disagree_verdict = determine_overall_verdict(claim_jensen, assessment, simulated_ml_refuted)
    print(f"Simulated Disagreement Verdict: {disagree_verdict['overall_verdict']}")
    print(f"Disagreement Reason: {disagree_verdict['verdict_reason']}")
    assert disagree_verdict["overall_verdict"] == "Review Required"
    assert disagree_verdict["has_disagreement"] is True
    assert "disagreement" in disagree_verdict["verdict_reason"].lower()

    print("\n" + "=" * 70)
    print("ALL JENSEN HUANG & COLETTE KRESS REGRESSION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_jensen_regression()
