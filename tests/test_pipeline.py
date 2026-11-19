
import pytest

import evidence.pipeline as pipeline


CLAIM = (
    "The Reserve Bank of India reduced the repo rate "
    "by 25 basis points on November 5, 2026."
)


def test_pipeline_rejects_empty_claim():
    with pytest.raises(ValueError):
        pipeline.investigate_claim("")


def test_pipeline_preserves_event_and_publication_dates(monkeypatch):
    article = (
        "The Reserve Bank of India announced a monetary policy "
        "decision on November 5, 2026. "
        "The RBI reduced the policy repo rate by 25 basis points."
    )

    monkeypatch.setattr(
        pipeline,
        "search_claim",
        lambda claim, max_results: [
            {
                "url": "https://rbi.org.in/example",
                "title": "RBI policy decision",
            }
        ],
    )

    monkeypatch.setattr(
        pipeline,
        "extract_page",
        lambda url: {
            "success": True,
            "title": "RBI policy decision",
            "published_date": "2026-11-06",
            "date_source": "page_metadata",
            "content": article,
            "author": None,
            "sitename": "Reserve Bank of India",
        },
    )

    monkeypatch.setattr(
        pipeline,
        "classify_source",
        lambda url: {
            "domain": "rbi.org.in",
            "source_class": "official_primary",
            "classification_reason": "Test classification",
        },
    )

    result = pipeline.investigate_claim(CLAIM)

    assert result["sources_found"] == 1
    assert result["pages_extracted"] == 1

    source = result["evidence"][0]

    assert source["publication_date"] == "2026-11-06"
    assert source["publication_date_source"] == "page_metadata"

    contextual_passages = [
        passage
        for passage in source["relevant_passages"]
        if passage.get("source_context_date") == "2026-11-05"
    ]

    assert contextual_passages

    passage = contextual_passages[0]

    assert passage["event_date_source"] == "source_context"
    assert passage["date_association_type"] == "adjacent_event_context"
    assert passage["source_context_sentence"]
    assert passage["source_context_action_sentence"]
    assert passage["temporal_relation"] == "same_date"
    assert passage["evidence_relation"] == "potential_support"


def test_pipeline_does_not_inherit_ambiguous_event_date():
    candidates = [
        {
            "date": "2026-10-07",
            "action": "reduced",
            "source_sentence": "The RBI reduced the repo rate.",
            "date_sentence": "The RBI announced a monetary policy decision on October 7, 2026.",
            "association_type": "adjacent_event_context",
        },
        {
            "date": "2026-11-05",
            "action": "reduced",
            "source_sentence": "The RBI reduced the repo rate.",
            "date_sentence": "The RBI announced a monetary policy decision on November 5, 2026.",
            "association_type": "adjacent_event_context",
        },
    ]

    result = pipeline.find_matching_source_context_date(
        passage_text="The RBI reduced the repo rate by 25 basis points.",
        passage_action="reduced",
        source_date_candidates=candidates,
    )

    assert result is None


def test_pipeline_matches_date_to_correct_action_sentence():
    passage = "The RBI reduced the repo rate by 25 basis points."

    candidates = [
        {
            "date": "2026-11-05",
            "action": "reduced",
            "source_sentence": passage,
            "date_sentence": "The RBI announced its decision on November 5, 2026.",
            "association_type": "adjacent_event_context",
        },
        {
            "date": "2026-10-07",
            "action": "reduced",
            "source_sentence": "The RBI reduced another rate in an earlier decision.",
            "date_sentence": "The decision was announced on October 7, 2026.",
            "association_type": "adjacent_event_context",
        },
    ]

    result = pipeline.find_matching_source_context_date(
        passage,
        "reduced",
        candidates,
    )

    assert result is not None
    assert result["date"] == "2026-11-05"


def test_pipeline_does_not_inherit_date_from_unrelated_sentence():
    passage = "The RBI reduced the repo rate by 25 basis points."

    candidates = [
        {
            "date": "2026-11-05",
            "action": "reduced",
            "source_sentence": "The RBI reduced a different rate.",
            "date_sentence": "The decision was announced on November 5, 2026.",
            "association_type": "adjacent_event_context",
        },
    ]

    result = pipeline.find_matching_source_context_date(
        passage,
        "reduced",
        candidates,
    )

    assert result is None
