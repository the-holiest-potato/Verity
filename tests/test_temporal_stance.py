import pytest

from evidence.temporal_stance import (
    analyse_temporal_stance,
    extract_dates,
    parse_publication_date,
)


CLAIM = (
    "The Reserve Bank of India reduced the repo rate "
    "by 25 basis points on November 5, 2026."
)


@pytest.mark.parametrize(
    "passage, expected_stance, expected_relation",
    [
        (
            "The repo rate was raised by 25 basis points on 7 October 2026.",
            "contradicts_action",
            "different_event_date",
        ),
        (
            "ANZ expects the RBI to deliver a rate cut of 25 basis points.",
            "prediction",
            "forecast_not_confirmation",
        ),
        (
            "The RBI reduced the repo rate by 25 basis points on November 5, 2026.",
            "supports_action",
            "potential_support",
        ),
        (
            "The RBI kept its repo rate unchanged on November 5, 2026.",
            "contradicts_action",
            "potential_contradiction",
        ),
        (
            "The central bank had cut the repo rate by 100 basis points in 2025.",
            "historical_reference",
            "historical_context",
        ),
        (
            "The Reserve Bank raises the repo rate by 25 basis points.",
            "contradicts_action",
            "conflicting_action_date_unconfirmed",
        ),
    ],
)
def test_temporal_classification(
    passage,
    expected_stance,
    expected_relation,
):
    result = analyse_temporal_stance(CLAIM, passage)

    assert result["stance"] == expected_stance
    assert result["evidence_relation"] == expected_relation


def test_same_date_is_detected():
    result = analyse_temporal_stance(
        CLAIM,
        "The RBI reduced the repo rate on November 5, 2026.",
    )

    assert result["temporal_relation"] == "same_date"


def test_different_date_is_detected():
    result = analyse_temporal_stance(
        CLAIM,
        "The RBI reduced the repo rate on October 7, 2026.",
    )

    assert result["temporal_relation"] == "different_date"


def test_publication_date_is_not_automatically_event_date():
    result = analyse_temporal_stance(
        CLAIM,
        "The RBI reduced its policy repo rate by 25 basis points.",
        publication_date="2026-11-05",
    )

    assert result["passage_dates"] == []
    assert result["date_provenance"]["publication_date"] == "2026-11-05"
    assert result["temporal_relation"] == "passage_date_not_found"


def test_invalid_calendar_dates_are_ignored():
    dates = extract_dates(
        "The RBI announced a decision on February 31, 2026."
    )

    assert dates == []


def test_publication_date_parsing():
    result = parse_publication_date("2026-11-05")

    assert result.isoformat() == "2026-11-05"