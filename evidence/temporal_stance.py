
import re
from datetime import date, datetime, timedelta


MONTHS = {
    "january": 1, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12
}

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5,
    "sunday": 6
}


def extract_dates(text):
    """Extract valid explicit calendar dates."""

    if not text:
        return []

    dates = set()
    month_pattern = "|".join(MONTHS.keys())

    patterns = [
        (
            rf"\b({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
            "month_first"
        ),
        (
            rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})\b",
            "day_first"
        )
    ]

    for pattern, format_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if format_type == "month_first":
                month = MONTHS[match.group(1).lower()]
                day = int(match.group(2))
                year = int(match.group(3))
            else:
                day = int(match.group(1))
                month = MONTHS[match.group(2).lower()]
                year = int(match.group(3))

            try:
                dates.add(date(year, month, day).isoformat())
            except ValueError:
                pass

    for match in re.finditer(
        r"\b(\d{4})-(\d{2})-(\d{2})\b", text
    ):
        year, month, day = map(int, match.groups())

        try:
            dates.add(date(year, month, day).isoformat())
        except ValueError:
            pass

    return sorted(dates)


def extract_event_dates(text):
    """
    Extract dates that may describe events.

    Social-media timestamps such as:
    'Wednesday, October 7, 2026 at 12:09 PM'
    are excluded.

    This is still a rule-based filter, not a guarantee that
    every remaining date is an event date.
    """

    if not text:
        return []

    cleaned_text = text

    month_pattern = "|".join(MONTHS.keys())

    timestamp_patterns = [
        # Wednesday, October 7, 2026 at 12:09 PM
        rf"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        rf",?\s+(?:{month_pattern})\s+\d{{1,2}},?\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",

        # October 7, 2026 at 12:09 PM
        rf"\b(?:{month_pattern})\s+\d{{1,2}},?\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",

        # 7 October 2026 at 12:09 PM
        rf"\b\d{{1,2}}\s+(?:{month_pattern})\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",

        # 2026-10-07 at 12:09 PM
        r"\b\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)\b"
    ]

    for pattern in timestamp_patterns:
        cleaned_text = re.sub(
            pattern,
            " ",
            cleaned_text,
            flags=re.IGNORECASE
        )

    return extract_dates(cleaned_text)


def parse_publication_date(publication_date):
    """Convert supported date formats into datetime.date."""

    if not publication_date:
        return None

    if isinstance(publication_date, datetime):
        return publication_date.date()

    if isinstance(publication_date, date):
        return publication_date

    value = str(publication_date).strip()

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).date()
    except ValueError:
        pass

    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def resolve_weekday_dates(text, publication_date=None):
    """
    Resolve weekday references relative to publication date.

    This is an estimate, not a verified event date.
    """

    publication_day = parse_publication_date(publication_date)

    if not publication_day:
        return []

    resolved_dates = set()
    text = (text or "").lower()

    for weekday, weekday_number in WEEKDAYS.items():

        if not re.search(rf"\b{weekday}\b", text):
            continue

        days_back = (
            publication_day.weekday() - weekday_number
        ) % 7

        if re.search(rf"\blast\s+{weekday}\b", text):
            if days_back == 0:
                days_back = 7

        resolved_day = publication_day - timedelta(
            days=days_back
        )

        resolved_dates.add(resolved_day.isoformat())

    return sorted(resolved_dates)


def detect_action(text):
    """Detect reduced, raised, held or unclear."""

    text = (text or "").lower()

    strong_hold_patterns = [
        r"\bkept\b.{0,80}\bunchanged\b",
        r"\bleft\b.{0,80}\bunchanged\b",
        r"\bheld\b.{0,80}\b(?:steady|unchanged)\b",
        r"\bheld its\b.{0,50}\brate\b",
        r"\bmaintained\b.{0,50}\brate\b",
        r"\bno change\b"
    ]

    if any(
        re.search(pattern, text)
        for pattern in strong_hold_patterns
    ):
        return "held"

    patterns = {
        "reduced": [
            r"\breduc(?:e|es|ed|ing)\b",
            r"\blower(?:s|ed|ing)?\b",
            r"\bcut\b",
            r"\bcutting\b",
            r"\brate cut(?:s|ting)?\b",
            r"\beas(?:e|es|ed|ing)\b"
        ],
        "raised": [
            r"\brais(?:e|es|ed|ing)\b",
            r"\bhik(?:e|es|ed|ing)\b",
            r"\bincreas(?:e|es|ed|ing)\b",
            r"\brate hikes?\b"
        ],
        "held": [
            r"\bheld\b",
            r"\bunchanged\b",
            r"\bkept steady\b",
            r"\bno change\b",
            r"\bmaintained\b"
        ]
    }

    detected = set()

    for action, action_patterns in patterns.items():
        if any(
            re.search(pattern, text)
            for pattern in action_patterns
        ):
            detected.add(action)

    if len(detected) == 1:
        return detected.pop()

    return "unclear"


def is_prediction(text):
    """Identify forecasts and uncertain future statements."""

    text = (text or "").lower()

    patterns = [
        r"\bexpects?\b",
        r"\bexpected\b",
        r"\bforecast(?:s|ed|ing)?\b",
        r"\bpredict(?:s|ed|ing)?\b",
        r"\bproject(?:s|ed|ing)?\b",
        r"\bcould\b",
        r"\bmay\b",
        r"\bmight\b",
        r"\bpossible\b",
        r"\bpossibility\b",
        r"\bnot a given\b",
        r"\bcannot be ruled out\b",
        r"\bdoor is open\b",
        r"\bwould be driven by\b",
        r"\blikely to\b",
        r"\bpotential(?:ly)?\b"
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def is_historical(text):
    """Identify passages explicitly describing past actions."""

    text = (text or "").lower()

    current_hold_patterns = [
        r"\bkept\b.{0,80}\bunchanged\b",
        r"\bleft\b.{0,80}\bunchanged\b",
        r"\bheld its\b.{0,50}\brate\b",
        r"\bmaintained\b.{0,50}\brate\b"
    ]

    if any(
        re.search(pattern, text)
        for pattern in current_hold_patterns
    ):
        return False

    patterns = [
        r"\bhad (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\bpreviously (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\bwas previously (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\b(?:cut|reduced|raised|hiked|lowered|increased)\b"
        r".{0,60}\b(?:in 20\d{2}|last year|in the past)\b",
        r"\b(?:first|second) half of 20\d{2}\b"
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def analyse_temporal_stance(
    claim,
    passage,
    publication_date=None
):
    """
    Compare a claim with a retrieved evidence passage.

    Publication date is source metadata. It is not automatically
    treated as the date of the event described in the passage.
    """

    claim = claim or ""
    passage = passage or ""

    claim_dates = extract_event_dates(claim)
    passage_dates = extract_event_dates(passage)

    weekday_dates = resolve_weekday_dates(
        passage,
        publication_date
    )

    all_passage_dates = sorted(
        set(passage_dates + weekday_dates)
    )

    parsed_publication_date = parse_publication_date(
        publication_date
    )

    claim_event_date = None

    if claim_dates:
        claim_event_date = date.fromisoformat(claim_dates[0])

    source_predates_claim = (
        parsed_publication_date is not None
        and claim_event_date is not None
        and parsed_publication_date < claim_event_date
    )

    claim_action = detect_action(claim)
    passage_action = detect_action(passage)

    historical = is_historical(passage)
    prediction = is_prediction(passage)

    # Compare possible event dates.
    if claim_dates and all_passage_dates:
        if set(claim_dates).intersection(set(all_passage_dates)):
            temporal_relation = "same_date"
        else:
            temporal_relation = "different_date"

    elif all_passage_dates:
        temporal_relation = "claim_date_not_found"

    else:
        temporal_relation = "passage_date_not_found"

    # Classify the evidence.
    if prediction:
        stance = "prediction"
        evidence_relation = "forecast_not_confirmation"

    elif historical:
        stance = "historical_reference"
        evidence_relation = "historical_context"

    elif temporal_relation == "different_date":
        if (
            claim_action != "unclear"
            and passage_action != "unclear"
            and claim_action != passage_action
        ):
            stance = "contradicts_action"
        else:
            stance = "same_action_different_date"

        evidence_relation = "different_event_date"

    elif (
        source_predates_claim
        and temporal_relation != "same_date"
        and claim_action != "unclear"
        and passage_action != "unclear"
    ):
        # Only apply this to passages describing an actual action.
        # Generic definitions and background explanations should
        # not be classified as earlier reporting.
        stance = "earlier_reporting"
        evidence_relation = "earlier_source_not_confirmation"

    elif (
        claim_action == "unclear"
        or passage_action == "unclear"
    ):
        stance = "unclear"
        evidence_relation = "needs_review"

    elif claim_action == passage_action:
        stance = "supports_action"

        if temporal_relation == "same_date":
            evidence_relation = "potential_support"
        else:
            evidence_relation = "matching_action_date_unconfirmed"

    else:
        stance = "contradicts_action"

        if temporal_relation == "same_date":
            evidence_relation = "potential_contradiction"
        else:
            evidence_relation = "conflicting_action_date_unconfirmed"

    return {
        "claim_dates": claim_dates,
        "passage_dates": passage_dates,
        "resolved_weekday_dates": weekday_dates,
        "all_passage_dates": all_passage_dates,
        "date_provenance": {
            "explicit_passage_dates": passage_dates,
            "weekday_estimates": weekday_dates,
            "publication_date": (
                parsed_publication_date.isoformat()
                if parsed_publication_date else None
            )
        },
        "temporal_relation": temporal_relation,
        "claim_action": claim_action,
        "passage_action": passage_action,
        "is_historical": historical,
        "is_prediction": prediction,
        "source_publication_date": publication_date,
        "source_predates_claim": source_predates_claim,
        "stance": stance,
        "evidence_relation": evidence_relation
    }
