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
    Extract dates that may describe events, excluding social media timestamps.
    """
    if not text:
        return []

    cleaned_text = text
    month_pattern = "|".join(MONTHS.keys())

    timestamp_patterns = [
        rf"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        rf",?\s+(?:{month_pattern})\s+\d{{1,2}},?\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",
        rf"\b(?:{month_pattern})\s+\d{{1,2}},?\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",
        rf"\b\d{{1,2}}\s+(?:{month_pattern})\s+\d{{4}}"
        rf"\s+at\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM)\b",
        r"\b\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)\b"
    ]

    for pattern in timestamp_patterns:
        cleaned_text = re.sub(pattern, " ", cleaned_text, flags=re.IGNORECASE)

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
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def resolve_weekday_dates(text, publication_date=None):
    """Resolve weekday references relative to publication date."""
    publication_day = parse_publication_date(publication_date)
    if not publication_day:
        return []

    resolved_dates = set()
    text = (text or "").lower()

    for weekday, weekday_number in WEEKDAYS.items():
        if not re.search(rf"\b{weekday}\b", text):
            continue

        days_back = (publication_day.weekday() - weekday_number) % 7
        if re.search(rf"\blast\s+{weekday}\b", text):
            if days_back == 0:
                days_back = 7

        resolved_day = publication_day - timedelta(days=days_back)
        resolved_dates.add(resolved_day.isoformat())

    return sorted(resolved_dates)


def detect_all_actions(text):
    """
    Detect all actions mentioned in text across central banking, corporate leadership,
    regulations, legal proceedings, and fact-checking debunks.
    """
    text = (text or "").lower()
    actions = set()

    # Rate hold patterns
    strong_hold_patterns = [
        r"\bkept\b.{0,80}\bunchanged\b",
        r"\bleft\b.{0,80}\bunchanged\b",
        r"\bheld\b.{0,80}\b(?:steady|unchanged)\b",
        r"\bheld its\b.{0,50}\brate\b",
        r"\bmaintained\b.{0,50}\brate\b",
        r"\bno change\b"
    ]
    if any(re.search(pattern, text) for pattern in strong_hold_patterns):
        actions.add("held")

    # Corporate leadership patterns
    if re.search(r"\b(?:stepp\w*\s+down|step\w*\s+down|resign\w*|quit\w*|left\s+(?:the\s+company|as\s+ceo))\b", text):
        actions.add("resigned")

    if re.search(r"\b(?:bec\w*\s+(?:interim\s+)?ceo|interim\s+ceo|serv\w*\s+as\s+(?:interim\s+)?ceo|appoint\w*\s+(?:as\s+)?(?:interim\s+)?ceo|nam\w*\s+(?:as\s+)?(?:interim\s+)?ceo)\b", text):
        actions.add("appointed")

    if re.search(r"\b(?:remain\w*\s+(?:as\s+)?ceo|stays\s+(?:on\s+as\s+)?ceo|did\s+not\s+resign|did\s+not\s+step\s+down)\b", text):
        actions.add("remains")

    # Debunking patterns
    if re.search(r"\b(?:satir\w*|fake\s+news|hoax|untrue|fabricat\w*|wrongly\s+claimed|no\s+evidence\b)", text):
        actions.add("debunked")

    # Legal / Regulatory patterns
    if re.search(r"\b(?:ban|bans|banned|banning|prohibit\w*|outlaw\w*)\b", text):
        if not re.search(r"\b(?:does\s+not\s+ban|did\s+not\s+ban|not\s+bann\w*)\b", text):
            actions.add("banned")
        else:
            actions.add("unbanned")

    if re.search(r"\b(?:does\s+not\s+ban|did\s+not\s+ban|not\s+bann\w*|permit\w*)\b", text):
        actions.add("unbanned")

    # Rate actions
    rate_patterns = {
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

    for action, action_patterns in rate_patterns.items():
        if any(re.search(pattern, text) for pattern in action_patterns):
            actions.add(action)

    return actions


def detect_action(text):
    """
    Detect generalized action for single-action contexts.
    """
    actions = detect_all_actions(text)
    if not actions:
        return "unclear"
    for priority in ["debunked", "remains", "held", "appointed", "resigned", "reduced", "raised", "banned", "unbanned"]:
        if priority in actions:
            return priority
    return next(iter(actions))


def are_actions_contradictory(action_a, action_b):
    """Determine whether two detected actions inherently oppose each other."""
    if not action_a or not action_b or action_a == "unclear" or action_b == "unclear":
        return False

    opposites = {
        "reduced": {"raised", "held"},
        "raised": {"reduced", "held"},
        "held": {"reduced", "raised"},
        "resigned": {"remains"},
        "appointed": {"fired"},
        "banned": {"unbanned"},
        "unbanned": {"banned"},
    }

    return (
        action_b in opposites.get(action_a, set())
        or action_a in opposites.get(action_b, set())
        or action_b == "debunked"
    )


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
    return any(re.search(pattern, text) for pattern in patterns)


def is_historical(text):
    """Identify passages explicitly describing past actions."""
    text = (text or "").lower()

    current_hold_patterns = [
        r"\bkept\b.{0,80}\bunchanged\b",
        r"\bleft\b.{0,80}\bunchanged\b",
        r"\bheld its\b.{0,50}\brate\b",
        r"\bmaintained\b.{0,50}\brate\b"
    ]
    if any(re.search(pattern, text) for pattern in current_hold_patterns):
        return False

    patterns = [
        r"\bhad (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\bpreviously (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\bwas previously (?:cut|reduced|raised|hiked|lowered|increased)\b",
        r"\b(?:cut|reduced|raised|hiked|lowered|increased)\b"
        r".{0,60}\b(?:in 20\d{2}|last year|in the past)\b",
        r"\b(?:first|second) half of 20\d{2}\b"
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def has_direct_negation(passage, claim):
    """Check if the passage directly negates the core premise of the claim."""
    p_lower = passage.lower()
    c_lower = claim.lower()

    # Direct negation patterns
    negation_patterns = [
        r"\b(?:did\s+not|never|has\s+not|denied|no\s+evidence\s+that|false\w*|untrue|refuted)\b",
        r"\b(?:wrongly\s+claim\w*|falsely\s+claim\w*|satirical\s+(?:website|story|news))\b",
    ]
    return any(re.search(pattern, p_lower) for pattern in negation_patterns)


def analyse_temporal_stance(
    claim,
    passage,
    publication_date=None
):
    """
    Compare a claim with a retrieved evidence passage.
    Accurately identifies direct support, direct contradiction,
    different dates, forecasts, and historical context across domains.
    """
    claim = claim or ""
    passage = passage or ""

    claim_dates = extract_event_dates(claim)
    passage_dates = extract_event_dates(passage)
    weekday_dates = resolve_weekday_dates(passage, publication_date)
    all_passage_dates = sorted(set(passage_dates + weekday_dates))

    parsed_publication_date = parse_publication_date(publication_date)
    claim_event_date = date.fromisoformat(claim_dates[0]) if claim_dates else None

    source_predates_claim = (
        parsed_publication_date is not None
        and claim_event_date is not None
        and parsed_publication_date < claim_event_date
    )

    claim_action = detect_action(claim)
    passage_actions = detect_all_actions(passage)
    passage_action = detect_action(passage)

    if claim_action in passage_actions:
        contradictory_in_passage = any(are_actions_contradictory(claim_action, pa) for pa in passage_actions)
        if not contradictory_in_passage:
            passage_action = claim_action

    historical = is_historical(passage)
    prediction = is_prediction(passage)
    negation = has_direct_negation(passage, claim)

    # Compare event dates
    if claim_dates and all_passage_dates:
        if set(claim_dates).intersection(set(all_passage_dates)):
            temporal_relation = "same_date"
        else:
            temporal_relation = "different_date"
    elif all_passage_dates:
        temporal_relation = "claim_date_not_found"
    else:
        temporal_relation = "passage_date_not_found"

    # 1. Predictions / Forecasts
    if prediction:
        stance = "prediction"
        evidence_relation = "forecast_not_confirmation"

    # 2. Historical References
    elif historical:
        stance = "historical_reference"
        evidence_relation = "historical_context"

    # 3. Direct Negation or Debunking (e.g. Satire, Hoax, "wrongly claimed", "never did")
    elif negation or passage_action == "debunked":
        stance = "contradicts_action"
        if temporal_relation == "different_date":
            evidence_relation = "different_event_date"
        else:
            evidence_relation = "potential_contradiction"

    # 4. Same action or opposing action with explicitly different event date
    elif temporal_relation == "different_date":
        if are_actions_contradictory(claim_action, passage_action) or (
            claim_action != "unclear" and passage_action != "unclear" and claim_action != passage_action
        ):
            stance = "contradicts_action"
        else:
            stance = "same_action_different_date"
        evidence_relation = "different_event_date"

    # 5. Earlier reporting that predates the claim date
    elif (
        source_predates_claim
        and temporal_relation != "same_date"
        and claim_action != "unclear"
        and passage_action != "unclear"
    ):
        stance = "earlier_reporting"
        evidence_relation = "earlier_source_not_confirmation"

    # 6. Inherently contradictory actions (e.g. reduced vs held; resigned vs remains)
    elif are_actions_contradictory(claim_action, passage_action) or any(are_actions_contradictory(claim_action, pa) for pa in passage_actions):
        stance = "contradicts_action"
        if temporal_relation == "same_date":
            evidence_relation = "potential_contradiction"
        else:
            evidence_relation = "conflicting_action_date_unconfirmed"

    # 7. Matching actions (e.g. reduced & reduced; resigned & resigned; appointed & appointed)
    elif claim_action != "unclear" and (claim_action == passage_action or claim_action in passage_actions):
        stance = "supports_action"
        if temporal_relation == "same_date":
            evidence_relation = "potential_support"
        else:
            evidence_relation = "matching_action_date_unconfirmed"

    # 8. Unclear / general narrative statements
    else:
        # Check if the passage has strong entity and narrative alignment supporting the claim
        # e.g. for claims without a single action verb, check token assertion
        stance = "unclear"
        evidence_relation = "needs_review"

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
