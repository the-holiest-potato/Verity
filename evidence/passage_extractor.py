
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "of", "to", "in", "on", "at", "by", "for", "from",
    "with", "and", "or", "but", "as", "that", "this", "it", "its",
    "their", "they", "he", "she", "we", "you", "i", "will",
    "would", "could", "should", "may", "might", "can", "has",
    "have", "had", "after", "before", "during", "about", "into",
    "than", "also", "according", "reportedly", "reports"
}

BOILERPLATE_PATTERNS = [
    r"\bsubscribe\b",
    r"\bsign in\b",
    r"\blog in\b",
    r"\bread more\b",
    r"\bwatch now\b",
    r"\bclick here\b",
    r"\ball rights reserved\b",
    r"\bterms and conditions\b",
    r"\bprivacy policy\b",
    r"\bfollow us\b",
]


def clean_article_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    if not text:
        return []

    words = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def split_into_sentences(text):
    if not text:
        return []

    text = clean_article_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)

    return [
        part.strip(" \n\t-•")
        for part in parts
        if 20 <= len(part.strip()) <= 700
    ]


def extract_date_strings(text):
    if not text:
        return []

    patterns = [
        r"\b\d{1,2}\s+[A-Za-z]+\s+20\d{2}\b",
        r"\b[A-Za-z]+\s+\d{1,2},?\s+20\d{2}\b",
        r"\b20\d{2}-\d{2}-\d{2}\b",
    ]

    dates = []

    for pattern in patterns:
        dates.extend(
            re.findall(pattern, text, flags=re.IGNORECASE)
        )

    return [item.lower() for item in dates]


def extract_action_terms(text):
    text = (text or "").lower()

    patterns = {
        "reduced": r"\b(?:reduc\w*|lower\w*|cut|cuts|cutting|eas\w*)\b",
        "raised": r"\b(?:rais\w*|hike\w*|increas\w*)\b",
        "held": r"\b(?:held|hold|holds|unchanged|maintain\w*|steady)\b",
    }

    return [
        action for action, pattern in patterns.items()
        if re.search(pattern, text)
    ]


def calculate_relevance(claim_tokens, passage_tokens):
    """
    Calculate passage relevance using token sets.

    This is a retrieval score, not a truth probability.
    """
    if not claim_tokens or not passage_tokens:
        return 0.0

    overlap = claim_tokens.intersection(passage_tokens)

    if not overlap:
        return 0.0

    precision = len(overlap) / len(passage_tokens)
    recall = len(overlap) / len(claim_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_context_bonus(claim, passage):
    """
    Reward exact subject, date and event context.
    Do not reward only matching actions: opposing evidence is relevant too.
    """
    claim_lower = claim.lower()
    passage_lower = passage.lower()

    bonus = 0.0

    important_phrases = [
        "repo rate",
        "policy repo rate",
        "reserve bank of india",
        "monetary policy committee",
        "interest rate",
        "rate cut",
        "rate hike",
        "basis points",
    ]

    for phrase in important_phrases:
        if phrase in claim_lower and phrase in passage_lower:
            bonus += 0.12

    # Penalise related but different RBI rates.
    if "repo rate" in claim_lower:
        if (
            "sdf rate" in passage_lower
            or "standing deposit facility" in passage_lower
        ):
            bonus -= 0.25

        if (
            "msf rate" in passage_lower
            or "marginal standing facility" in passage_lower
        ):
            bonus -= 0.25

    claim_dates = extract_date_strings(claim)
    passage_dates = extract_date_strings(passage)

    if claim_dates and passage_dates:
        if any(item in passage_dates for item in claim_dates):
            bonus += 0.15

    claim_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?\b", claim_lower)
    )
    passage_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?\b", passage_lower)
    )

    if claim_numbers.intersection(passage_numbers):
        bonus += 0.05

    if extract_action_terms(claim) and extract_action_terms(passage):
        bonus += 0.10

    return bonus


def extract_relevant_passages(claim, article_content, top_k=5):
    if not claim or not article_content:
        return []

    claim_tokens = set(tokenize(claim))
    sentences = split_into_sentences(article_content)

    results = []
    seen = set()

    for sentence in sentences:
        normalised = re.sub(
            r"\s+", " ", sentence.lower()
        ).strip()

        if normalised in seen:
            continue

        seen.add(normalised)

        passage_tokens = set(tokenize(sentence))
        matched_terms = sorted(
            claim_tokens.intersection(passage_tokens)
        )

        if len(matched_terms) < 2:
            continue

        base_score = calculate_relevance(
            claim_tokens,
            passage_tokens
        )

        score = base_score + calculate_context_bonus(
            claim,
            sentence
        )

        score = max(0.0, min(score, 1.0))

        if score < 0.05:
            continue

        results.append({
            "passage": sentence,
            "relevance_score": round(score, 3),
            "matched_terms": matched_terms,
        })

    results.sort(
        key=lambda item: item["relevance_score"],
        reverse=True
    )

    return results[:top_k]
