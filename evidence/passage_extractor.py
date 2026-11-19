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
    r"\bcookie settings\b",
    r"\ball rights reserved\b",
]

GENERAL_ACTION_PATTERNS = {
    # Leadership & Appointments
    "resigned": r"\b(?:resign\w*|stepp\w*\s+down|step\w*\s+down|quit\w*|retir\w*|vacat\w*)\b",
    "appointed": r"\b(?:appoint\w*|nam\w*|elect\w*|bec\w*\s+(?:ceo|interim|director|president)|assum\w*|promot\w*|serv\w*\s+as)\b",
    "fired": r"\b(?:fir\w*|dismiss\w*|remov\w*|terminat\w*|oust\w*)\b",
    "remains": r"\b(?:remain\w*|stay\w*|continu\w*|retain\w*|reaffirm\w*)\b",
    # Financial / Monetary
    "reduced": r"\b(?:reduc\w*|lower\w*|cut|cuts|cutting|eas\w*)\b",
    "raised": r"\b(?:rais\w*|hike\w*|increas\w*)\b",
    "held": r"\b(?:held|hold|holds|unchanged|maintain\w*|steady|no\s+change)\b",
    # Regulatory & Legal
    "banned": r"\b(?:ban|bans|banned|banning|prohibit\w*|outlaw\w*|bar\w*)\b",
    "settled": r"\b(?:settl\w*|plead\w*|agre\w*\s+to\s+pay|fine|fined|penalt\w*)\b",
    "approved": r"\b(?:approv\w*|pass\w*|enact\w*|authoriz\w*|ratif\w*)\b",
    # Debunking & Verification
    "debunked": r"\b(?:fals\w*|fake|hoax|untrue|fabricat\w*|satir\w*|wrongly|mislead\w*|no\s+evidence|myth)\b",
}


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
        dates.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return [item.lower() for item in dates]


def extract_action_terms(text):
    text = (text or "").lower()
    detected = []
    for action, pattern in GENERAL_ACTION_PATTERNS.items():
        if re.search(pattern, text):
            detected.append(action)
    return detected


def extract_key_entities(text):
    """Extract capitalized multi-word sequences, titles, and acronyms."""
    if not text:
        return []
    # Capitalized entity phrases (e.g. Jensen Huang, Colette Kress, Steve Jobs)
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    # High-signal acronyms (e.g. CEO, CFO, RBI, SEC, UN, WHO)
    acronyms = re.findall(r"\b(?:CEO|CFO|COO|CTO|RBI|SEC|MPC|FDA|WHO|UN|EDGAR)\b", text)
    # Distinct single capitalized names (e.g. Huang, Kress, Connery, Gaetz, Trump)
    single_names = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
    # Filter common sentence-starters from single names
    common_starters = {"The", "This", "That", "When", "What", "Where", "Which", "Some", "According", "Many"}
    filtered_names = [n for n in single_names if n not in common_starters]

    combined = [e.lower() for e in entities + acronyms + filtered_names]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def calculate_relevance(claim_tokens, passage_tokens):
    """Calculate token overlap F1 score between claim and passage."""
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
    Reward exact subject entity, date, numbers, and action predicates across all domains.
    """
    claim_lower = claim.lower()
    passage_lower = passage.lower()

    bonus = 0.0

    # 1. Subject entity matching (Names, Acronyms, Organizations)
    claim_entities = extract_key_entities(claim)
    if claim_entities:
        matched_entities = [e for e in claim_entities if e in passage_lower]
        if matched_entities:
            # Reward up to 0.35 for matching core entities like "Jensen Huang" or "CEO"
            bonus += min(0.35, len(matched_entities) * 0.15)

    # 2. Action predicate alignment (Reward both matching and directly opposing actions)
    claim_actions = extract_action_terms(claim)
    passage_actions = extract_action_terms(passage)
    if claim_actions and passage_actions:
        # If passage addresses the action (supporting or refuting), it is relevant
        bonus += 0.15

    # 3. Explicit debunking / fact-checking terms in passage
    debunk_patterns = r"\b(?:satir\w*|fake\s+news|hoax|untrue|fabricat\w*|no\s+evidence|wrongly\s+claimed)\b"
    if re.search(debunk_patterns, passage_lower):
        bonus += 0.20

    # 4. Domain-specific disambiguation (e.g. RBI rate precision)
    if "repo rate" in claim_lower:
        if (
            "sdf rate" in passage_lower
            or "standing deposit facility" in passage_lower
            or "msf rate" in passage_lower
            or "marginal standing facility" in passage_lower
        ):
            bonus -= 0.25

    # 5. Exact date string matching
    claim_dates = extract_date_strings(claim)
    passage_dates = extract_date_strings(passage)
    if claim_dates and passage_dates:
        if any(item in passage_dates for item in claim_dates):
            bonus += 0.15

    # 6. Specific numerical matches
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_lower))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", passage_lower))
    if claim_numbers and claim_numbers.intersection(passage_numbers):
        bonus += 0.08

    return bonus


def extract_relevant_passages(claim, article_content, top_k=5):
    """
    Extract and rank sentences from an article by relevance to the claim.
    """
    if not claim or not article_content:
        return []

    claim_tokens = set(tokenize(claim))
    sentences = split_into_sentences(article_content)

    results = []
    seen = set()

    for sentence in sentences:
        normalised = re.sub(r"\s+", " ", sentence.lower()).strip()
        if normalised in seen:
            continue
        seen.add(normalised)

        passage_tokens = set(tokenize(sentence))
        matched_terms = sorted(claim_tokens.intersection(passage_tokens))

        if len(matched_terms) < 1:
            continue

        base_score = calculate_relevance(claim_tokens, passage_tokens)
        score = base_score + calculate_context_bonus(claim, sentence)
        score = max(0.0, min(score, 1.0))

        # Filter out negligible relevance
        if score < 0.10:
            continue

        results.append({
            "passage": sentence,
            "relevance_score": round(score, 3),
            "matched_terms": matched_terms,
        })

    results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return results[:top_k]
