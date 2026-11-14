
import re

from evidence.search import search_claim
from evidence.page_fetcher import extract_page
from evidence.source_classifier import classify_source
from evidence.assessment import (
    assess_evidence,
    classify_evidence_relation
)
from evidence.passage_extractor import (
    extract_relevant_passages,
    split_into_sentences
)
from evidence.temporal_stance import (
    analyse_temporal_stance,
    extract_dates,
    detect_action
)


def has_rate_subject(text):
    pattern = (
        r"\brepo\s+rate\b"
        r"|\bpolicy\s+rate\b"
        r"|\breserve\s+bank\b"
        r"|\brbi\b"
        r"|\bmonetary\s+policy\s+committee\b"
        r"|\bmpc\b"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))


def is_event_context_sentence(text):
    """
    Identify sentences explicitly describing a rate decision,
    meeting or announcement.
    """

    pattern = (
        r"\bdecision\b"
        r"|\bmeeting\b"
        r"|\bannouncement\b"
        r"|\bannounced\b"
        r"|\bpolicy review\b"
        r"|\bmonetary policy\b"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))


def is_post_timestamp_sentence(text):
    """
    Identify social-media-style publication timestamps.

    These are recorded separately and are not automatically
    interpreted as dates of the underlying event.
    """

    return bool(
        re.search(
            r"\bat\s+\d{1,2}:\d{2}\s*(?:AM|PM)\b",
            text,
            re.IGNORECASE
        )
    )


def extract_source_event_date_candidates(article_content):
    """
    Extract dates associated with rate actions.

    Supports:
    1. Date and action in the same sentence.
    2. A decision-date sentence adjacent to an explicit
       rate-action sentence.

    Does not treat ordinary social media timestamps as
    event dates.
    """

    sentences = split_into_sentences(article_content)

    candidates = []

    for index, sentence in enumerate(sentences):

        dates = extract_dates(sentence)

        if not dates:
            continue

        # A social-media publication timestamp is not necessarily
        # the date of the event being described.
        if is_post_timestamp_sentence(sentence):
            continue

        if not has_rate_subject(sentence):
            continue

        action = detect_action(sentence)

        # Case 1: date and action in the same sentence.
        if action != "unclear":

            for date in dates:
                candidates.append({
                    "date": str(date),
                    "action": action,
                    "source_sentence": sentence,
                    "date_sentence": sentence,
                    "association_type": "same_sentence"
                })

            continue

        # Case 2: date appears in an event-context sentence.
        if not is_event_context_sentence(sentence):
            continue

        nearby_actions = []

        for neighbour_index in (index - 1, index + 1):

            if neighbour_index < 0:
                continue

            if neighbour_index >= len(sentences):
                continue

            neighbour = sentences[neighbour_index]

            if not has_rate_subject(neighbour):
                continue

            neighbour_action = detect_action(neighbour)

            if neighbour_action != "unclear":
                nearby_actions.append({
                    "action": neighbour_action,
                    "sentence": neighbour
                })

        # Only inherit an action when the nearby context is unique.
        unique_actions = {
            item["action"]
            for item in nearby_actions
        }

        if len(unique_actions) != 1:
            continue

        matched_action = next(iter(unique_actions))

        matching_sentence = next(
            item["sentence"]
            for item in nearby_actions
            if item["action"] == matched_action
        )

        for date in dates:
            candidates.append({
                "date": str(date),
                "action": matched_action,
                "source_sentence": matching_sentence,
                "date_sentence": sentence,
                "association_type": "adjacent_event_context"
            })

    # Deduplicate while preserving order.
    unique_candidates = []
    seen = set()

    for candidate in candidates:

        key = (
            candidate["date"],
            candidate["action"],
            candidate["association_type"]
        )

        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


def extract_source_post_dates(article_content):
    """
    Find dates appearing in social-media-style timestamp sentences.

    These are metadata/context dates, not confirmed event dates.
    """

    post_dates = []

    for sentence in split_into_sentences(article_content):

        if not is_post_timestamp_sentence(sentence):
            continue

        dates = extract_dates(sentence)

        for date in dates:
            post_dates.append({
                "date": str(date),
                "source_sentence": sentence,
                "date_type": "post_timestamp"
            })

    return post_dates


def find_matching_source_context_date(
    passage_text,
    passage_action,
    source_date_candidates
):
    """
    Inherit an event date only when exactly one date matches
    the action and relevant rate subject.
    """

    if not passage_action:
        return None

    if extract_dates(passage_text):
        return None

    if not has_rate_subject(passage_text):
        return None

    matching_candidates = [
        candidate
        for candidate in source_date_candidates
        if candidate["action"] == passage_action
    ]

    matching_dates = {
        candidate["date"]
        for candidate in matching_candidates
    }

    if len(matching_dates) != 1:
        return None

    matched_date = next(iter(matching_dates))

    return next(
        candidate
        for candidate in matching_candidates
        if candidate["date"] == matched_date
    )


def investigate_claim(claim, max_sources=5):
    """
    Investigate a claim using search, page extraction,
    source classification, passage extraction and temporal analysis.
    """

    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty.")

    search_results = search_claim(
        claim,
        max_results=max_sources
    )

    evidence_records = []

    for result in search_results:

        url = result.get("url")

        if not url:
            continue

        page = extract_page(url)

        source_info = classify_source(url)

        article_content = page.get("content", "")

        extraction_success = (
            page.get("success", False)
            and bool(article_content)
        )

        # Publication date and its provenance.
        page_publication_date = page.get("published_date")
        search_publication_date = result.get("published_date")

        if page_publication_date:
            publication_date = page_publication_date
            publication_date_source = page.get("date_source")

        elif search_publication_date:
            publication_date = search_publication_date
            publication_date_source = "search_result"

        else:
            publication_date = None
            publication_date_source = None

        # Extract source-level event and post dates.
        if extraction_success:

            source_date_candidates = (
                extract_source_event_date_candidates(article_content)
            )

            source_post_dates = (
                extract_source_post_dates(article_content)
            )

            passages = extract_relevant_passages(
                claim,
                article_content,
                top_k=5
            )

        else:

            source_date_candidates = []
            source_post_dates = []
            passages = []

        # Analyse each relevant passage.
        for passage in passages:

            original_text = passage["passage"]

            initial_analysis = analyse_temporal_stance(
                claim=claim,
                passage=original_text,
                publication_date=publication_date
            )

            passage.update(initial_analysis)

            context_candidate = find_matching_source_context_date(
                passage_text=original_text,
                passage_action=passage.get("passage_action"),
                source_date_candidates=source_date_candidates
            )

            if context_candidate:

                enriched_text = (
                    original_text
                    + " "
                    + context_candidate["date_sentence"]
                    + " "
                    + context_candidate["source_sentence"]
                )

                contextual_analysis = analyse_temporal_stance(
                    claim=claim,
                    passage=enriched_text,
                    publication_date=publication_date
                )

                passage.update(contextual_analysis)

                passage["event_date_source"] = "source_context"
                passage["source_context_date"] = (
                    context_candidate["date"]
                )
                passage["source_context_sentence"] = (
                    context_candidate["date_sentence"]
                )
                passage["source_context_action_sentence"] = (
                    context_candidate["source_sentence"]
                )
                passage["date_association_type"] = (
                    context_candidate["association_type"]
                )

            else:

                explicit_dates = extract_dates(original_text)

                passage["event_date_source"] = (
                    "passage_text"
                    if explicit_dates
                    else "unknown"
                )

                passage["source_context_date"] = None
                passage["source_context_sentence"] = None
                passage["source_context_action_sentence"] = None
                passage["date_association_type"] = None

            passage["source_publication_date_source"] = (
                publication_date_source
            )

            passage["evidence_category"] = (
                classify_evidence_relation(passage)
            )

        evidence_record = {
            "title": (
                page.get("title")
                or result.get("title")
            ),

            "url": url,

            "domain": source_info.get("domain"),

            "source_class": source_info.get("source_class"),

            "classification_reason": source_info.get(
                "classification_reason"
            ),

            "publication_date": publication_date,
            "publication_date_source": publication_date_source,
            "page_published_date": page_publication_date,
            "search_published_date": search_publication_date,

            "author": page.get("author"),
            "sitename": page.get("sitename"),

            "search_snippet": result.get("content"),
            "search_relevance_score": result.get(
                "relevance_score"
            ),

            "extraction_success": extraction_success,
            "article_content": article_content[:12000],

            "source_event_date_candidates": source_date_candidates,
            "source_post_dates": source_post_dates,

            "relevant_passages": passages
        }

        evidence_records.append(evidence_record)

    assessment = assess_evidence(evidence_records)

    evidence_summary = {
        "supporting": [],
        "contradicting": [],
        "historical": [],
        "predictions": [],
        "different_date": [],
        "earlier_reporting": [],
        "unresolved": []
    }

    for record in evidence_records:

        for passage in record.get("relevant_passages", []):

            category = passage.get(
                "evidence_category",
                "unresolved"
            )

            if category not in evidence_summary:
                category = "unresolved"

            evidence_summary[category].append({

                "source_title": record.get("title"),
                "url": record.get("url"),
                "domain": record.get("domain"),
                "source_class": record.get("source_class"),

                "publication_date": record.get("publication_date"),

                "passage": passage.get("passage"),
                "relevance_score": passage.get("relevance_score"),

                "claim_action": passage.get("claim_action"),
                "passage_action": passage.get("passage_action"),

                "claim_dates": passage.get("claim_dates"),
                "passage_dates": passage.get("all_passage_dates"),

                "event_date_source": passage.get(
                    "event_date_source"
                ),

                "source_context_date": passage.get(
                    "source_context_date"
                ),

                "source_context_sentence": passage.get(
                    "source_context_sentence"
                ),

                "date_association_type": passage.get(
                    "date_association_type"
                ),

                "temporal_relation": passage.get(
                    "temporal_relation"
                ),

                "evidence_relation": passage.get(
                    "evidence_relation"
                )
            })

    return {
        "claim": claim,

        "sources_found": len(search_results),

        "pages_processed": len(evidence_records),

        "pages_extracted": sum(
            1
            for record in evidence_records
            if record["extraction_success"]
        ),

        "assessment": assessment,

        "evidence_summary": evidence_summary,

        "evidence": evidence_records
    }
