import re


def classify_evidence_relation(passage):
    """
    Map temporal-analysis output into an evidence category.
    Direct supporting passages and contemporary reports count as supporting.
    Opposing actions and debunking statements count as contradicting.
    """
    relation = passage.get("evidence_relation")

    categories = {
        "potential_support": "supporting",
        "contemporary_support": "supporting",
        "matching_action_date_unconfirmed": "supporting",
        "potential_contradiction": "contradicting",
        "conflicting_action_date_unconfirmed": "contradicting",
        "conflicting_action": "contradicting",
        "historical_context": "historical",
        "forecast_not_confirmation": "predictions",
        "different_event_date": "different_date",
        "earlier_source_not_confirmation": "earlier_reporting",
        "needs_review": "unresolved"
    }

    return categories.get(relation, "unresolved")


def are_passages_syndicated(text_a, text_b):
    """Check if two texts are duplicates from the same syndicated wire report."""
    if not text_a or not text_b:
        return False
    words_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
    words_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
    if not words_a or not words_b:
        return False
    overlap = len(words_a.intersection(words_b))
    union = len(words_a.union(words_b))
    return (overlap / union) > 0.85


def assess_evidence(evidence_records):
    """
    Assess retrieval coverage, source credibility, independence, and
    evidence stance across extracted records.
    """
    total_sources = len(evidence_records)

    extracted_records = [
        record for record in evidence_records
        if record.get("extraction_success")
    ]

    unique_domains = {
        record.get("domain")
        for record in evidence_records
        if record.get("domain")
    }

    official_primary = sum(
        1 for record in evidence_records
        if record.get("source_class") == "official_primary"
    )

    official_secondary = sum(
        1 for record in evidence_records
        if record.get("source_class") == "official_secondary"
    )

    research_sources = sum(
        1 for record in evidence_records
        if record.get("source_class") in ("research", "research_expert")
    )

    news_sources = sum(
        1 for record in evidence_records
        if record.get("source_class") == "established_news"
    )

    social_media_sources = sum(
        1 for record in evidence_records
        if record.get("source_class") == "social_media"
    )

    publication_dates_available = sum(
        1 for record in evidence_records
        if record.get("publication_date")
    )

    page_metadata_dates = sum(
        1 for record in evidence_records
        if record.get("publication_date_source") == "page_metadata"
    )

    search_result_dates = sum(
        1 for record in evidence_records
        if record.get("publication_date_source") in ("search_result", "tavily_extract")
    )

    # Collect extracted passages and group by category
    evidence_counts = {
        "supporting": 0,
        "contradicting": 0,
        "historical": 0,
        "predictions": 0,
        "different_date": 0,
        "earlier_reporting": 0,
        "unresolved": 0
    }

    evidence_sources = {
        category: set()
        for category in evidence_counts
    }

    primary_evidence_sources = {
        category: set()
        for category in evidence_counts
    }

    total_passages = 0
    seen_passages = []
    syndication_detected = False

    for record in extracted_records:
        src_class = record.get("source_class", "other_secondary")
        source_identifier = (
            record.get("domain")
            or record.get("url")
            or record.get("title")
        )

        for passage in record.get("relevant_passages", []):
            p_text = passage.get("passage", "")
            # Check for syndication
            for prev_text in seen_passages:
                if are_passages_syndicated(p_text, prev_text):
                    syndication_detected = True
                    break
            seen_passages.append(p_text)

            total_passages += 1
            category = classify_evidence_relation(passage)
            evidence_counts[category] += 1

            if source_identifier:
                evidence_sources[category].add(source_identifier)
                if src_class == "official_primary":
                    primary_evidence_sources[category].add(source_identifier)

    evidence_summary = {}
    for category, count in evidence_counts.items():
        evidence_summary[category] = {
            "passage_count": count,
            "source_count": len(evidence_sources[category]),
            "primary_source_count": len(primary_evidence_sources[category]),
        }

    # Assess retrieval coverage
    reasons = []

    if official_primary > 0 and len(extracted_records) >= 1:
        coverage_level = "High"
        reasons.append(
            f"Retrieved {official_primary} official primary source(s) with extracted content."
        )
    elif len(extracted_records) >= 3 and len(unique_domains) >= 2:
        coverage_level = "Moderate"
        reasons.append(
            "Multiple webpages were retrieved from multiple independent domains."
        )
    elif len(extracted_records) >= 1:
        coverage_level = "Limited"
        reasons.append(
            "Some webpages were retrieved, but coverage is limited."
        )
    else:
        coverage_level = "Insufficient"
        reasons.append(
            "No usable webpage content was retrieved."
        )

    if official_primary == 0:
        reasons.append(
            "No recognised primary-source domain was found."
        )

    if syndication_detected:
        reasons.append(
            "Multiple sources appear to repeat the same syndicated wire report; treating as single reporting thread."
        )

    if publication_dates_available == 0:
        reasons.append(
            "Publication dates are unavailable; temporal relevance cannot yet be fully confirmed."
        )

    if evidence_counts["supporting"] == 0:
        reasons.append(
            "No direct supporting passage was identified."
        )
    else:
        reasons.append(
            f"Identified {evidence_counts['supporting']} supporting passage(s) across {len(evidence_sources['supporting'])} source(s)."
        )

    if evidence_counts["contradicting"] == 0:
        reasons.append(
            "No direct contradicting passage was identified."
        )
    else:
        reasons.append(
            f"Identified {evidence_counts['contradicting']} contradicting passage(s) across {len(evidence_sources['contradicting'])} source(s)."
        )

    return {
        "coverage_level": coverage_level,
        "total_sources_found": total_sources,
        "pages_extracted": len(extracted_records),
        "unique_domains": len(unique_domains),
        "official_primary_sources": official_primary,
        "official_secondary_sources": official_secondary,
        "research_sources": research_sources,
        "news_sources": news_sources,
        "social_media_sources": social_media_sources,
        "publication_dates_available": publication_dates_available,
        "page_metadata_dates": page_metadata_dates,
        "search_result_dates": search_result_dates,
        "total_passages_analysed": total_passages,
        "evidence_summary": evidence_summary,
        "supporting_sources_count": len(evidence_sources["supporting"]),
        "contradicting_sources_count": len(evidence_sources["contradicting"]),
        "primary_supporting_sources_count": len(primary_evidence_sources["supporting"]),
        "primary_contradicting_sources_count": len(primary_evidence_sources["contradicting"]),
        "reasons": reasons,
        "warning": (
            "Coverage measures retrieval breadth, not truth. "
            "Evidence categories are preliminary rule-based assessments. "
            "Source independence and factual accuracy still require review."
        )
    }
