
def classify_evidence_relation(passage):
    """
    Map temporal-analysis output into an evidence category.

    Only same-date potential support and contradiction count
    as direct evidence.
    """

    relation = passage.get("evidence_relation")

    categories = {
        "potential_support": "supporting",
        "potential_contradiction": "contradicting",
        "historical_context": "historical",
        "forecast_not_confirmation": "predictions",
        "different_event_date": "different_date",
        "earlier_source_not_confirmation": "earlier_reporting",
        "matching_action_date_unconfirmed": "unresolved",
        "conflicting_action_date_unconfirmed": "unresolved",
        "needs_review": "unresolved"
    }

    return categories.get(relation, "unresolved")


def assess_evidence(evidence_records):
    """Assess retrieval coverage and summarise extracted evidence."""

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
        if record.get("source_class") == "research"
    )

    news_sources = sum(
        1 for record in evidence_records
        if record.get("source_class") == "established_news"
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
        if record.get("publication_date_source") == "search_result"
    )

    # Collect all extracted passages.
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

    total_passages = 0

    for record in extracted_records:
        for passage in record.get("relevant_passages", []):

            total_passages += 1

            category = classify_evidence_relation(passage)

            evidence_counts[category] += 1

            source_identifier = (
                record.get("url")
                or record.get("domain")
                or record.get("title")
            )

            if source_identifier:
                evidence_sources[category].add(source_identifier)

    evidence_summary = {}

    for category, count in evidence_counts.items():
        evidence_summary[category] = {
            "passage_count": count,
            "source_count": len(evidence_sources[category])
        }

    # Assess retrieval breadth, not truth.
    reasons = []

    if len(extracted_records) >= 3 and len(unique_domains) >= 2:
        coverage_level = "Moderate"
        reasons.append(
            "Multiple webpages were retrieved from multiple domains."
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

    if publication_dates_available == 0:
        reasons.append(
            "Publication dates are unavailable; temporal relevance "
            "cannot yet be assessed."
        )

    if evidence_counts["supporting"] == 0:
        reasons.append(
            "No direct same-date supporting passage was identified."
        )

    if evidence_counts["contradicting"] == 0:
        reasons.append(
            "No direct same-date contradicting passage was identified."
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

        "publication_dates_available": publication_dates_available,
        "page_metadata_dates": page_metadata_dates,
        "search_result_dates": search_result_dates,

        "total_passages_analysed": total_passages,
        "evidence_summary": evidence_summary,

        "reasons": reasons,

        "warning": (
            "Coverage measures retrieval breadth, not truth. "
            "Evidence categories are preliminary rule-based assessments. "
            "Source independence and factual accuracy still require review."
        )
    }
