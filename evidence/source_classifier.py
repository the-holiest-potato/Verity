
from urllib.parse import urlparse


OFFICIAL_PRIMARY_DOMAINS = [
    "rbi.org.in",
    "pib.gov.in",
    "india.gov.in",
    "who.int",
    "un.org",
    "europa.eu",
]

OFFICIAL_SECONDARY_DOMAINS = [
    "newsonair.gov.in",
]

RESEARCH_DOMAINS = [
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org",
    "ieee.org",
    "acm.org",
]

NEWS_DOMAINS = [
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "apnews.com",
    "theguardian.com",
    "ndtv.com",
    "indianexpress.com",
    "thehindu.com",
    "cnbc.com",
    "aljazeera.com",
    "newsonair.gov.in",
]


def domain_matches(domain, known_domain):
    """Check whether a domain matches a known domain or its subdomain."""
    return (
        domain == known_domain
        or domain.endswith("." + known_domain)
    )


def classify_source(url):
    """
    Classify a source based on its domain.

    This identifies the type of publisher, not the truthfulness
    or reliability of an individual article.
    """

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()

    # Remove www. if present
    if domain.startswith("www."):
        domain = domain[4:]

    if not domain:
        return {
            "domain": None,
            "source_class": "other_unknown",
            "classification_reason": "Invalid or missing URL domain",
        }

    # 1. Official primary sources
    if (
        any(
            domain_matches(domain, d)
            for d in OFFICIAL_PRIMARY_DOMAINS
        )
        or domain.endswith(".nic.in")
    ):
        source_class = "official_primary"
        reason = "Official primary-source domain"

    # 2. Official secondary sources
    elif (
        any(
            domain_matches(domain, d)
            for d in OFFICIAL_SECONDARY_DOMAINS
        )
        or domain.endswith(".gov.in")
        or domain.endswith(".gov")
    ):
        source_class = "official_secondary"
        reason = "Official government publisher or reporting organisation"

    # 3. Research and expert publications
    elif any(
        domain_matches(domain, d)
        for d in RESEARCH_DOMAINS
    ):
        source_class = "research_expert"
        reason = "Recognised research or expert publication domain"

    # 4. Established news organisations
    elif any(
        domain_matches(domain, d)
        for d in NEWS_DOMAINS
    ):
        source_class = "established_news"
        reason = "Recognised news organisation domain"

    # 5. Everything else
    else:
        source_class = "other_unknown"
        reason = "Domain not present in the current classification list"

    return {
        "domain": domain,
        "source_class": source_class,
        "classification_reason": reason,
    }
