import re
from urllib.parse import urlparse

OFFICIAL_PRIMARY_DOMAINS = [
    # Central banks & financial regulators
    "rbi.org.in",
    "federalreserve.gov",
    "sec.gov",
    "edgar.sec.gov",
    "sebi.gov.in",
    "bankofengland.co.uk",
    "ecb.europa.eu",
    "bis.org",
    # Governments & official gazettes
    "pib.gov.in",
    "india.gov.in",
    "whitehouse.gov",
    "justice.gov",
    "treasury.gov",
    "state.gov",
    "fda.gov",
    "cdc.gov",
    "ftc.gov",
    # International governing bodies
    "who.int",
    "un.org",
    "europa.eu",
    "worldbank.org",
    "imf.org",
    "wto.org",
    "nato.int",
    "oecd.org",
]

OFFICIAL_SECONDARY_DOMAINS = [
    "newsonair.gov.in",
    "c-span.org",
    "pbs.org",
    "npr.org",
    "voanews.com",
    "dw.com",
    "france24.com",
    "bbc.co.uk",
]

RESEARCH_DOMAINS = [
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "ieee.org",
    "acm.org",
    "cell.com",
    "thelancet.com",
    "nejm.org",
    "biorxiv.org",
    "medrxiv.org",
    "pnas.org",
    "springer.com",
    "wiley.com",
    "jstor.org",
]

NEWS_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "theguardian.com",
    "economist.com",
    "cnbc.com",
    "aljazeera.com",
    "time.com",
    "forbes.com",
    "fortune.com",
    "cbsnews.com",
    "nbcnews.com",
    "abcnews.go.com",
    "cnn.com",
    "thehindu.com",
    "indianexpress.com",
    "ndtv.com",
    "livemint.com",
    "economictimes.indiatimes.com",
    "business-standard.com",
    "moneycontrol.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "yahoo.com",
    "finance.yahoo.com",
]

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "reddit.com",
    "threads.net",
    "linkedin.com",
    "pinterest.com",
    "t.me",
    "telegram.org",
    "whatsapp.com",
]

# Company entity mapping to official domains
COMPANY_OFFICIAL_DOMAINS = {
    "nvidia": ["nvidia.com", "nvidianews.nvidia.com", "investor.nvidia.com", "blogs.nvidia.com"],
    "apple": ["apple.com", "investor.apple.com", "newsroom.apple.com"],
    "microsoft": ["microsoft.com", "news.microsoft.com", "investor.microsoft.com"],
    "google": ["google.com", "blog.google", "abc.xyz"],
    "alphabet": ["abc.xyz", "google.com", "blog.google"],
    "tesla": ["tesla.com", "ir.tesla.com"],
    "meta": ["meta.com", "about.fb.com", "investor.fb.com"],
    "amazon": ["amazon.com", "aboutamazon.com", "ir.aboutamazon.com"],
    "openai": ["openai.com"],
}

TIER_ORDER = {
    "official_primary": 1,
    "official_secondary": 2,
    "research_expert": 3,
    "established_news": 4,
    "other_secondary": 5,
    "social_media": 6,
    "other_unknown": 7,
}


def domain_matches(domain, known_domain):
    """Check whether a domain matches a known domain or its subdomain."""
    if not domain or not known_domain:
        return False
    return (
        domain == known_domain
        or domain.endswith("." + known_domain)
    )


def extract_claim_entities(claim_text):
    """Extract known corporate, government, or institutional subjects from claim."""
    if not claim_text:
        return []
    claim_lower = claim_text.lower()
    entities = []
    for entity in COMPANY_OFFICIAL_DOMAINS:
        if re.search(rf"\b{re.escape(entity)}\b", claim_lower):
            entities.append(entity)
    return entities


def classify_source(url, claim=None):
    """
    Classify a source based on its domain and relevance to the claim.

    Prioritises:
    1. Official primary sources (government regulators, SEC EDGAR, official company newsrooms)
    2. Official secondary sources (government media, national public media)
    3. Research and expert publications (academic journals, scientific archives)
    4. Established independent news organisations
    5. Other secondary sources
    6. Social media / user-generated content (demoted)
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    if not domain:
        return {
            "domain": None,
            "source_class": "other_unknown",
            "tier": TIER_ORDER["other_unknown"],
            "classification_reason": "Invalid or missing URL domain",
            "is_primary": False,
            "is_social_media": False,
        }

    # Check for social media first to prevent social posts from masquerading as news
    if any(domain_matches(domain, d) for d in SOCIAL_MEDIA_DOMAINS):
        return {
            "domain": domain,
            "source_class": "social_media",
            "tier": TIER_ORDER["social_media"],
            "classification_reason": "User-generated content or social media platform",
            "is_primary": False,
            "is_social_media": True,
        }

    # Dynamic company official domain recognition
    if claim:
        entities = extract_claim_entities(claim)
        for entity in entities:
            official_domains = COMPANY_OFFICIAL_DOMAINS.get(entity, [])
            if any(domain_matches(domain, d) for d in official_domains):
                return {
                    "domain": domain,
                    "source_class": "official_primary",
                    "tier": TIER_ORDER["official_primary"],
                    "classification_reason": f"Official company announcement or newsroom domain for {entity.upper()}",
                    "is_primary": True,
                    "is_social_media": False,
                }

    # Official primary (Governments, Regulators, Central Banks, SEC)
    is_gov = (
        domain.endswith(".gov")
        or domain.endswith(".gov.in")
        or domain.endswith(".gov.uk")
        or domain.endswith(".gov.au")
        or domain.endswith(".nic.in")
        or domain.endswith(".mil")
    )

    if (
        any(domain_matches(domain, d) for d in OFFICIAL_PRIMARY_DOMAINS)
        or is_gov
    ):
        return {
            "domain": domain,
            "source_class": "official_primary",
            "tier": TIER_ORDER["official_primary"],
            "classification_reason": "Official government, regulatory, or primary institutional domain",
            "is_primary": True,
            "is_social_media": False,
        }

    # Research and expert publications
    is_edu = domain.endswith(".edu") or domain.endswith(".ac.uk") or domain.endswith(".edu.in")
    if (
        any(domain_matches(domain, d) for d in RESEARCH_DOMAINS)
        or is_edu
    ):
        return {
            "domain": domain,
            "source_class": "research_expert",
            "tier": TIER_ORDER["research_expert"],
            "classification_reason": "Recognised research, academic, or scientific publication domain",
            "is_primary": False,
            "is_social_media": False,
        }

    # Official secondary sources
    if any(domain_matches(domain, d) for d in OFFICIAL_SECONDARY_DOMAINS):
        return {
            "domain": domain,
            "source_class": "official_secondary",
            "tier": TIER_ORDER["official_secondary"],
            "classification_reason": "Official government publisher or national broadcasting organisation",
            "is_primary": False,
            "is_social_media": False,
        }

    # Established news organisations
    if any(domain_matches(domain, d) for d in NEWS_DOMAINS):
        return {
            "domain": domain,
            "source_class": "established_news",
            "tier": TIER_ORDER["established_news"],
            "classification_reason": "Recognised reputable news organisation domain",
            "is_primary": False,
            "is_social_media": False,
        }

    # Everything else
    return {
        "domain": domain,
        "source_class": "other_secondary",
        "tier": TIER_ORDER["other_secondary"],
        "classification_reason": "Secondary web publication domain",
        "is_primary": False,
        "is_social_media": False,
    }
