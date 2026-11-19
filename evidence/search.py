import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

from evidence.source_classifier import classify_source

# Load .env from the project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Add it to the project's .env file."
    )

client = TavilyClient(api_key=api_key)


def normalise_title(title):
    if not title:
        return ""
    return "".join(c.lower() for c in title if c.isalnum())


def search_claim(claim, max_results=5):
    """
    Search the web for evidence related to one extracted claim.
    Prioritises official primary sources, research, and reputable news.
    Filters out duplicates and demotes social media.
    """
    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty.")

    # Request extra candidates to allow ranking and deduplication
    fetch_count = min(12, max(max_results * 2, 6))

    try:
        response = client.search(
            query=claim,
            search_depth="advanced",
            topic="general",
            max_results=fetch_count,
        )
    except Exception:
        # Fall back to basic if advanced encounters issues
        response = client.search(
            query=claim,
            search_depth="basic",
            topic="general",
            max_results=fetch_count,
        )

    candidates = []
    seen_urls = set()
    seen_titles = set()

    for result in response.get("results", []):
        url = result.get("url")
        if not url:
            continue

        parsed = urlparse(url)
        clean_url = f"{parsed.netloc}{parsed.path}".rstrip("/")

        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        title = (result.get("title") or "").strip()
        norm_title = normalise_title(title)
        if norm_title and norm_title in seen_titles:
            continue
        if norm_title:
            seen_titles.add(norm_title)

        source_info = classify_source(url, claim=claim)
        relevance = result.get("score") or 0.5

        candidates.append({
            "title": title,
            "url": url,
            "content": result.get("content"),
            "published_date": result.get("published_date"),
            "relevance_score": relevance,
            "source_info": source_info,
            "tier": source_info["tier"],
        })

    # Sort primarily by source quality tier (1=Official Primary ... 6=Social Media),
    # secondarily by Tavily relevance score descending
    candidates.sort(key=lambda x: (x["tier"], -x["relevance_score"]))

    sources = []
    for item in candidates[:max_results]:
        sources.append({
            "title": item["title"],
            "url": item["url"],
            "content": item["content"],
            "published_date": item["published_date"],
            "relevance_score": item["relevance_score"],
            "source_tier": item["tier"],
            "source_class": item["source_info"]["source_class"],
        })

    return sources
