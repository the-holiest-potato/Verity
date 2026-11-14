
import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient


# Load .env from the project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Add it to the project's .env file."
    )

client = TavilyClient(api_key=api_key)


def search_claim(claim, max_results=5):
    """
    Search the web for evidence related to one extracted claim.
    Returns a list of source records.
    """

    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty.")

    response = client.search(
        query=claim,
        search_depth="basic",
        topic="general",
        max_results=max_results
    )

    sources = []

    for result in response.get("results", []):
        sources.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "published_date": result.get("published_date"),
            "relevance_score": result.get("score")
        })

    return sources
