
from trafilatura import fetch_url, extract, extract_metadata

from evidence.search import client


def extract_page(url):
    """
    Fetch a webpage and extract its text and available metadata.

    First attempts direct extraction with Trafilatura.
    If that fails, falls back to Tavily Extract.
    """

    # Attempt 1: Direct webpage extraction
    try:
        downloaded = fetch_url(url)

        if downloaded:
            metadata = extract_metadata(downloaded)

            content = extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True
            )

            if content:
                return {
                    "success": True,
                    "url": url,
                    "title": metadata.title if metadata else None,
                    "author": metadata.author if metadata else None,
                    "published_date": metadata.date if metadata else None,
                    "sitename": metadata.sitename if metadata else None,
                    "date_source": "page_metadata"
                    if metadata and metadata.date else None,
                    "content": content[:12000]
                }

    except Exception as error:
        print(f"Direct extraction failed for {url}: {error}")

    # Attempt 2: Tavily Extract fallback
    try:
        response = client.extract(
            urls=[url],
            extract_depth="basic",
            format="text"
        )

        results = response.get("results", [])

        if results:
            result = results[0]
            content = result.get("raw_content", "")

            if content:
                published_date = result.get("published_date")

                return {
                    "success": True,
                    "url": result.get("url", url),
                    "title": result.get("title"),
                    "author": result.get("author"),
                    "published_date": published_date,
                    "sitename": result.get("sitename"),
                    "date_source": "tavily_extract"
                    if published_date else None,
                    "content": content[:12000]
                }

    except Exception as error:
        print(f"Tavily extraction failed for {url}: {error}")

    # Both extraction methods failed
    return {
        "success": False,
        "url": url,
        "title": None,
        "author": None,
        "published_date": None,
        "sitename": None,
        "date_source": None,
        "content": ""
    }
