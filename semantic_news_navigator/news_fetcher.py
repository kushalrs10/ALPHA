"""
news_fetcher.py  —  Fetch news from NewsAPI or load fallback sample data.
"""

import json
import os
import requests
from typing import Optional


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/top-headlines"
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "sample_news.json")

CATEGORY_OPTIONS = [
    "general",
    "technology",
    "business",
    "sports",
    "health",
    "entertainment",
]


def fetch_news(api_key: str, category: str = "general", page_size: int = 50) -> tuple[list[dict], bool]:
    """
    Fetch top headlines from NewsAPI.

    Returns:
        (articles, used_fallback)
        articles     : list of normalised article dicts
        used_fallback: True if sample data was used instead of live API
    """
    if not api_key or api_key.strip() == "":
        return _load_fallback(filter_category=category), True

    params = {
        "apiKey": api_key.strip(),
        "category": category,
        "language": "en",
        "pageSize": min(page_size, 100),
        "country": "us",
    }

    try:
        response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            return _load_fallback(filter_category=category), True

        articles = _normalise_newsapi(data.get("articles", []))
        if len(articles) < 5:
            return _load_fallback(filter_category=category), True

        return articles, False

    except Exception:
        return _load_fallback(filter_category=category), True


def _normalise_newsapi(raw_articles: list[dict]) -> list[dict]:
    """Convert raw NewsAPI response into a clean unified format."""
    normalised = []
    for art in raw_articles:
        title = (art.get("title") or "").strip()
        description = (art.get("description") or "").strip()
        if not title or title == "[Removed]":
            continue
        normalised.append(
            {
                "title": title,
                "description": description or "No description available.",
                "source": (art.get("source") or {}).get("name", "Unknown"),
                "url": art.get("url", ""),
                "publishedAt": art.get("publishedAt", ""),
                "category": "general",
            }
        )
    return normalised


def _load_fallback(filter_category: Optional[str] = None) -> list[dict]:
    """Load sample_news.json as fallback data."""
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Return all categories for 'general' or if filter is None
    if filter_category and filter_category != "general":
        filtered = [a for a in articles if a.get("category") == filter_category]
        return filtered if filtered else articles  # graceful fallback to all

    return articles
