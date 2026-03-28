#!/usr/bin/env python3
"""
Шаблон скрейпинга. Скопируй в проект или адаптируй внутри agents/data_collection_agent.scrape().

TODO: URL, CSS-селектор, парсинг текста, обработка ошибок HTTP, rate limiting / robots.txt.
"""
from __future__ import annotations

import sys

import requests
from bs4 import BeautifulSoup

DEFAULT_URL = "https://example.com"
DEFAULT_SELECTOR = "p"  # пример: абзацы на example.com


def scrape_to_records(url: str, selector: str) -> list[dict]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    elements = soup.select(selector)
    return [{"raw_html": str(el), "text": el.get_text(strip=True)} for el in elements]


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    sel = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SELECTOR
    rows = scrape_to_records(url, sel)
    print(f"rows={len(rows)}")
    for i, r in enumerate(rows[:5]):
        print(f"[{i}] {r['text'][:120]!r}...")


if __name__ == "__main__":
    main()