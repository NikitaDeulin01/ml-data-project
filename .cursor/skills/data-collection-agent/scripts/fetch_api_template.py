#!/usr/bin/env python3
"""
Шаблон загрузки JSON API → список плоских записей.

TODO: endpoint, params, путь к массиву в JSON (data/items/results), пагинация.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import requests


def flatten_records(payload: Any, records_key: str | None = None) -> list[dict]:
    if records_key and isinstance(payload, dict):
        payload = payload[records_key]
    if isinstance(payload, list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload]
    if isinstance(payload, dict):
        return [payload]
    return []


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://jsonplaceholder.typicode.com/posts?_limit=5"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = flatten_records(data, records_key=None)
    print(json.dumps(rows[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()