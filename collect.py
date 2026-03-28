#!/usr/bin/env python3
"""CLI: python collect.py --config config.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.data_collection_agent import DataCollectionAgent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сбор данных: HF + API -> data/raw/unified_dataset.csv"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    agent = DataCollectionAgent(cfg_path)
    df = agent.run()
    out = agent.config.get("output", {}).get("path", "data/raw/unified_dataset.csv")
    print(f"OK: {len(df)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
