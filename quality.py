#!/usr/bin/env python3
"""CLI: очистка данных. Пример: python quality.py --input data/raw/unified_dataset.csv --strategy strategy.yaml"""

from agents.data_quality_agent import main

if __name__ == "__main__":
    main()
