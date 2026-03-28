"""
DataQualityAgent: detect_issues, fix, compare; опционально explain_and_recommend (LLM).
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

QualityReport = dict[str, Any]
ComparisonReport = pd.DataFrame


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


class DataQualityAgent:
    def __init__(
        self,
        label_column: str = "label",
        text_column: str = "text",
        duplicate_subset: list[str] | None = None,
        outlier_length_column: str = "text",
    ):
        self.label_column = label_column
        self.text_column = text_column
        self.duplicate_subset = duplicate_subset or [text_column, label_column]
        self.outlier_length_column = outlier_length_column

    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        if df is None or len(df) == 0:
            return {
                "n_rows": 0,
                "missing": {"by_column": {}, "total_missing_cells": 0},
                "duplicates": {"n_duplicate_rows": 0, "subset": self.duplicate_subset},
                "outliers": {
                    "text_length_iqr": {"count": 0, "low": None, "high": None},
                },
                "imbalance": {
                    "label_column": self.label_column,
                    "counts": {},
                    "entropy": None,
                    "majority_minority_ratio": None,
                },
            }

        n = len(df)
        missing_by: dict[str, dict[str, Any]] = {}
        total_missing = 0
        for col in df.columns:
            miss = df[col].isna().sum()
            total_missing += int(miss)
            missing_by[col] = {"count": int(miss), "ratio": float(miss / n) if n else 0.0}

        sub = [c for c in self.duplicate_subset if c in df.columns]
        dup_mask = df.duplicated(subset=sub, keep=False) if sub else pd.Series(False, index=df.index)
        n_dup = int(dup_mask.sum())

        tl = df[self.outlier_length_column].astype(str).str.len() if self.outlier_length_column in df.columns else pd.Series(dtype=float)
        out_count = 0
        lo = hi = None
        if len(tl) > 0 and tl.notna().any():
            lo, hi = _iqr_bounds(tl.astype(float))
            out_count = int(((tl < lo) | (tl > hi)).sum())

        imb: dict[str, Any] = {"label_column": self.label_column}
        if self.label_column in df.columns:
            vc = df[self.label_column].astype(str).value_counts()
            counts = vc.to_dict()
            vals = list(counts.values())
            maj = max(vals) if vals else 0
            minr = min(vals) if vals else 0
            imb["counts"] = counts
            imb["entropy"] = _entropy(vals)
            imb["majority_minority_ratio"] = float(maj / minr) if minr else None
        else:
            imb["counts"] = {}
            imb["entropy"] = None
            imb["majority_minority_ratio"] = None
            imb["note"] = f"Колонка '{self.label_column}' не найдена."

        return {
            "n_rows": n,
            "missing": {"by_column": missing_by, "total_missing_cells": total_missing},
            "duplicates": {
                "n_duplicate_rows": n_dup,
                "subset": sub,
            },
            "outliers": {
                "text_length_iqr": {
                    "column": self.outlier_length_column,
                    "count": out_count,
                    "low": lo,
                    "high": hi,
                    "k_iqr": 1.5,
                },
            },
            "imbalance": imb,
        }

    def fix(self, df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df.copy() if df is not None else pd.DataFrame()

        out = df.copy()
        st = {k: v for k, v in strategy.items()}

        miss = st.get("missing", "skip")
        if miss == "drop_rows":
            out = out.dropna(subset=[c for c in [self.text_column, self.label_column] if c in out.columns], how="any")
        elif miss == "fill_modal_empty":
            for col in ("audio", "image"):
                if col in out.columns:
                    out[col] = out[col].where(out[col].notna(), "")

        dup = st.get("duplicates", "none")
        if dup == "drop":
            sub = [c for c in self.duplicate_subset if c in out.columns]
            if sub:
                out = out.drop_duplicates(subset=sub, keep="first")

        outl = st.get("outliers", "none")
        if outl == "clip_iqr_text_length" and self.outlier_length_column in out.columns:
            tl = out[self.outlier_length_column].astype(str).str.len().astype(float)
            lo, hi = _iqr_bounds(tl, k=float(st.get("iqr_k", 1.5)))
            ok = (tl >= lo) & (tl <= hi)
            out = out.loc[ok].reset_index(drop=True)

        imb = st.get("imbalance", "none")
        if imb == "undersample_majority" and self.label_column in out.columns:
            vc = out[self.label_column].astype(str).value_counts()
            if len(vc) >= 2:
                m = int(vc.min())
                parts = []
                for lab in vc.index:
                    part = out[out[self.label_column].astype(str) == lab]
                    parts.append(part.sample(n=min(len(part), m), random_state=42))
                out = pd.concat(parts, ignore_index=True)

        return out

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
        r0 = self.detect_issues(df_before)
        r1 = self.detect_issues(df_after)

        def row(metric: str, before: Any, after: Any) -> dict[str, Any]:
            b = before if not isinstance(before, (dict, list)) else json.dumps(before, ensure_ascii=False)
            a = after if not isinstance(after, (dict, list)) else json.dumps(after, ensure_ascii=False)
            delta = ""
            try:
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    delta = after - before
            except TypeError:
                delta = ""
            return {"metric": metric, "before": b, "after": a, "delta": delta}

        rows = [
            row("n_rows", r0["n_rows"], r1["n_rows"]),
            row("total_missing_cells", r0["missing"]["total_missing_cells"], r1["missing"]["total_missing_cells"]),
            row("n_duplicate_rows", r0["duplicates"]["n_duplicate_rows"], r1["duplicates"]["n_duplicate_rows"]),
            row(
                "outliers_text_length_iqr",
                r0["outliers"]["text_length_iqr"]["count"],
                r1["outliers"]["text_length_iqr"]["count"],
            ),
            row(
                "class_entropy",
                r0["imbalance"].get("entropy"),
                r1["imbalance"].get("entropy"),
            ),
            row(
                "majority_minority_ratio",
                r0["imbalance"].get("majority_minority_ratio"),
                r1["imbalance"].get("majority_minority_ratio"),
            ),
        ]
        return pd.DataFrame(rows)

    def explain_and_recommend(
        self,
        report: QualityReport,
        task_description: str = "Классификация текста",
    ) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return (
                "LLM выключен: задайте ANTHROPIC_API_KEY для вызова Claude. "
                "Кратко по отчёту: проверьте пропуски в text/label, дубликаты по (text,label), "
                "длину текстов (IQR) и баланс классов; затем выберите strategy в strategy.yaml."
            )
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            summary = {
                "n_rows": report.get("n_rows"),
                "missing_total": report.get("missing", {}).get("total_missing_cells"),
                "duplicates": report.get("duplicates", {}).get("n_duplicate_rows"),
                "outliers_len": report.get("outliers", {}).get("text_length_iqr", {}).get("count"),
                "imbalance": report.get("imbalance", {}),
            }
            msg = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Задача ML: {task_description}\n\n"
                            f"Краткий отчёт о качестве (JSON): {json.dumps(summary, ensure_ascii=False)}\n\n"
                            "Дай рекомендации по strategy с ключами: missing, duplicates, outliers, imbalance. "
                            "Допустимые значения: missing — drop_rows | fill_modal_empty | skip; "
                            "duplicates — drop | none; outliers — clip_iqr_text_length | none; "
                            "imbalance — undersample_majority | none."
                        ),
                    }
                ],
            )
            block = msg.content[0]
            return block.text if hasattr(block, "text") else str(block)
        except Exception as e:
            return f"LLM недоступен: {e}"


def load_strategy(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("strategy", data)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Очистка данных (DataQualityAgent)")
    p.add_argument("--input", required=True, help="Входной CSV")
    p.add_argument("--strategy", default="strategy.yaml", help="YAML со strategy")
    p.add_argument("--output", default="data/processed/cleaned.csv")
    p.add_argument("--report-json", default="data/processed/quality_report.json")
    p.add_argument("--comparison-csv", default="data/processed/quality_comparison.csv")
    p.add_argument("--label-column", default="label")
    args = p.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    agent = DataQualityAgent(label_column=args.label_column)
    report = agent.detect_issues(df)
    strategy = load_strategy(args.strategy)
    cleaned = agent.fix(df, strategy)
    comp = agent.compare(df, cleaned)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_p, index=False, encoding="utf-8")

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    comp.to_csv(args.comparison_csv, index=False, encoding="utf-8")
    print(f"OK: {len(df)} -> {len(cleaned)} rows; {out_p}")


if __name__ == "__main__":
    main()
