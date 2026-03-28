"""
DataCollectionAgent: HF (CSV с Hub / Datasets Server) + HTTP (JSON или CSV по URL).
Единая схема: text, audio, image, label, source, collected_at.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_ROWS_MAX_BATCH = 100

UNIFIED_COLUMNS = ["text", "audio", "image", "label", "source", "collected_at"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scrape(url: str, selector: str) -> pd.DataFrame:
    raise NotImplementedError("scrape не используется в текущем конфиге.")


def fetch_api(
    endpoint: str,
    params: dict[str, Any] | None = None,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    records_path: str | None = None,
) -> pd.DataFrame:
    method_u = method.upper()
    if method_u == "GET":
        r = requests.get(endpoint, params=params or {}, timeout=120)
    elif method_u == "POST":
        r = requests.post(endpoint, params=params or {}, json=json_body, timeout=120)
    else:
        raise ValueError(f"Unsupported method: {method}")

    r.raise_for_status()
    data = r.json()
    if records_path:
        for key in records_path.split("."):
            data = data[key]
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise ValueError("JSON root must be a list or object")


def _parse_hf_split(split: str) -> tuple[str, int | None]:
    s = split.replace(" ", "")
    m = re.match(r"^(train|test|validation|unsupervised)(?:\[:(\d+)\])?$", s)
    if not m:
        raise ValueError(
            f"Неподдерживаемый split '{split}'; ожидается train[:N], test[:N] и т.п."
        )
    name = m.group(1)
    n = int(m.group(2)) if m.group(2) else None
    return name, n


def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for col in UNIFIED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[UNIFIED_COLUMNS]


def _http_get_rows(url: str, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(12):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(min(5.0 * (attempt + 1), 90.0))
            continue
        r.raise_for_status()
        return r.json()
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def _iter_hf_rows(
    dataset: str,
    config: str,
    split: str,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    offset = 0
    while max_rows is None or len(rows_out) < max_rows:
        need = HF_ROWS_MAX_BATCH
        if max_rows is not None:
            need = min(HF_ROWS_MAX_BATCH, max_rows - len(rows_out))
        if need <= 0:
            break
        payload = _http_get_rows(
            HF_ROWS_URL,
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": need,
            },
        )
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row") or {}
            rows_out.append(row)
            if max_rows is not None and len(rows_out) >= max_rows:
                return rows_out
        offset += len(batch)
        if len(batch) < need:
            break
        time.sleep(0.12)
    return rows_out


def _normalize_hf_row(
    row: pd.Series | dict[str, Any],
    text_col: str,
    label_col: str,
    source_id: str,
    label_map: dict[str, str] | None,
    collected_at: str,
) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.get
    text = get(text_col, pd.NA)
    if pd.notna(text) and not isinstance(text, str):
        text = str(text)

    raw_label = get(label_col, pd.NA)
    if label_map is not None and pd.notna(raw_label):
        key = str(raw_label)
        label = label_map.get(key, label_map.get(raw_label, key))
    else:
        label = raw_label if pd.isna(raw_label) else str(raw_label)

    return {
        "text": text,
        "audio": pd.NA,
        "image": pd.NA,
        "label": label,
        "source": source_id,
        "collected_at": collected_at,
    }


def _normalize_api_frame(
    df: pd.DataFrame,
    text_field: str,
    label_field: str,
    source_id: str,
    collected_at: str,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        t = row.get(text_field, pd.NA)
        if pd.notna(t) and not isinstance(t, str):
            t = str(t)
        lab = row.get(label_field, pd.NA)
        if pd.notna(lab) and not isinstance(lab, str):
            lab = str(lab)
        rows.append(
            {
                "text": t,
                "audio": pd.NA,
                "image": pd.NA,
                "label": lab,
                "source": source_id,
                "collected_at": collected_at,
            }
        )
    return pd.DataFrame(rows)


class DataCollectionAgent:
    def __init__(self, config: str | Path | dict[str, Any]):
        if isinstance(config, (str, Path)):
            path = Path(config)
            with open(path, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.config_path = path
        else:
            self.config = config
            self.config_path = None

    def run(
        self,
        sources: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        src_list = sources if sources is not None else self.config.get("sources", [])
        collected_at = _utc_now_iso()
        frames: list[pd.DataFrame] = []

        for spec in src_list:
            stype = spec.get("type")
            if stype == "hf_dataset":
                frames.append(self._run_hf(spec, collected_at))
            elif stype == "api":
                frames.append(self._run_api(spec, collected_at))
            elif stype == "scrape":
                df_raw = scrape(spec["url"], spec.get("selector", "body"))
                frames.append(
                    self._normalize_generic(df_raw, spec, collected_at)
                )
            else:
                raise ValueError(f"Unknown source type: {stype}")

        merged = merge(frames)
        out_cfg = self.config.get("output", {})
        out_path = out_cfg.get("path", "data/raw/unified_dataset.csv")
        if out_cfg.get("save", True):
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(path, index=False, encoding="utf-8")
        return merged

    def _run_hf(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        if spec.get("loader") == "hub_csv" or spec.get("hf_file"):
            return self._run_hf_hub_csv(spec, collected_at)

        name = spec["name"]
        split = spec.get("split", "train[:1000]")
        hf_config = spec.get("hf_config", spec.get("subset", "default"))
        text_col = spec.get("text_column", "text")
        label_col = spec.get("label_column", "label")
        source_id = spec.get("source_id", f"hf:{name}")
        label_map = spec.get("label_map")

        split_name, max_rows = _parse_hf_split(split)
        raw_rows = _iter_hf_rows(name, hf_config, split_name, max_rows)
        records = [
            _normalize_hf_row(
                r, text_col, label_col, source_id, label_map, collected_at
            )
            for r in raw_rows
        ]
        return pd.DataFrame(records)

    def _run_hf_hub_csv(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        name = spec["name"]
        hf_file = spec["hf_file"]
        max_rows = int(spec.get("max_rows", 1000))
        text_col = spec.get("text_column", "text")
        label_col = spec.get("label_column", "label")
        source_id = spec.get("source_id", f"hf:{name}")
        label_map = spec.get("label_map")

        url = f"https://huggingface.co/datasets/{name}/resolve/main/{hf_file}"
        df = pd.read_csv(url, nrows=max_rows)
        records = [
            _normalize_hf_row(
                row.to_dict(),
                text_col,
                label_col,
                source_id,
                label_map,
                collected_at,
            )
            for _, row in df.iterrows()
        ]
        return pd.DataFrame(records)

    def _run_api(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        kind = spec.get("api_kind", "json_flat")
        if kind == "hf_datasets_rows":
            return self._run_api_hf_rows(spec, collected_at)
        if kind == "csv_http":
            return self._run_api_csv_http(spec, collected_at)
        if kind == "csv_http_concat":
            return self._run_api_csv_concat(spec, collected_at)

        endpoint = spec["endpoint"]
        params = spec.get("params")
        method = spec.get("method", "GET")
        json_body = spec.get("json_body")
        records_path = spec.get("records_path")
        text_field = spec.get("text_field", "text")
        label_field = spec.get("label_field", "label")
        source_id = spec.get("source_id", "api:unknown")

        raw = fetch_api(
            endpoint,
            params=params,
            method=method,
            json_body=json_body,
            records_path=records_path,
        )
        return _normalize_api_frame(
            raw, text_field, label_field, source_id, collected_at
        )

    def _run_api_hf_rows(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        ds = spec["hf_dataset"]
        hf_config = spec.get("hf_config", "default")
        split_name = spec.get("split", "test")
        max_rows = int(spec.get("max_rows", 1000))
        text_col = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", f"api:hf-rows:{ds}")
        label_map = spec.get("label_map")
        label_allow = spec.get("label_allow")

        if label_allow is not None:
            allow = set(map(str, label_allow))
            raw_rows = _iter_hf_rows(ds, hf_config, split_name, max_rows * 8)
            kept: list[dict[str, Any]] = []
            for r in raw_rows:
                lab = r.get(label_col)
                if str(lab) not in allow:
                    continue
                kept.append(r)
                if len(kept) >= max_rows:
                    break
        else:
            kept = _iter_hf_rows(ds, hf_config, split_name, max_rows)

        records = [
            _normalize_hf_row(
                r, text_col, label_col, source_id, label_map, collected_at
            )
            for r in kept
        ]
        return pd.DataFrame(records)

    def _run_api_csv_http(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        url = spec["endpoint"]
        max_rows = int(spec.get("max_rows", 1000))
        text_col = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", "api:csv")
        label_map = spec.get("label_map")

        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), nrows=max_rows)
        records = [
            _normalize_hf_row(
                row.to_dict(),
                text_col,
                label_col,
                source_id,
                label_map,
                collected_at,
            )
            for _, row in df.iterrows()
        ]
        return pd.DataFrame(records)

    def _run_api_csv_concat(self, spec: dict[str, Any], collected_at: str) -> pd.DataFrame:
        urls: list[str] = spec["endpoints"]
        max_rows = int(spec.get("max_rows", 1000))
        text_col = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", "api:csv-concat")
        label_map = spec.get("label_map")

        parts: list[pd.DataFrame] = []
        remaining = max_rows
        for url in urls:
            if remaining <= 0:
                break
            part = pd.read_csv(url, nrows=remaining)
            parts.append(part)
            remaining -= len(part)

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        records = [
            _normalize_hf_row(
                row.to_dict(),
                text_col,
                label_col,
                source_id,
                label_map,
                collected_at,
            )
            for _, row in df.iterrows()
        ]
        return pd.DataFrame(records)

    def _normalize_generic(
        self, df: pd.DataFrame, spec: dict[str, Any], collected_at: str
    ) -> pd.DataFrame:
        text_col = spec.get("text_column", "text")
        label_col = spec.get("label_column", "label")
        source_id = spec.get("source_id", "unknown")
        return _normalize_api_frame(df, text_col, label_col, source_id, collected_at)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Сбор данных (DataCollectionAgent)")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    agent = DataCollectionAgent(args.config)
    df = agent.run()
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
