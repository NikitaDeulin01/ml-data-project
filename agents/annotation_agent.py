"""
AnnotationAgent: auto_label (text), generate_spec, check_quality, export_to_labelstudio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

QualityMetrics = dict[str, Any]


class AnnotationAgent:
    """Текстовая модальность: линейная модель на TF-IDF (демо без тяжёлых весов)."""

    def __init__(
        self,
        modality: str = "text",
        text_column: str = "text",
        gold_label_column: str = "label",
        auto_label_column: str = "label_auto",
        confidence_column: str = "confidence",
        random_state: int = 42,
    ):
        self.modality = modality
        self.text_column = text_column
        self.gold_label_column = gold_label_column
        self.auto_label_column = auto_label_column
        self.confidence_column = confidence_column
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.modality != "text":
            raise NotImplementedError(f"Модальность {self.modality!r} не реализована.")
        if self.text_column not in df.columns:
            raise ValueError(f"Нет колонки {self.text_column!r}")
        if self.gold_label_column not in df.columns:
            raise ValueError(
                f"Нет колонки {self.gold_label_column!r} (нужна для обучения демо-модели)."
            )

        out = df.copy()
        texts = out[self.text_column].fillna("").astype(str)
        y = out[self.gold_label_column].astype(str)

        self._vectorizer = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            min_df=2,
        )
        X = self._vectorizer.fit_transform(texts)
        self._clf = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._clf.fit(X, y)

        proba = self._clf.predict_proba(X)
        classes = list(self._clf.classes_)
        pred_idx = np.argmax(proba, axis=1)
        out[self.auto_label_column] = [classes[i] for i in pred_idx]
        out[self.confidence_column] = proba.max(axis=1).astype(float)
        return out

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
        output_path: str | Path = "annotation_spec.md",
    ) -> Path:
        path = Path(output_path)
        labels = sorted(df[self.gold_label_column].astype(str).unique()) if self.gold_label_column in df.columns else []

        def examples_for(lab: str, k: int = 3) -> list[str]:
            if self.gold_label_column not in df.columns:
                return []
            sub = df[df[self.gold_label_column].astype(str) == lab]
            if len(sub) == 0:
                return []
            sample = sub.head(k)
            return [str(t)[:400].replace("\n", " ") for t in sample[self.text_column].tolist()]

        lines = [
            "# Спецификация разметки: тональность текста (RU)\n",
            "## Задача ML\n",
            f"**{task}**: бинарная классификация тональности русскоязычного текста для последующего обучения модели.\n",
            "## Классы\n",
        ]
        for lab in labels:
            lines.append(f"- **`{lab}`**: целевая тональность фрагмента (согласовано с эталонным датасетом).\n")

        lines.append("\n## Примеры по классам (≥3 на класс, фрагменты)\n")
        for lab in labels:
            lines.append(f"### `{lab}`\n")
            for i, ex in enumerate(examples_for(lab, 3), 1):
                lines.append(f"{i}. {ex}\n")

        lines.append("\n## Граничные случаи\n")
        lines.append(
            "- Смешанная тональность: выбрать **доминирующую** по смыслу фрагмента; при равенстве — класс с меньшей уверенностью модели или эскалация ревьюеру.\n"
        )
        lines.append(
            "- Нерелевантный/пустой текст: пометить как ошибку данных; в пайплайне такие строки должны отсекаться на этапе качества.\n"
        )
        lines.append(
            "- Несовпадение авто-метки и эталона: смотреть колонку `confidence`; при низкой уверенности — очередь `review_queue.csv`.\n"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def check_quality(self, df_labeled: pd.DataFrame) -> QualityMetrics:
        gold = self.gold_label_column
        auto = self.auto_label_column
        conf = self.confidence_column

        dist = (
            df_labeled[auto].astype(str).value_counts().to_dict()
            if auto in df_labeled.columns
            else {}
        )
        mean_conf = float(df_labeled[conf].mean()) if conf in df_labeled.columns else None

        kappa = None
        agreement = None
        note = None
        if gold in df_labeled.columns and auto in df_labeled.columns:
            m = df_labeled[gold].notna() & df_labeled[auto].notna()
            if m.sum() > 0:
                kappa = float(
                    cohen_kappa_score(
                        df_labeled.loc[m, gold].astype(str),
                        df_labeled.loc[m, auto].astype(str),
                    )
                )
                agreement = float(
                    (
                        df_labeled.loc[m, gold].astype(str)
                        == df_labeled.loc[m, auto].astype(str)
                    ).mean()
                )
        else:
            note = "Нет колонок для сравнения эталона и авторазметки."

        return {
            "kappa": kappa,
            "agreement": agreement,
            "label_dist": dist,
            "confidence_mean": mean_conf,
            "note": note,
        }

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        output_path: str | Path = "labelstudio_import.json",
        text_key: str = "text",
    ) -> Path:
        """JSON-массив задач для импорта (Label Studio 1.x, Text classification)."""
        path = Path(output_path)
        tasks: list[dict[str, Any]] = []
        for i, row in df.iterrows():
            text = str(row.get(self.text_column, ""))
            rid = int(i) if isinstance(i, (int, np.integer)) else str(i)
            item: dict[str, Any] = {
                "data": {
                    text_key: text,
                    "row_id": rid,
                    "ref_label": str(row[self.gold_label_column])
                    if self.gold_label_column in row
                    else "",
                },
            }
            if self.auto_label_column in row and pd.notna(row[self.auto_label_column]):
                lab = str(row[self.auto_label_column])
                item["predictions"] = [
                    {
                        "model_version": "sklearn_tfidf_lr",
                        "score": float(row[self.confidence_column])
                        if self.confidence_column in row and pd.notna(row[self.confidence_column])
                        else None,
                        "result": [
                            {
                                "from_name": "sentiment",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [lab]},
                            }
                        ],
                    }
                ]
            tasks.append(item)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def export_low_confidence(
        self,
        df: pd.DataFrame,
        threshold: float = 0.55,
        output_path: str | Path = "review_queue.csv",
    ) -> Path | None:
        if self.confidence_column not in df.columns:
            return None
        sub = df[df[self.confidence_column] < threshold].copy()
        if len(sub) == 0:
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(path, index=False, encoding="utf-8")
        return path


def load_annotate_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Этап 3: авторазметка и экспорт")
    p.add_argument("--input", default="data/processed/cleaned.csv")
    p.add_argument("--config", default="annotation_config.yaml")
    p.add_argument("--output-labeled", default="data/labeled/labeled.csv")
    p.add_argument("--spec", default="annotation_spec.md")
    p.add_argument("--ls-json", default="data/labeled/labelstudio_import.json")
    p.add_argument("--metrics-json", default="data/labeled/annotation_metrics.json")
    p.add_argument("--review-threshold", type=float, default=0.55)
    args = p.parse_args()

    cfg = {}
    if Path(args.config).is_file():
        cfg = load_annotate_config(args.config)

    df = pd.read_csv(args.input, encoding="utf-8")
    agent = AnnotationAgent(
        modality=cfg.get("modality", "text"),
        text_column=cfg.get("text_column", "text"),
        gold_label_column=cfg.get("gold_label_column", "label"),
    )
    labeled = agent.auto_label(df)
    spec_path = agent.generate_spec(
        labeled,
        task=cfg.get("task", "sentiment_classification"),
        output_path=args.spec,
    )
    ls_path = agent.export_to_labelstudio(labeled, output_path=args.ls_json)
    metrics = agent.check_quality(labeled)

    out_p = Path(args.output_labeled)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_p, index=False, encoding="utf-8")

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    rq = agent.export_low_confidence(
        labeled,
        threshold=float(cfg.get("confidence_threshold", args.review_threshold)),
        output_path=cfg.get("review_queue_path", "review_queue.csv"),
    )

    print(f"OK: labeled={out_p} rows={len(labeled)}")
    print(f"spec={spec_path} ls={ls_path} metrics={args.metrics_json}")
    if rq:
        print(f"review_queue={rq}")


if __name__ == "__main__":
    main()
