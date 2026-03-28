"""
ActiveLearningAgent: fit → query (entropy | margin | random) → симуляция меток из пула → evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

QueryStrategy = Literal["entropy", "margin", "random"]

History = list[dict[str, Any]]


class ActiveLearningAgent:
    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        random_state: int = 42,
        max_features: int = 20_000,
    ):
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None
        self._query_nonce = 0

    def fit(self, labeled_df: pd.DataFrame) -> None:
        if len(labeled_df) == 0:
            raise ValueError("labeled_df пуст.")
        texts = labeled_df[self.text_column].fillna("").astype(str)
        y = labeled_df[self.label_column].astype(str)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        X = self._vectorizer.fit_transform(texts)
        self._clf = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._clf.fit(X, y)

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: QueryStrategy,
        batch_size: int,
    ) -> np.ndarray:
        """Возвращает позиции iloc в pool_df для доразметки."""
        if self._vectorizer is None or self._clf is None:
            raise RuntimeError("Сначала вызовите fit().")
        n = len(pool_df)
        if n == 0 or batch_size <= 0:
            return np.array([], dtype=int)
        batch_size = min(batch_size, n)

        texts = pool_df[self.text_column].fillna("").astype(str)
        Xp = self._vectorizer.transform(texts)
        if strategy == "random":
            self._query_nonce += 1
            rng = np.random.default_rng(self.random_state + self._query_nonce * 10_007)
            idx = rng.choice(n, size=batch_size, replace=False)
            return np.sort(idx)

        proba = self._clf.predict_proba(Xp)
        if strategy == "entropy":
            p = np.clip(proba, 1e-12, 1.0)
            scores = -(p * np.log(p)).sum(axis=1)
            order = np.argsort(-scores)
        elif strategy == "margin":
            if proba.shape[1] < 2:
                order = np.arange(n)
            else:
                ps = np.sort(proba, axis=1)
                scores = ps[:, -1] - ps[:, -2]
                order = np.argsort(scores)
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

        return np.sort(order[:batch_size])

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, float]:
        if self._vectorizer is None or self._clf is None:
            raise RuntimeError("Сначала вызовите fit().")
        if len(test_df) == 0:
            return {"accuracy": float("nan"), "f1_macro": float("nan")}
        Xt = self._vectorizer.transform(
            test_df[self.text_column].fillna("").astype(str)
        )
        y_true = test_df[self.label_column].astype(str)
        y_pred = self._clf.predict(Xt)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: QueryStrategy,
        n_iterations: int,
        batch_size: int,
    ) -> History:
        """
        Симуляция: метки для отобранных примеров берутся из pool_df[label_column] (оракул).
        """
        self._query_nonce = 0
        labeled = labeled_df.copy().reset_index(drop=True)
        pool = pool_df.copy().reset_index(drop=True)
        history: History = []

        rng = np.random.default_rng(self.random_state)

        def snapshot(it: int, phase: str) -> None:
            self.fit(labeled)
            metrics = self.evaluate(test_df)
            history.append(
                {
                    "iteration": it,
                    "phase": phase,
                    "n_labeled": len(labeled),
                    "strategy": strategy,
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                }
            )

        snapshot(0, "start")

        for it in range(1, n_iterations + 1):
            if len(pool) == 0:
                break
            idx = self.query(pool, strategy, batch_size)
            picked = pool.iloc[idx].copy()
            labeled = pd.concat([labeled, picked], ignore_index=True)
            mask = np.ones(len(pool), dtype=bool)
            mask[idx] = False
            pool = pool.iloc[mask].reset_index(drop=True)
            snapshot(it, "after_query")

        return history

    def report(
        self,
        histories: dict[str, History],
        output_path: str | Path = "reports/al_learning_curve.png",
        title: str = "Active Learning (симуляция меток из пула)",
    ) -> Path:
        import matplotlib.pyplot as plt

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        for name, hist in histories.items():
            xs = [h["n_labeled"] for h in hist]
            ys = [h["f1_macro"] for h in hist]
            plt.plot(xs, ys, marker="o", label=name)

        plt.xlabel("Число размеченных примеров")
        plt.ylabel("F1 (macro) на тесте")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return path


def train_test_pool_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed_size: int = 128,
    label_column: str = "label",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Тест (фиксированный), стартовая разметка, пул."""
    from sklearn.model_selection import train_test_split

    strat = None
    if label_column in df.columns:
        vc = df[label_column].astype(str).value_counts()
        if vc.min() >= 2:
            strat = df[label_column].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    train_df = train_df.reset_index(drop=True)
    if len(train_df) <= seed_size:
        seed = train_df.iloc[: len(train_df) // 2].copy()
        pool = train_df.iloc[len(train_df) // 2 :].copy()
    else:
        seed = train_df.iloc[:seed_size].copy()
        pool = train_df.iloc[seed_size:].copy()
    return seed, pool.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Этап 4: Active Learning")
    p.add_argument("--input", default="data/labeled/labeled.csv")
    p.add_argument("--config", default="al_config.yaml")
    p.add_argument("--out-json", default="reports/al_history.json")
    p.add_argument("--out-plot", default="reports/al_learning_curve.png")
    args = p.parse_args()

    cfg: dict[str, Any] = {}
    if Path(args.config).is_file():
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    df = pd.read_csv(args.input, encoding="utf-8")
    text_col = cfg.get("text_column", "text")
    label_col = cfg.get("label_column", "label")
    rs = int(cfg.get("random_state", 42))
    test_size = float(cfg.get("test_size", 0.2))
    seed_size = int(cfg.get("seed_size", 128))
    n_iter = int(cfg.get("n_iterations", 6))
    batch = int(cfg.get("batch_size", 48))

    seed_df, pool_df, test_df = train_test_pool_split(
        df,
        test_size=test_size,
        seed_size=seed_size,
        label_column=label_col,
        random_state=rs,
    )

    agent = ActiveLearningAgent(
        text_column=text_col,
        label_column=label_col,
        random_state=rs,
    )

    primary = str(cfg.get("primary_strategy", "entropy"))
    baseline = str(cfg.get("baseline_strategy", "random"))

    hist_primary = agent.run_cycle(
        seed_df.copy(),
        pool_df.copy(),
        test_df,
        strategy=primary,  # type: ignore[arg-type]
        n_iterations=n_iter,
        batch_size=batch,
    )
    hist_random = agent.run_cycle(
        seed_df.copy(),
        pool_df.copy(),
        test_df,
        strategy="random",
        n_iterations=n_iter,
        batch_size=batch,
    )

    out: dict[str, Any] = {
        "primary_strategy": primary,
        "baseline_strategy": baseline,
        "primary_history": hist_primary,
        "random_history": hist_random,
        "meta": {
            "seed_size": len(seed_df),
            "pool_initial": len(pool_df),
            "test_size": len(test_df),
            "n_iterations": n_iter,
            "batch_size": batch,
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    plot_path = agent.report(
        {f"AL ({primary})": hist_primary, "random": hist_random},
        output_path=args.out_plot,
    )
    print(f"OK: {args.out_json} plot={plot_path}")


if __name__ == "__main__":
    main()
