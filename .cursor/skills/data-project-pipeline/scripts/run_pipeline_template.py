#!/usr/bin/env python3
"""
Шаблон единого пайплайна: сбор → качество → разметка → HITL → AL → обучение → отчёты.

Скопируй в корень проекта как run_pipeline.py (или run.py).
Замени импорты и вызовы на реальные API агентов.

HITL: пайплайн записывает review_queue.csv и ожидает review_queue_corrected.csv
       (или используй флаг --skip-hitl только для CI/demo — задокументируй в README).
"""
from __future__ import annotations

# from agents.data_collection_agent import DataCollectionAgent
# from agents.data_quality_agent import DataQualityAgent
# from agents.annotation_agent import AnnotationAgent
# from agents.al_agent import ActiveLearningAgent


def main() -> None:
    # raw = DataCollectionAgent(config="config.yaml").run(...)
    # report = DataQualityAgent().detect_issues(raw)
    # clean = DataQualityAgent().fix(raw, strategy={...})
    # labeled = AnnotationAgent(modality="text").auto_label(clean)
    # low = labeled[labeled["confidence"] < 0.7]
    # low.to_csv("review_queue.csv", index=False)
    # raise SystemExit("HITL: исправьте метки, сохраните как review_queue_corrected.csv и перезапустите с --resume")
    # ...
    # ActiveLearningAgent(...).run_cycle(...)
    # train / save model / reports
    print("TODO: реализовать вызовы агентов и сохранение артефактов (см. skill data-project-pipeline).")


if __name__ == "__main__":
    main()