# ML Data Project — сбор и качество (RU, бинарный сентимент)

## Задача

**Бинарная классификация тональности** русскоязычного текста: метки `neg` / `pos`.

## Схема данных

Колонки: `text`, `audio`, `image`, `label`, `source`, `collected_at` → **`data/raw/unified_dataset.csv`** (UTF-8).

## Этап 1 — сбор

Все строки из [`sepidmnorozy/Russian_sentiment`](https://huggingface.co/datasets/sepidmnorozy/Russian_sentiment), два способа загрузки (HF + HTTP):

| Часть | Тип | Как |
|-------|-----|-----|
| **1000** | `hf_dataset` + `hub_csv` | `train.csv`, первые 1000 строк |
| **1000** | `api` + `csv_http_concat` | `dev.csv` + `test.csv` до 1000 строк |

```bash
pip install -r requirements.txt
python collect.py --config config.yaml
```

## Этап 2 — качество (`DataQualityAgent`)

Модуль: `agents/data_quality_agent.py`. Публичный API: `detect_issues(df)`, `fix(df, strategy)`, `compare(df_before, df_after)`.

**Детект (≥3 типа проблем):** пропуски по колонкам, дубликаты строк (по подмножеству колонок), выбросы по **длине текста** (IQR, множитель **1.5**), дисбаланс классов по колонке **`label`** (счётчики, энтропия, отношение majority/minority).

**Стратегия `strategy`** (ключи в `strategy.yaml` под блоком `strategy:`):

| Ключ | Допустимые значения | Смысл |
|------|---------------------|--------|
| `missing` | `drop_rows`, `fill_modal_empty`, `skip` | Пропуски: удалить строки с NaN в важных полях / заполнить `audio`/`image` пустой строкой / не трогать |
| `duplicates` | `drop`, `none` | Дубликаты по **`text` + `label`**, при `drop` оставляется **первая** строка |
| `outliers` | `clip_iqr_text_length`, `none` | Удалить строки, где длина `text` вне \[Q1−1.5·IQR, Q3+1.5·IQR\] |
| `imbalance` | `undersample_majority`, `none` | Случайное уравнивание классов до минимального класса (`random_state=42`) или не трогать |

Числовые правила применяются к производной **длине текста**; модальности `audio`/`image` при `fill_modal_empty` заполняются строкой `""`.

**Запуск CLI:**

```bash
python quality.py --input data/raw/unified_dataset.csv --strategy strategy.yaml --output data/processed/cleaned.csv --report-json data/processed/quality_report.json --comparison-csv data/processed/quality_comparison.csv
```

Артефакты: **`data/processed/cleaned.csv`**, **`data/processed/quality_report.json`**, **`data/processed/quality_comparison.csv`**.

**Ноутбук:** `notebooks/data_quality.ipynb` — детект, визуализации, два прогона (`balanced` и `conservative`), сводная таблица `compare`.

**Интерактивный пайплайн:** после просмотра отчёта можно сменить стратегию в `strategy.yaml` и перезапустить `quality.py`. Опционально: `explain_and_recommend(report, ...)` — при наличии **`ANTHROPIC_API_KEY`** можно подключить Claude (пакет `anthropic` не входит в зависимости по умолчанию).

## Этап 3 — разметка (`AnnotationAgent`)

Модуль: `agents/annotation_agent.py`. Публичный API: `auto_label(df)`, `generate_spec(df, task=...)`, `check_quality(df_labeled)`, `export_to_labelstudio(df)`, опционально `export_low_confidence(...)`.

**Модальность `text`:** демо **авторазметка** — `TfidfVectorizer` + `LogisticRegression` обучаются на колонке эталона **`label`** (как прокси обучающих меток) и добавляют:

- **`label_auto`** — предсказанный класс (`neg` / `pos`);
- **`confidence`** — max вероятность по классам.

В продакшене замените `auto_label` на инференс выбранной модели; контракт колонок сохраняется.

**Артефакты по умолчанию:**

| Файл | Назначение |
|------|------------|
| `data/labeled/labeled.csv` | Исходные поля + `label_auto`, `confidence` |
| `annotation_spec.md` | Инструкция для разметчиков (классы, ≥3 примера на класс, граничные случаи) |
| `data/labeled/labelstudio_import.json` | Импорт в [Label Studio](https://labelstud.io/guide/import.html) (массив задач с `data.text`, опционально `predictions`) |
| `data/labeled/annotation_metrics.json` | κ (Cohen между `label` и `label_auto`), `agreement`, `label_dist`, `confidence_mean` |
| `review_queue.csv` | Строки с `confidence` ниже порога (`annotation_config.yaml` → `confidence_threshold`) |

**Label Studio:** создайте проект **Text classification**, поле ввода с именем **`text`**, control **Choices** с `from_name` = **`sentiment`**, `to_name` = **`text`**, варианты `neg` и `pos` — в соответствии с полем `predictions` в JSON. Проверено на совместимости с форматом импорта задач LS **1.11.x** (при смене мажорной версии сверьте [документацию](https://labelstud.io/guide/import.html)).

**Запуск:**

```bash
python annotate.py --input data/processed/cleaned.csv --config annotation_config.yaml
```

**HITL:** после ручной правки меток объедините колонку (например `label_human`) с кадром и снова вызовите `check_quality` в Python.

**Ноутбук:** `notebooks/annotation_eval.ipynb`.

## Этап 4 — Active Learning (`ActiveLearningAgent`)

Модуль: `agents/al_agent.py`. Методы: `fit`, `query` (стратегии **`entropy`**, **`margin`**, **`random`**), `evaluate`, `run_cycle`, `report`.

**Симуляция:** фиксированный **тестовый** набор (20% строк, стратификация по `label` при достаточном числе примеров в классах); из оставшего **train** берётся **seed** (стартовая разметка) и **пул**. На каждой итерации модель (TF-IDF + логистическая регрессия, как на этапе 3) дообучается на текущем размеченном множестве; из пула выбирается батч по стратегии; **истинные метки** для отобранных строк берутся из колонки **`label`** (оракул). Метрики на тесте: **accuracy** и **F1 macro**.

**Сравнение:** по умолчанию два прогона `run_cycle` — основная стратегия (`al_config.yaml` → `primary_strategy`, обычно **entropy**) и baseline **`random`**; строится один график **`reports/al_learning_curve.png`** (ось X — число размеченных, ось Y — F1 macro), история — **`reports/al_history.json`**.

**Запуск:**

```bash
python al_run.py --input data/labeled/labeled.csv --config al_config.yaml
```

Параметры цикла: `seed_size`, `n_iterations`, `batch_size`, `test_size` — в `al_config.yaml`.

**Примечание:** для стратегий на основе вероятностей нужен `predict_proba`; при одном классе в пуле возможны вырожденные случаи — см. логи/историю.

## Зависимости

`pandas`, `pyyaml`, `requests`, `matplotlib`, `numpy`, `scikit-learn`.
