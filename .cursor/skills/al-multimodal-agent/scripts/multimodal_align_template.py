"""
Шаблон выравнивания нескольких модальностей. Перенеси в MultimodalAgent.align().

TODO: типы файлов (parquet/csv), ключ sample_id vs timestamp, outer join и учёт unmatched.
"""
from __future__ import annotations

# Псевдокод:
#
# text_df = pd.read_parquet("data/text.parquet")
# img_df = pd.read_csv("data/images_manifest.csv")
# aligned = text_df.merge(img_df, on="sample_id", how="outer", indicator=True)
# unmatched = aligned[aligned["_merge"] != "both"]


def main() -> None:
    print("Скопируй merge-логику в agents/multimodal_agent.py; добавь describe() для unmatched.")


if __name__ == "__main__":
    main()