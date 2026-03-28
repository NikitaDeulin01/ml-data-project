"""
Шаблон цикла Active Learning. Не рабочий агент — перенеси логику в ActiveLearningAgent.run_cycle().

TODO: векторизация текста, реальный pool с скрытыми метками, test_df, стратегии entropy/margin/random.
"""
from __future__ import annotations

# Псевдокод структуры итерации:
#
# model = fit(labeled_df)
# for it in range(n_iterations):
#     idx = query(pool_df, strategy=...)
#     batch = pool_df.loc[idx]
#     # симуляция: перенос истинных меток из ground_truth
#     labeled_df = pd.concat([labeled_df, batch_with_labels])
#     pool_df = pool_df.drop(idx)
#     metrics = evaluate(labeled_df, test_df)
#     history.append({"iteration": it, "n_labeled": len(labeled_df), **metrics})

def main() -> None:
    print("Скопируй структуру цикла в agents/al_agent.py и подключи sklearn / выбранную модель.")


if __name__ == "__main__":
    main()