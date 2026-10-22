"""Метрики релевантности и поведения рекомендаций."""
from __future__ import annotations

from collections import Counter

import numpy as np


def relevance(recs: np.ndarray, targets: list[set], k=10, hist: list[set] | None = None) -> dict:
    """Recall@k / NDCG@k / MAP@k по множеству треков из следующих N прослушиваний.
    Если передан hist, дополнительно считаем «новизну»: метрики только по новым для юзера трекам."""
    disc = 1 / np.log2(np.arange(2, k + 2))
    R, N, M, H = [], [], [], []
    for r, t in zip(recs[:, :k], targets):
        if not t:
            continue
        rel = np.array([x in t for x in r], dtype=float)
        m = min(len(t), k)
        R.append(rel.sum() / m)
        N.append((rel * disc).sum() / disc[:m].sum())
        M.append((np.cumsum(rel) / np.arange(1, k + 1) * rel).sum() / m)
        H.append(rel.any())
    out = {f"recall@{k}": float(np.mean(R)), f"ndcg@{k}": float(np.mean(N)), f"map@{k}": float(np.mean(M)),
           f"hit@{k}": float(np.mean(H))}
    return out


def behaviour(recs: np.ndarray, hist: list[set], item_stats: np.ndarray, vocab: int, k=10, pop_rank=None) -> dict:
    """Чем рекомендации отличаются: повторы, покрытие, концентрация, популярность, длина треков."""
    top = recs[:, :k]
    flat = top.ravel()
    c = Counter(flat.tolist())
    counts = np.array(sorted(c.values()))
    n = len(flat)
    # Gini экспозиции по всему словарю
    full = np.zeros(vocab - 1)
    full[: len(counts)] = counts
    full = np.sort(full)
    cum = np.cumsum(full)
    gini = float(1 - 2 * (cum / cum[-1]).sum() / len(full) + 1 / len(full))
    rep = np.mean([np.mean([x in h for x in r]) for r, h in zip(top, hist)])
    pop, length = item_stats[:, 0], item_stats[:, 1]
    out = {
        "repeat_rate": float(rep),                      # доля рекомендаций из уже прослушанного
        "coverage": len(c) / (vocab - 1),               # доля словаря, попавшая хоть кому-то в топ-k
        "top1_item_share_users": c.most_common(1)[0][1] / len(top),  # у какой доли юзеров есть самый частый трек
        "top10_items_exposure": sum(v for _, v in c.most_common(10)) / n,
        "gini_exposure": gini,
        "mean_log_pop": float(np.log1p(pop[flat]).mean()),
        "mean_length_sec": float(length[flat].mean()),
    }
    if pop_rank is not None:
        out["share_top100_popular"] = float(np.isin(flat, pop_rank[:100]).mean())
    return out


def jaccard_vs(a: np.ndarray, b: np.ndarray, k=10) -> float:
    """Насколько списки двух моделей совпадают для тех же юзеров."""
    return float(np.mean([len(set(x[:k]) & set(y[:k])) / len(set(x[:k]) | set(y[:k])) for x, y in zip(a, b)]))
