"""Подготовка Yambda-50M (сжатый срез: дни 180-300, топ-20k треков) и временное разбиение.

Глобальные границы времени (в днях от начала датасета):
  train   : события с t < T_TRAIN (=280)  — на них учатся GPT-2, reward model и RL
  val     : контекст = всё до 280, таргет = первые N событий в [280, 290)
  test    : контекст = всё до 290, таргет = первые N событий в [290, 300]
Модели НИКОГДА не видят событий >= 280 при обучении; контекст теста (280..290) — это прошлое
на момент теста, его использование легально (как у реального сервиса).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

DAY = 86400
T_TRAIN, T_VAL, T_TEST, T_END = 280, 280, 290, 301
N_TARGET = 20          # таргет: первые 20 прослушиваний после границы
SESSION_GAP = 30 * 60  # продолжение сессии: следующее событие в пределах 30 минут
PAD = 0
ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data", ROOT / "data" / "proc"


def prepare():
    PROC.mkdir(parents=True, exist_ok=True)
    d = pl.read_parquet(RAW / "listens.parquet").sort(["uid", "timestamp"])
    likes = pl.read_parquet(RAW / "likes.parquet")
    dislikes = pl.read_parquet(RAW / "dislikes.parquet")

    # словарь: только по train-периоду (без утечки), id 1..V, 0 = pad
    cnt = d.filter(pl.col("timestamp") < T_TRAIN * DAY).group_by("item_id").len().sort(["len", "item_id"], descending=[True, False])
    vocab = cnt["item_id"].to_list()
    imap = {it: k + 1 for k, it in enumerate(vocab)}
    d = d.filter(pl.col("item_id").is_in(vocab)).with_columns(pl.col("item_id").replace_strict(imap).cast(pl.Int32).alias("tok"))

    # отклики на событие
    def flag_join(ev, name):
        j = (d.select("uid", "item_id", "timestamp")
             .join(ev.select("uid", "item_id", pl.col("timestamp").alias("et")), on=["uid", "item_id"], how="inner")
             .filter((pl.col("et") >= pl.col("timestamp")) & (pl.col("et") < pl.col("timestamp") + DAY))
             .select("uid", "item_id", "timestamp").unique().with_columns(pl.lit(1, pl.Int8).alias(name)))
        return j

    d = d.join(flag_join(likes, "like"), on=["uid", "item_id", "timestamp"], how="left") \
         .join(flag_join(dislikes, "dislike"), on=["uid", "item_id", "timestamp"], how="left") \
         .with_columns(pl.col("like").fill_null(0), pl.col("dislike").fill_null(0))
    d = d.with_columns(
        (pl.col("timestamp").shift(-1).over("uid") - pl.col("timestamp")).alias("gap_next"),
    ).with_columns(
        # долгосрочный прокси: пользователь продолжил слушать (следующее событие в пределах 30 мин)
        ((pl.col("gap_next") <= SESSION_GAP) & pl.col("gap_next").is_not_null()).cast(pl.Int8).alias("cont"),
        # повтор: трек уже был в истории пользователя
        (~pl.col("tok").is_first_distinct().over("uid")).cast(pl.Int8).alias("repeat"),
    )
    d.write_parquet(PROC / "events.parquet")

    # статические признаки треков (по train)
    tr = d.filter(pl.col("timestamp") < T_TRAIN * DAY)
    item = (tr.group_by("tok").agg(pl.len().alias("pop"), pl.col("track_length_seconds").median().alias("length"),
                                   pl.col("played_ratio_pct").mean().alias("mean_ratio"))
            .sort("tok"))
    V = len(vocab)
    arr = np.zeros((V + 1, 3), dtype=np.float32)
    arr[item["tok"].to_numpy()] = item.select("pop", "length", "mean_ratio").to_numpy()
    np.save(PROC / "item_stats.npy", arr)  # [pop, length_sec, mean_ratio]
    with open(PROC / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"events={d.height} users={d['uid'].n_unique()} V={V} like_rate={d['like'].mean():.4f} "
          f"cont_rate={d['cont'].mean():.3f} repeat={d['repeat'].mean():.3f}")


def load_events():
    return pl.read_parquet(PROC / "events.parquet")


def item_stats():
    return np.load(PROC / "item_stats.npy")


def vocab_size():
    with open(PROC / "vocab.pkl", "rb") as f:
        return len(pickle.load(f)) + 1


def user_sequences(ev: pl.DataFrame, t_end_day: int):
    """Последовательности событий каждого пользователя строго до t_end_day."""
    e = ev.filter(pl.col("timestamp") < t_end_day * DAY)
    g = e.group_by("uid", maintain_order=True).agg(
        pl.col("tok"), pl.col("played_ratio_pct"), pl.col("like"), pl.col("cont"), pl.col("timestamp"))
    return g


def eval_split(ev: pl.DataFrame, t_day: int, t_end_day: int, ctx_len: int, min_hist: int = 5):
    """Контекст = последние ctx_len событий до t_day; таргет = первые N_TARGET событий в [t_day, t_end_day).

    Возвращает dict: uid, ctx (U, L) left-padded, target (список множеств токенов), target_seq (упорядоченный),
    hist_set (все треки истории до t_day — для метрик повторов/новизны) и отклики таргета.
    """
    hist = ev.filter(pl.col("timestamp") < t_day * DAY).group_by("uid", maintain_order=True).agg(
        pl.col("tok"), pl.col("played_ratio_pct").alias("ratio_h"))
    fut = (ev.filter((pl.col("timestamp") >= t_day * DAY) & (pl.col("timestamp") < t_end_day * DAY))
           .group_by("uid", maintain_order=True).agg(pl.col("tok").head(N_TARGET), pl.col("played_ratio_pct").head(N_TARGET),
                                                     pl.col("track_length_seconds").head(N_TARGET)))
    j = hist.join(fut, on="uid", suffix="_f").filter(pl.col("tok").list.len() >= min_hist).sort("uid")
    ctx = np.zeros((j.height, ctx_len), dtype=np.int64)
    ctx_ratio = np.zeros((j.height, ctx_len), dtype=np.float32)
    for r, (s, q) in enumerate(zip(j["tok"].to_list(), j["ratio_h"].to_list())):
        s, q = s[-ctx_len:], q[-ctx_len:]
        ctx[r, -len(s):] = s
        ctx_ratio[r, -len(q):] = q
    return dict(uid=j["uid"].to_numpy(), ctx=ctx, ctx_ratio=ctx_ratio, hist_set=[set(x) for x in j["tok"].to_list()],
                target_seq=j["tok_f"].to_list(), target=[set(x) for x in j["tok_f"].to_list()],
                target_ratio=j["played_ratio_pct"].to_list())


if __name__ == "__main__":
    prepare()
