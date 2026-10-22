"""Проверки временного разбиения и отсутствия утечек. Запуск: pytest -q tests/"""
import pickle

import numpy as np
import polars as pl

from src.data import DAY, PROC, T_END, T_TEST, T_TRAIN, T_VAL, eval_split, load_events
from src.model import CTX
from src.reward import user_arrays

EV = load_events()


def test_vocab_built_on_train_only():
    vocab = pickle.load(open(PROC / "vocab.pkl", "rb"))
    train_items = set(EV.filter(pl.col("timestamp") < T_TRAIN * DAY)["item_id"].unique().to_list())
    assert set(vocab) <= train_items


def test_training_arrays_strictly_before_train_end():
    ua = user_arrays(EV, T_TRAIN)
    n = sum(len(u["tok"]) for u in ua)
    assert n == EV.filter(pl.col("timestamp") < T_TRAIN * DAY).height


def test_eval_contexts_are_past_and_targets_are_future():
    for t, te in [(T_VAL, T_TEST), (T_TEST, T_END)]:
        sp = eval_split(EV, t, te, CTX)
        e = EV.sort(["uid", "timestamp"])
        for k in np.random.default_rng(0).choice(len(sp["uid"]), 50, replace=False):
            u = sp["uid"][k]
            eu = e.filter(pl.col("uid") == u)
            past = eu.filter(pl.col("timestamp") < t * DAY)["tok"].to_list()
            fut = eu.filter((pl.col("timestamp") >= t * DAY) & (pl.col("timestamp") < te * DAY))["tok"].to_list()[:20]
            ctx = [x for x in sp["ctx"][k] if x != 0]
            assert ctx == past[-CTX:]
            assert sp["target_seq"][k] == fut


def test_val_and_test_windows_do_not_overlap_training():
    assert T_TRAIN <= T_VAL < T_TEST < T_END
