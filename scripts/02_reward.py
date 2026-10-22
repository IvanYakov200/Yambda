"""Шаг 2: reward model (proxy, для RL) + независимый оценщик (LightGBM) и их качество на отложенных данных.

Отложенная выборка: первые 20 событий каждого юзера в [280, 290) (val) и [290, 300] (test),
контекст — 64 события строго перед оцениваемым. Модели учатся только на событиях < 280.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DAY, N_TARGET, T_END, T_TEST, T_TRAIN, T_VAL, item_stats, load_events, user_sequences, vocab_size
from src.model import CTX
from src.reward import (GBM_COLS, EventSampler, RewardModel, gbm_features, ratio_bucket, train_rm, user_arrays)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
R, A = Path("results"), Path("artifacts")
torch.manual_seed(1)
ev = load_events()
V, IS = vocab_size(), item_stats()


def holdout(t_day, t_end):
    """Примеры для оценки RM: (контекст, ответ) для первых N_TARGET событий юзера в окне."""
    e = ev.filter(pl.col("timestamp") < t_end * DAY).group_by("uid", maintain_order=True).agg(
        pl.col("tok"), pl.col("played_ratio_pct"), pl.col("like"), pl.col("cont"), pl.col("timestamp"))
    C, RB, CR, Y, CAND = [], [], [], [], []
    for tok, rat, lk, ct, ts in zip(*[e[c].to_list() for c in ["tok", "played_ratio_pct", "like", "cont", "timestamp"]]):
        ts = np.array(ts)
        start = np.searchsorted(ts, t_day * DAY)
        if start < 5:
            continue
        for j in range(start, min(start + N_TARGET, len(tok))):
            lo = max(0, j - CTX)
            c = np.zeros(CTX, np.int64); c[CTX - (j - lo):] = tok[lo:j]
            r = np.zeros(CTX, np.float32); r[CTX - (j - lo):] = np.minimum(rat[lo:j], 100)
            C.append(c); CR.append(r); RB.append(np.where(c > 0, ratio_bucket(r), 0))
            CAND.append(tok[j]); Y.append((min(rat[j], 100) / 100.0, lk[j], ct[j]))
    return dict(ctx=np.array(C), rb=np.array(RB), ctx_ratio=np.array(CR), cand=np.array(CAND), y=np.array(Y, np.float32))


VAL, TEST = holdout(T_VAL, T_TEST), holdout(T_TEST, T_END)
print("holdout examples", len(VAL["cand"]), len(TEST["cand"]), flush=True)


def scores(pred_ratio, pred_like, pred_cont, H):
    y = H["y"]
    rep = (H["ctx"] == H["cand"][:, None]).any(1)
    out = {
        "ratio_mae": float(np.abs(pred_ratio - y[:, 0]).mean()),
        "ratio_rmse": float(np.sqrt(((pred_ratio - y[:, 0]) ** 2).mean())),
        "ratio_spearman": float(spearmanr(pred_ratio, y[:, 0]).statistic),
        "skip_auc": float(roc_auc_score(y[:, 0] < 0.1, -pred_ratio)),
        "skip_auc_new_items": float(roc_auc_score(y[~rep, 0] < 0.1, -pred_ratio[~rep])),
        "skip_auc_repeats": float(roc_auc_score(y[rep, 0] < 0.1, -pred_ratio[rep])),
        "share_repeats_in_holdout": float(rep.mean()),
    }
    if pred_like is not None:
        out["like_auc"] = float(roc_auc_score(y[:, 1], pred_like))
        out["cont_auc"] = float(roc_auc_score(y[:, 2], pred_cont))
    return out


def rm_predict(rm, H):
    p = rm.predict(torch.from_numpy(H["ctx"]), torch.from_numpy(H["rb"]), torch.from_numpy(H["cand"][:, None]))[:, 0].numpy()
    return p[:, 0], p[:, 1], p[:, 2]


# ---------------------------------------------------------------- proxy RM (нейросеть)
UA = user_arrays(ev, T_TRAIN)
rm = RewardModel(V, IS)
base = torch.load(A / "gpt2_base.pt")
rm.gpt.load_state_dict(base)  # инициализация энкодера из базовой GPT-2 (как RM из SFT-модели в RLHF)
sub = np.random.default_rng(0).choice(len(VAL["cand"]), 20000, replace=False)
VSUB = {k: v[sub] for k, v in VAL.items()}


def ev_fn(m):
    pr, pl_, pc = rm_predict(m, VSUB)
    s = scores(pr, pl_, pc, VSUB)
    return {k: round(s[k], 4) for k in ("ratio_mae", "skip_auc", "like_auc", "cont_auc")}


if (A / "rm_proxy.pt").exists() and "retrain" not in sys.argv:  # чекпоинт уже есть — не переобучаем
    rm.load_state_dict(torch.load(A / "rm_proxy.pt"))
    log = json.loads((R / "rm_train_log.json").read_text())
else:
    log = train_rm(rm, EventSampler(UA), steps=STEPS, eval_fn=ev_fn, eval_every=500, log=lambda r: print(r, flush=True))
    torch.save(rm.state_dict(), A / "rm_proxy.pt")
    (R / "rm_train_log.json").write_text(json.dumps(log))

# ---------------------------------------------------------------- независимый оценщик (LightGBM)
rng = np.random.default_rng(7)
S = EventSampler(UA, seed=7)
Xs, Ys = [], []
for _ in range(120):
    ctx, rb, cand, y = S.batch(2500)
    ctx, cand, y = ctx.numpy(), cand.numpy()[:, 0], y.numpy()
    # для признаков нужен «сырой» ratio контекста: восстановим из бакета по середине интервала
    mid = np.array([0, 5, 20, 50, 85, 100], np.float32)
    Xs.append(gbm_features(ctx, mid[rb.numpy()], cand, IS)); Ys.append(y)
X, Y = np.concatenate(Xs), np.concatenate(Ys)
gbm = {}
for k, name in enumerate(["ratio", "like", "cont"]):
    params = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.9, num_threads=2, verbose=-1, seed=0)
    gbm[name] = lgb.train(params, lgb.Dataset(X, Y[:, k], feature_name=GBM_COLS), 400)
    gbm[name].save_model(str(A / f"gbm_{name}.txt"))


def gbm_predict(H):
    mid = np.array([0, 5, 20, 50, 85, 100], np.float32)
    F_ = gbm_features(H["ctx"], mid[H["rb"]], H["cand"], IS)
    return tuple(gbm[n].predict(F_) for n in ("ratio", "like", "cont"))


# ---------------------------------------------------------------- отчёт
res = {"train_log": log}
gm = float(np.mean([u["ratio"].mean() for u in UA]) / 100)
for split, H in [("val", VAL), ("test", TEST)]:
    X_ = gbm_features(H["ctx"], np.array([0, 5, 20, 50, 85, 100], np.float32)[H["rb"]], H["cand"], IS)
    u_mean = np.nan_to_num(X_[:, 0] / 100, nan=gm)
    item_mean = IS[H["cand"], 2] / 100
    prev = np.where(np.isnan(X_[:, 6]), item_mean * 100, X_[:, 6]) / 100
    res.setdefault("baseline_global_mean", {})[split] = scores(np.full(len(H["cand"]), gm), None, None, H)
    res.setdefault("baseline_item_mean", {})[split] = scores(item_mean, None, None, H)
    res.setdefault("baseline_user_mean", {})[split] = scores(u_mean, None, None, H)
    res.setdefault("baseline_user+item", {})[split] = scores(0.5 * u_mean + 0.5 * item_mean, None, None, H)
    res.setdefault("baseline_prev_ratio_of_track", {})[split] = scores(prev, None, None, H)
    res.setdefault("rm_proxy_gpt2", {})[split] = scores(*rm_predict(rm, H), H)
    res.setdefault("evaluator_lgbm", {})[split] = scores(*gbm_predict(H), H)
    # согласованность proxy и оценщика
    pr = rm_predict(rm, H)[0]
    res.setdefault("proxy_vs_evaluator_spearman", {})[split] = float(spearmanr(pr, gbm_predict(H)[0]).statistic)
for k, v in res.items():
    if k != "train_log":
        print(k, json.dumps(v["test"] if isinstance(v, dict) else v)[:300])
(R / "reward.json").write_text(json.dumps(res, indent=1))
