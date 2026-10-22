"""Шаг 3: RL-дообучение и анализ.

A) reward=ratio, сетка β (KL к базовой модели) — компромисс «награда vs релевантность»
B) разные функции награды при одном β — как форма награды меняет поведение
C) multi-step RL (траектория из 5 треков)
Все модели оцениваются одинаково: релевантность (NDCG/Recall/MAP@10 по реальным следующим 20 прослушиваниям),
поведение (повторы, покрытие, концентрация, длина, популярность), награда по proxy-RM и по независимому
LightGBM-оценщику, KL к базе, Jaccard с базовыми рекомендациями. Только контексты TRAIN-периода для обучения.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import T_END, T_TEST, T_TRAIN, T_VAL, eval_split, item_stats, load_events, user_sequences, vocab_size
from src.metrics import behaviour, jaccard_vs, relevance
from src.model import CTX, make_gpt2, next_logits, topk
from src.reward import RewardModel, gbm_features, ratio_bucket, user_arrays
from src.rl import REWARDS, ContextSampler, train_rl

R, A = Path("results"), Path("artifacts")
ONLY = sys.argv[1:]  # можно запускать отдельные эксперименты по имени
ev = load_events()
V, IS = vocab_size(), item_stats()
pop_rank = np.argsort(-IS[:, 0])
VAL, TEST = eval_split(ev, T_VAL, T_TEST, CTX), eval_split(ev, T_TEST, T_END, CTX)
base = make_gpt2(V); base.load_state_dict(torch.load(A / "gpt2_base.pt")); base.eval()
rm = RewardModel(V, IS); rm.load_state_dict(torch.load(A / "rm_proxy.pt")); rm.eval()
GBM = {n: lgb.Booster(model_file=str(A / f"gbm_{n}.txt")) for n in ("ratio", "like", "cont")}
SAMPLER = ContextSampler(user_arrays(ev, T_TRAIN))
MID = np.array([0, 5, 20, 50, 85, 100], np.float32)
sub = np.random.default_rng(0).choice(len(VAL["uid"]), 1500, replace=False)
VSUB = {k: ([v[i] for i in sub] if isinstance(v, list) else v[sub]) for k, v in VAL.items()}
BASE_RECS = {"val": topk(base, VAL["ctx"], 50), "test": topk(base, TEST["ctx"], 50), "vsub": topk(base, VSUB["ctx"], 50)}


@torch.no_grad()
def reward_of_recs(split, recs, reward, k=10):
    """Средняя награда топ-k: по proxy-RM и по независимому LightGBM-оценщику."""
    ctx = torch.from_numpy(split["ctx"]); cand = torch.from_numpy(recs[:, :k])
    rb = torch.from_numpy(np.where(split["ctx"] > 0, ratio_bucket(split["ctx_ratio"]), 0))
    out_p, out_e, raw_p = [], [], []
    for b in range(0, len(ctx), 512):
        h = rm.encode(ctx[b:b + 512], rb[b:b + 512])
        p = torch.sigmoid(rm.score_h(h, ctx[b:b + 512], cand[b:b + 512]))
        out_p.append(REWARDS[reward](p, ctx[b:b + 512], cand[b:b + 512], rm.istats).numpy())
        raw_p.append(p.numpy())
    N = len(ctx)
    c_flat = np.repeat(split["ctx"], k, 0)
    r_flat = MID[np.where(split["ctx"] > 0, ratio_bucket(split["ctx_ratio"]), 0)]
    X = gbm_features(c_flat, np.repeat(r_flat, k, 0), recs[:, :k].ravel(), IS)
    pe = np.stack([GBM[n].predict(X) for n in ("ratio", "like", "cont")], -1).reshape(N, k, 3)
    re = REWARDS[reward](torch.from_numpy(pe).float(), ctx, cand, rm.istats).numpy()
    raw_p = np.concatenate(raw_p)
    return {"reward_proxy": float(np.concatenate(out_p).mean()), "reward_evaluator": float(re.mean()),
            "pred_ratio_proxy": float(raw_p[..., 0].mean()), "pred_ratio_evaluator": float(pe[..., 0].mean()),
            "pred_like_proxy": float(raw_p[..., 1].mean()), "pred_like_evaluator": float(pe[..., 1].mean()),
            "pred_cont_proxy": float(raw_p[..., 2].mean())}


@torch.no_grad()
def kl_entropy(policy, split):
    c = torch.from_numpy(split["ctx"])
    kls, ents = [], []
    for b in range(0, len(c), 512):
        lp = F.log_softmax(next_logits(policy, c[b:b + 512]), -1)
        lr = F.log_softmax(next_logits(base, c[b:b + 512]), -1)
        kls.append((lp.exp() * (lp - lr)).sum(-1)); ents.append(-(lp.exp() * lp).sum(-1))
    return float(torch.cat(kls).mean()), float(torch.cat(ents).mean())


def actual_ratio_of_hits(recs, split, k=10):
    """Реально залогированная доля прослушивания для рекомендованных треков, которые юзер потом послушал."""
    vals = []
    for r, ts, rs in zip(recs[:, :k], split["target_seq"], split["target_ratio"]):
        m = dict(zip(ts, rs))
        vals += [m[x] for x in r if x in m]
    return float(np.mean(vals)) if vals else float("nan")


def full_eval(policy, split, key, reward):
    recs = topk(policy, split["ctx"], 50)
    out = relevance(recs, split["target"], 10)
    out["recall@50"] = relevance(recs, split["target"], 50)["recall@50"]
    out["ndcg@10_new_items"] = relevance(recs, [t - h for t, h in zip(split["target"], split["hist_set"])])["ndcg@10"]
    out.update(behaviour(recs, split["hist_set"], IS, V, pop_rank=pop_rank))
    out.update(reward_of_recs(split, recs, reward))
    out["kl_to_base"], out["entropy"] = kl_entropy(policy, split)
    out["jaccard@10_vs_base"] = jaccard_vs(recs, BASE_RECS[key])
    out["actual_ratio_of_hits"] = actual_ratio_of_hits(recs, split)
    return out, recs


def quick(reward):
    def f(p):
        o, _ = full_eval(p, VSUB, "vsub", reward)
        return {k: round(o[k], 4) for k in ("ndcg@10", "repeat_rate", "reward_proxy", "reward_evaluator", "coverage", "jaccard@10_vs_base")}
    return f


EXPS = {}
for b in [0.0, 0.02, 0.1, 0.5, 1.0, 2.0]:
    EXPS[f"ratio_b{b}"] = dict(reward="ratio", beta=b)
# формы награды сравниваем при β=0.5 — первой точке сетки, где политика не схлопывается
for rw in ["ratio_lift", "ratio_novel", "listen_time", "like", "cont", "engagement"]:
    EXPS[f"{rw}_b0.5"] = dict(reward=rw, beta=0.5)
EXPS["engagement_b0.5_h5"] = dict(reward="engagement", beta=0.5, horizon=5, bs=16, G=8)

STEPS = 300
resf = R / "rl.json"
res = json.loads(resf.read_text()) if resf.exists() else {}
res.setdefault("base", {})
for rw in REWARDS:  # базовая модель под каждой наградой (для сравнения)
    if rw not in res["base"]:
        res["base"][rw] = {s: full_eval(base, sp, s, rw)[0] for s, sp in [("val", VAL), ("test", TEST)]}
        resf.write_text(json.dumps(res, indent=1))

for name, cfg in EXPS.items():
    if name in res or (ONLY and name not in ONLY):
        continue
    t0 = time.time()
    print(f"=== {name} {cfg}", flush=True)
    kw = dict(cfg)
    pol, log = train_rl(base, base, rm, SAMPLER, steps=STEPS, eval_fn=quick(cfg["reward"]), eval_every=100,
                        log=lambda r: print(" ", r, flush=True), **kw)
    out = {"cfg": cfg, "log": log, "train_sec": round(time.time() - t0, 1)}
    for s, sp in [("val", VAL), ("test", TEST)]:
        out[s], recs = full_eval(pol, sp, s, cfg["reward"])
        np.save(A / f"recs_{name}_{s}.npy", recs[:, :10])
    torch.save(pol.state_dict(), A / f"policy_{name}.pt")
    res[name] = out
    print(name, "test", {k: round(v, 4) for k, v in out["test"].items()}, flush=True)
    resf.write_text(json.dumps(res, indent=1))
np.save(A / "recs_base_test.npy", BASE_RECS["test"][:, :10])
