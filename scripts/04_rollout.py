"""Шаг 4: генерация 5 треков подряд (каждый добавляется в контекст) — дрейф от реальной истории.

Сравниваем три способа получить 5 рекомендаций:
  one-shot      — топ-5 из одного прохода по реальной истории (стандартная выдача)
  autoregressive— 5 шагов, сгенерированный трек дописывается в контекст как будто прослушан
                  (его «отклик» для RM = предсказанный бакет досмотра)
  teacher-forced— на шаге h в контекст дописываются РЕАЛЬНЫЕ следующие треки юзера (оракул:
                  показывает, сколько теряется из-за того, что сгенерированное ≠ прослушанное)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import T_END, T_TEST, eval_split, item_stats, load_events, vocab_size
from src.model import CTX, make_gpt2, next_logits, topk
from src.reward import RewardModel, ratio_bucket
from src.rl import rollout

R, A = Path("results"), Path("artifacts")
H = 5
ev = load_events()
V, IS = vocab_size(), item_stats()
TEST = eval_split(ev, T_TEST, T_END, CTX)
idx = np.random.default_rng(1).choice(len(TEST["uid"]), 3000, replace=False)
ctx, cr = TEST["ctx"][idx], TEST["ctx_ratio"][idx]
tseq = [TEST["target_seq"][i] for i in idx]
tset = [TEST["target"][i] for i in idx]
hist = [TEST["hist_set"][i] for i in idx]
rm = RewardModel(V, IS); rm.load_state_dict(torch.load(A / "rm_proxy.pt")); rm.eval()


def load(name):
    m = make_gpt2(V)
    m.load_state_dict(torch.load(A / ("gpt2_base.pt" if name == "base" else f"policy_{name}.pt")))
    return m.eval()


@torch.no_grad()
def proxy_on_real_ctx(items):
    c = torch.from_numpy(ctx); b = torch.from_numpy(np.where(ctx > 0, ratio_bucket(cr), 0))
    return torch.sigmoid(rm.score_h(rm.encode(c, b), c, torch.from_numpy(items)))[..., 0].numpy()


@torch.no_grad()
def teacher_forced(m):
    """На шаге h контекст = реальная история + реальные первые h треков из будущего."""
    c = torch.from_numpy(ctx.copy()); hits = []
    for h in range(H):
        a = next_logits(m, c).argmax(-1).numpy()
        hits.append(np.mean([x in t for x, t in zip(a, tset)]))
        nxt = np.array([s[h] if len(s) > h else 0 for s in tseq])
        c = torch.cat([c[:, 1:], torch.from_numpy(nxt)[:, None]], 1)
    return hits


names = ["base"] + [n for n in ["ratio_b0.0", "ratio_b0.5", "ratio_lift_b0.5", "ratio_novel_b0.5", "listen_time_b0.5", "engagement_b0.5", "engagement_b0.5_h5"]
                    if (A / f"policy_{n}.pt").exists()]
res = {}
for n in names:
    m = load(n)
    toks, preds = rollout(m, rm, ctx, cr, horizon=H, greedy=True)
    one = topk(m, ctx, H)
    real_ctx_pred = proxy_on_real_ctx(toks)
    out = {
        "autoregressive": {
            "hit_by_step": [float(np.mean([x in t for x, t in zip(toks[:, h], tset)])) for h in range(H)],
            "exact_next_by_step": [float(np.mean([len(s) > h and x == s[h] for x, s in zip(toks[:, h], tseq)])) for h in range(H)],
            "repeat_by_step": [float(np.mean([x in hs for x, hs in zip(toks[:, h], hist)])) for h in range(H)],
            "pred_ratio_gen_ctx_by_step": preds[..., 0].mean(0).tolist(),
            "pred_ratio_real_ctx_by_step": real_ctx_pred.mean(0).tolist(),
            "pred_cont_by_step": preds[..., 2].mean(0).tolist(),
            "share_same_artistlike_as_step1": float(np.mean(toks[:, 1:] == toks[:, :1])),
            "unique_items_total": int(len(np.unique(toks))),
        },
        "one_shot": {
            "hit_by_rank": [float(np.mean([x in t for x, t in zip(one[:, h], tset)])) for h in range(H)],
            "overlap_with_autoregressive": float(np.mean([len(set(a) & set(b)) / H for a, b in zip(one, toks)])),
        },
        "teacher_forced_hit_by_step": teacher_forced(m),
    }
    res[n] = out
    print(n, json.dumps(out)[:600], flush=True)
(R / "rollout.json").write_text(json.dumps(res, indent=1))
