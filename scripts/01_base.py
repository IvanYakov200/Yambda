"""Шаг 1: базовая GPT-2 (next-item) + простые бейзлайны на одинаковом протоколе оценки."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import T_END, T_TEST, T_TRAIN, T_VAL, eval_split, item_stats, load_events, user_sequences, vocab_size
from src.metrics import behaviour, relevance
from src.model import CTX, WindowSampler, make_gpt2, topk, train_lm

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
R, A = Path("results"), Path("artifacts")
R.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
torch.manual_seed(0)

ev = load_events()
V, IS = vocab_size(), item_stats()
pop_rank = np.argsort(-IS[:, 0])
VAL = eval_split(ev, T_VAL, T_TEST, CTX)
TEST = eval_split(ev, T_TEST, T_END, CTX)
print("val users", len(VAL["uid"]), "test users", len(TEST["uid"]))
sub = np.random.default_rng(0).choice(len(VAL["uid"]), 1500, replace=False)


def quick_eval(m):
    rec = topk(m, VAL["ctx"][sub], k=10)
    return {"val_ndcg@10_sub": round(relevance(rec, [VAL["target"][i] for i in sub])["ndcg@10"], 4)}


seqs = [np.array(s) for s in user_sequences(ev, T_TRAIN)["tok"].to_list()]
model = make_gpt2(V)
RESUME = len(sys.argv) > 2 and sys.argv[2] == "resume"
if RESUME:  # второй этап: продолжаем с чекпоинта с меньшим lr (warm restart косинуса)
    model.load_state_dict(torch.load(A / "gpt2_base.pt"))
log = train_lm(model, WindowSampler(seqs, seed=1 if RESUME else 0), steps=STEPS, bs=128, lr=1e-3 if RESUME else 2e-3,
               warmup=50 if RESUME else 200, eval_fn=quick_eval, eval_every=250, log=lambda r: print(r, flush=True))
if RESUME:
    prev = json.loads((R / "base_stage1.json").read_text())["train_log"]
    log = prev + [dict(r, step=r["step"] + prev[-1]["step"]) for r in log]
torch.save(model.state_dict(), A / "gpt2_base.pt")


def popular_recs(split):
    return np.tile(pop_rank[pop_rank != 0][:50], (len(split["uid"]), 1))


def history_recs(split, hist_tokens):
    """Сильный бейзлайн для музыки: самые частые треки юзера за последние 64 прослушивания."""
    out = []
    for ctx in split["ctx"]:
        c = Counter([t for t in ctx if t != 0])
        r = [t for t, _ in c.most_common(50)]
        r += [p for p in pop_rank if p not in c and p != 0][: 50 - len(r)]
        out.append(r[:50])
    return np.array(out)


res = {"train_log": log}
for name, split in [("val", VAL), ("test", TEST)]:
    for mname, recs in [("popular", popular_recs(split)), ("history_top", history_recs(split, None)),
                        ("gpt2_base", topk(model, split["ctx"], k=50))]:
        r = relevance(recs, split["target"], k=10)
        r.update({k: v for k, v in relevance(recs, split["target"], k=50).items() if k.startswith("recall")})
        new_t = [t - h for t, h in zip(split["target"], split["hist_set"])]
        r["ndcg@10_new_items"] = relevance(recs, new_t)["ndcg@10"]
        r.update(behaviour(recs, split["hist_set"], IS, V, pop_rank=pop_rank))
        res.setdefault(mname, {})[name] = r
        print(name, mname, {k: round(v, 4) for k, v in r.items()}, flush=True)
(R / "base.json").write_text(json.dumps(res, indent=1))
