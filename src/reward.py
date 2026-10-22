"""Reward model: отклик пользователя на трек-кандидат с учётом истории.

Вход:  последние 64 события юзера (трек + бакет доли прослушивания — RM, в отличие от политики,
       видит, что юзер скипал, а что дослушивал) и трек-кандидат.
Выход (4 головы):
  ratio — ожидаемая доля прослушивания (0..1), BCE с мягкой меткой played_ratio/100
  like  — P(лайк в течение суток после прослушивания)
  cont  — P(пользователь продолжит слушать в течение 30 минут) — прокси «долгосрочного» вовлечения
Признаки кандидата: эмбеддинг, log-популярность, длина, средний досмотр трека (train), сколько раз
трек встречался в контексте (повтор) и давность последнего появления.

Второй, независимый оценщик — LightGBM на ручных признаках (другой класс модели, другие ошибки).
Он не участвует в RL и нужен, чтобы ловить «взлом» proxy-RM.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import CTX, make_gpt2

N_RB = 6  # бакеты played_ratio: pad, <10, <30, <70, <100, =100
HEADS = ["ratio", "like", "cont"]


def ratio_bucket(r):
    r = np.asarray(r)
    b = np.select([r < 10, r < 30, r < 70, r < 100], [1, 2, 3, 4], 5)
    return b


def cand_feats(ctx: torch.Tensor, cand: torch.Tensor, item_stats: torch.Tensor):
    """ctx (B,L), cand (B,K) -> (B,K,6) признаки кандидата."""
    eq = ctx.unsqueeze(1) == cand.unsqueeze(2)  # B,K,L
    cnt = eq.sum(-1).float()
    L = ctx.size(1)
    # давность: позиций от конца до последнего вхождения (L+1 если не было)
    idx = torch.arange(L, device=ctx.device).float()
    last = torch.where(eq, idx, torch.full_like(idx, -1.0)).max(-1).values
    rec = torch.where(last >= 0, (L - 1 - last), torch.full_like(last, L + 1.0)) / L
    st = item_stats[cand]  # B,K,3
    return torch.stack([torch.log1p(cnt), (cnt > 0).float(), rec, torch.log1p(st[..., 0]) / 10,
                        st[..., 1] / 300.0, st[..., 2] / 100.0], -1)


class RewardModel(nn.Module):
    def __init__(self, vocab, item_stats: np.ndarray, d=128, layers=2):
        super().__init__()
        self.gpt = make_gpt2(vocab, d=d, layers=layers)
        self.rb = nn.Embedding(N_RB, d)
        self.register_buffer("istats", torch.tensor(item_stats))
        self.head = nn.Sequential(nn.Linear(3 * d + 6, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
                                  nn.Linear(128, len(HEADS)))

    def encode(self, ctx, ctx_rb):
        mask = (ctx != 0).long()
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        emb = self.gpt.transformer.wte(ctx) + self.rb(ctx_rb)
        return self.gpt.transformer(inputs_embeds=emb, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]

    def score_h(self, h, ctx, cand):
        """h (B,d) — закодированный контекст; cand (B,K) -> логиты (B,K,3)."""
        e = self.gpt.transformer.wte(cand)
        hh = h.unsqueeze(1).expand_as(e)
        f = cand_feats(ctx, cand, self.istats)
        return self.head(torch.cat([hh, e, hh * e, f], -1))

    def forward(self, ctx, ctx_rb, cand):
        return self.score_h(self.encode(ctx, ctx_rb), ctx, cand)

    @torch.no_grad()
    def predict(self, ctx, ctx_rb, cand, bs=512):
        self.eval()
        out = []
        for b in range(0, len(ctx), bs):
            lg = self(ctx[b:b + bs], ctx_rb[b:b + bs], cand[b:b + bs])
            out.append(torch.sigmoid(lg))
        return torch.cat(out)  # B,K,3 вероятности/доли


class EventSampler:
    """Обучающие примеры: (последние 64 события до j, событие j) из train-периода."""

    def __init__(self, user_arrays, seed=0):
        self.U = [u for u in user_arrays if len(u["tok"]) >= 2]
        lens = np.array([len(u["tok"]) - 1 for u in self.U])
        self.p = lens / lens.sum()
        self.rng = np.random.default_rng(seed)

    def batch(self, bs):
        us = self.rng.choice(len(self.U), size=bs, p=self.p)
        ctx = np.zeros((bs, CTX), dtype=np.int64)
        rb = np.zeros((bs, CTX), dtype=np.int64)
        y = np.zeros((bs, 3), dtype=np.float32)
        cand = np.zeros((bs, 1), dtype=np.int64)
        for r, u in enumerate(us):
            a = self.U[u]
            j = self.rng.integers(1, len(a["tok"]))
            lo = max(0, j - CTX)
            ctx[r, CTX - (j - lo):] = a["tok"][lo:j]
            rb[r, CTX - (j - lo):] = a["rb"][lo:j]
            cand[r, 0] = a["tok"][j]
            y[r] = (a["ratio"][j] / 100.0, a["like"][j], a["cont"][j])
        return tuple(map(torch.from_numpy, (ctx, rb, cand, y)))


def rm_loss(model, ctx, rb, cand, y, like_w=5.0):
    lg = model(ctx, rb, cand)[:, 0]
    l_ratio = F.binary_cross_entropy_with_logits(lg[:, 0], y[:, 0])
    l_like = F.binary_cross_entropy_with_logits(lg[:, 1], y[:, 1], pos_weight=torch.tensor(like_w))
    l_cont = F.binary_cross_entropy_with_logits(lg[:, 2], y[:, 2])
    return l_ratio + 0.5 * l_like + 0.5 * l_cont


def train_rm(model, sampler, steps=3000, bs=256, lr=1e-3, log=print, eval_fn=None, eval_every=500):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, run, hist = time.time(), None, []
    for s in range(1, steps + 1):
        model.train()
        loss = rm_loss(model, *sampler.batch(bs))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if s % eval_every == 0 or s == steps:
            rec = dict(step=s, loss=round(run, 4), sec=round(time.time() - t0, 1))
            if eval_fn:
                rec.update(eval_fn(model))
            hist.append(rec)
            log(rec)
    return hist


def user_arrays(ev, t_day):
    """События юзеров строго до t_day -> список dict numpy-срезов (без python-списков: экономит RAM)."""
    import polars as pl
    from src.data import DAY
    e = ev.filter(pl.col("timestamp") < t_day * DAY).sort(["uid", "timestamp"])
    uid = e["uid"].to_numpy()
    tok = e["tok"].to_numpy().astype(np.int64)
    ratio = np.clip(e["played_ratio_pct"].to_numpy().astype(np.float32), 0, 100)  # бывают значения >100 (перемотка)
    like = e["like"].to_numpy().astype(np.float32)
    cont = e["cont"].to_numpy().astype(np.float32)
    rb = ratio_bucket(ratio).astype(np.int64)
    cut = np.flatnonzero(np.diff(uid)) + 1
    bounds = np.concatenate([[0], cut, [len(uid)]])
    return [dict(tok=tok[a:b], ratio=ratio[a:b], rb=rb[a:b], like=like[a:b], cont=cont[a:b])
            for a, b in zip(bounds[:-1], bounds[1:])]


# ---------------------------------------------------------------- независимый оценщик (LightGBM)
def gbm_features(ctx: np.ndarray, ctx_ratio: np.ndarray, cand: np.ndarray, item_stats: np.ndarray):
    """Ручные признаки пары (история, кандидат). ctx/ctx_ratio (N,L), cand (N,)."""
    m = ctx != 0
    n = m.sum(1).clip(min=1)
    eq = ctx == cand[:, None]
    cnt = eq.sum(1)
    L = ctx.shape[1]
    last = np.where(eq.any(1), L - 1 - (L - 1 - np.argmax(eq[:, ::-1], 1)), L + 1)
    rat = np.where(m, ctx_ratio, np.nan)
    with np.errstate(invalid="ignore"):
        u_mean = np.nanmean(rat, 1)
        u_skip = np.nanmean(np.where(m, ctx_ratio < 10, np.nan), 1)
        u_last5 = np.nanmean(rat[:, -5:], 1)
        cand_prev_ratio = np.nanmean(np.where(eq, ctx_ratio, np.nan), 1)
    st = item_stats[cand]
    return np.column_stack([u_mean, u_skip, u_last5, n, cnt, last, cand_prev_ratio,
                            np.log1p(st[:, 0]), st[:, 1], st[:, 2]]).astype(np.float32)


GBM_COLS = ["u_mean_ratio", "u_skip_rate", "u_last5_ratio", "ctx_len", "cand_cnt_in_ctx", "cand_last_pos",
            "cand_prev_ratio", "log_pop", "length", "item_mean_ratio"]
