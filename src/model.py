"""GPT-2 для next-item prediction: токены = id треков (0 = pad)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

torch.set_num_threads(2)
CTX = 64


def make_gpt2(vocab: int, ctx: int = CTX, d: int = 128, layers: int = 2, heads: int = 4, drop: float = 0.1):
    cfg = GPT2Config(vocab_size=vocab, n_positions=ctx, n_embd=d, n_layer=layers, n_head=heads,
                     resid_pdrop=drop, embd_pdrop=drop, attn_pdrop=drop, bos_token_id=0, eos_token_id=0,
                     tie_word_embeddings=True)
    return GPT2LMHeadModel(cfg)


def last_hidden(model, ctx: torch.Tensor):
    """Скрытое состояние на последней позиции (контекст выровнен по правому краю)."""
    mask = (ctx != 0).long()
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    out = model.transformer(input_ids=ctx, attention_mask=mask, position_ids=pos)
    return out.last_hidden_state[:, -1]


def next_logits(model, ctx: torch.Tensor):
    h = last_hidden(model, ctx)
    logits = model.lm_head(h)
    logits[:, 0] = -1e9  # pad никогда не рекомендуем
    return logits


class WindowSampler:
    """Случайные окна длины CTX+1 из train-последовательностей (вес ∝ числу позиций)."""

    def __init__(self, seqs: list[np.ndarray], ctx=CTX, seed=0):
        self.seqs = [s for s in seqs if len(s) >= 2]
        self.lens = np.array([len(s) - 1 for s in self.seqs])
        self.p = self.lens / self.lens.sum()
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

    def batch(self, bs):
        us = self.rng.choice(len(self.seqs), size=bs, p=self.p)
        X = np.zeros((bs, self.ctx + 1), dtype=np.int64)
        for r, u in enumerate(us):
            s = self.seqs[u]
            end = self.rng.integers(1, len(s)) + 1  # окно заканчивается на позиции end-1
            w = s[max(0, end - self.ctx - 1):end]
            X[r, -len(w):] = w
        return torch.from_numpy(X)


def lm_loss(model, X, last=16):
    """CE next-item. Лосс считаем только на последних `last` позициях окна: на инференсе у почти всех
    юзеров полный контекст (история длинная), а голова на 20k классов — основная стоимость на CPU."""
    inp, tgt = X[:, :-1], X[:, 1:].clone()
    mask = (inp != 0).long()
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    tgt[(tgt == 0) | (inp == 0)] = -100
    h = model.transformer(input_ids=inp, attention_mask=mask, position_ids=pos).last_hidden_state[:, -last:]
    logits = model.lm_head(h)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, -last:].reshape(-1), ignore_index=-100)


def train_lm(model, sampler, steps=3000, bs=64, lr=2e-3, warmup=200, eval_fn=None, eval_every=500, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    hist, t0, run = [], time.time(), 0.0
    for step in range(1, steps + 1):
        model.train()
        loss = lm_loss(model, sampler.batch(bs))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % eval_every == 0 or step == steps:
            rec = dict(step=step, loss=round(run, 4), sec=round(time.time() - t0, 1))
            if eval_fn is not None:
                rec.update(eval_fn(model))
            hist.append(rec)
            log(rec)
    return hist


@torch.no_grad()
def topk(model, ctx: np.ndarray, k=50, bs=512, exclude: list[set] | None = None):
    model.eval()
    out = []
    for b in range(0, len(ctx), bs):
        lg = next_logits(model, torch.from_numpy(ctx[b:b + bs]))
        if exclude is not None:
            for r, ex in enumerate(exclude[b:b + bs]):
                if ex:
                    lg[r, list(ex)] = -1e9
        out.append(lg.topk(k, dim=1).indices.numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs(model, ctx: np.ndarray, bs=512):
    model.eval()
    return torch.cat([F.softmax(next_logits(model, torch.from_numpy(ctx[b:b + bs])), -1)
                      for b in range(0, len(ctx), bs)])
