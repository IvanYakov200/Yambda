"""RL-дообучение GPT-2 рекомендателя против reward model.

Метод — GRPO-подобный policy gradient (как в RLHF для LLM, но «ответ» — это трек):
  для каждого контекста s сэмплируем G треков a_1..a_G ~ π(·|s), считаем r_i = R(s, a_i) по RM,
  advantage A_i = (r_i - mean(r)) / (std(r) + eps)  — группа заменяет value-функцию,
  loss = -mean(A_i * log π(a_i|s)) + β * KL(π(·|s) || π_ref(·|s)).
KL считаем ТОЧНО по всему словарю (у нас есть все логиты), а не по сэмплам — это дешевле и стабильнее.
Вариант multi-step: траектория из H треков, каждый следующий генерируется с уже добавленными
в контекст предыдущими (их «отклик» для RM = предсказанный бакет досмотра), награда — сумма по шагам.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.model import next_logits
from src.reward import ratio_bucket

REWARDS = {
    # имя: функция от (pred (B,K,3) [ratio, like, cont], ctx, cand, item_stats) -> (B,K)
    "ratio": lambda p, ctx, cand, st: p[..., 0],
    "listen_time": lambda p, ctx, cand, st: p[..., 0] * st[cand, 1] / 200.0,  # ожидаемые секунды / 200
    "like": lambda p, ctx, cand, st: p[..., 1] * 10.0,
    "cont": lambda p, ctx, cand, st: p[..., 2],
    # досмотр, но без «халявы» за повторы уже знакомого
    "ratio_novel": lambda p, ctx, cand, st: p[..., 0] - 0.5 * (ctx.unsqueeze(1) == cand.unsqueeze(2)).any(-1).float(),
    # персональный «прирост» досмотра: насколько трек зайдёт ИМЕННО этому юзеру сверх своего среднего досмотра
    # (убирает выгоду от треков, которые все дослушивают — коротких/«безопасных»)
    "ratio_lift": lambda p, ctx, cand, st: p[..., 0] - st[cand, 2] / 100.0,
    # «долгосрочный» прокси: удержание в сессии + лайк + досмотр, штраф за повтор
    "engagement": lambda p, ctx, cand, st: (0.4 * p[..., 0] + 0.4 * p[..., 2] + 2.0 * p[..., 1]
                                            - 0.3 * (ctx.unsqueeze(1) == cand.unsqueeze(2)).any(-1).float()),
}


class ContextSampler:
    """Состояния для RL: контексты из TRAIN-периода (64 трека + бакеты досмотра для RM)."""

    def __init__(self, user_arrays, ctx_len=64, seed=0):
        self.U = [u for u in user_arrays if len(u["tok"]) >= 10]
        self.L = ctx_len
        self.rng = np.random.default_rng(seed)

    def batch(self, bs):
        ctx = np.zeros((bs, self.L), np.int64)
        rb = np.zeros((bs, self.L), np.int64)
        for r, u in enumerate(self.rng.integers(0, len(self.U), bs)):
            a = self.U[u]
            j = self.rng.integers(5, len(a["tok"]) + 1)
            lo = max(0, j - self.L)
            ctx[r, self.L - (j - lo):] = a["tok"][lo:j]
            rb[r, self.L - (j - lo):] = a["rb"][lo:j]
        return torch.from_numpy(ctx), torch.from_numpy(rb)


def append_token(ctx, rb, tok, tok_rb):
    """Сдвигаем окно влево и дописываем сгенерированный трек (сгенерированный ≠ прослушанный!)."""
    return torch.cat([ctx[:, 1:], tok[:, None]], 1), torch.cat([rb[:, 1:], tok_rb[:, None]], 1)


@torch.no_grad()
def rm_scores(rm, ctx, rb, cand, istats_t, reward):
    h = rm.encode(ctx, rb)
    p = torch.sigmoid(rm.score_h(h, ctx, cand))
    return REWARDS[reward](p, ctx, cand, istats_t), p


def train_rl(policy, ref, rm, sampler, reward="ratio", beta=0.05, steps=300, bs=64, G=16, lr=5e-5,
             horizon=1, temp=1.0, log=print, eval_fn=None, eval_every=100, seed=0):
    torch.manual_seed(seed)
    policy = copy.deepcopy(policy)
    ref.eval(); rm.eval()
    for p in list(ref.parameters()) + list(rm.parameters()):
        p.requires_grad_(False)
    for p in policy.parameters():  # ref мог быть заморожен прошлым запуском — deepcopy копирует флаг
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ist = rm.istats
    hist, t0 = [], time.time()
    for step in range(1, steps + 1):
        policy.eval()  # dropout выключен (как в RLHF): иначе KL и log π шумят от дропаута
        ctx, rb = sampler.batch(bs)
        if horizon == 1:
            logits = next_logits(policy, ctx) / temp
            logp = F.log_softmax(logits, -1)
            with torch.no_grad():
                ref_logp = F.log_softmax(next_logits(ref, ctx), -1)
                acts = torch.multinomial(logp.exp(), G, replacement=True)  # B,G
                r, _ = rm_scores(rm, ctx, rb, acts, ist, reward)
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            pg = -(adv * logp.gather(1, acts)).mean()
            kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
            r_mean = r.mean().item()
        else:
            # траектории: повторяем каждый контекст G раз, генерируем horizon шагов
            c = ctx.repeat_interleave(G, 0); b = rb.repeat_interleave(G, 0)
            tot_logp = 0.0; R = torch.zeros(len(c)); kl = 0.0
            for h in range(horizon):
                logits = next_logits(policy, c) / temp
                logp = F.log_softmax(logits, -1)
                with torch.no_grad():
                    ref_logp = F.log_softmax(next_logits(ref, c), -1)
                    a = torch.multinomial(logp.exp(), 1)[:, 0]
                    r, p = rm_scores(rm, c, b, a[:, None], ist, reward)
                    R += r[:, 0]
                    a_rb = torch.from_numpy(ratio_bucket(p[:, 0, 0].numpy() * 100))
                tot_logp = tot_logp + logp.gather(1, a[:, None])[:, 0]
                kl = kl + (logp.exp() * (logp - ref_logp)).sum(-1).mean() / horizon
                c, b = append_token(c, b, a, a_rb)
            Rg = R.view(bs, G)
            adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-6)).view(-1)
            pg = -(adv * tot_logp).mean()
            r_mean = (R / horizon).mean().item()
        loss = pg + beta * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps or step == 1:
            rec = dict(step=step, train_reward_sampled=round(r_mean, 4), kl=round(float(kl.detach()), 4), sec=round(time.time() - t0, 1))
            if eval_fn is not None and (step % eval_every == 0 or step == steps):
                rec.update(eval_fn(policy))
            hist.append(rec)
            log(rec)
    return policy, hist


@torch.no_grad()
def rollout(policy, rm, ctx, ctx_ratio, horizon=5, greedy=True, exclude_generated=True):
    """Генерация horizon треков подряд. Возвращает (U,H) токены и предсказания RM на каждом шаге."""
    policy.eval(); rm.eval()
    c = torch.from_numpy(ctx); b = torch.from_numpy(np.where(ctx > 0, ratio_bucket(ctx_ratio), 0))
    toks, preds = [], []
    for h in range(horizon):
        lg = next_logits(policy, c)
        if exclude_generated and toks:
            for t in toks:
                lg.scatter_(1, t[:, None], -1e9)
        a = lg.argmax(-1) if greedy else torch.multinomial(F.softmax(lg, -1), 1)[:, 0]
        p = torch.sigmoid(rm.score_h(rm.encode(c, b), c, a[:, None]))[:, 0]
        toks.append(a); preds.append(p)
        c, b = torch.cat([c[:, 1:], a[:, None]], 1), torch.cat([b[:, 1:], torch.from_numpy(ratio_bucket(p[:, 0].numpy() * 100))[:, None]], 1)
    return torch.stack(toks, 1).numpy(), torch.stack(preds, 1).numpy()
