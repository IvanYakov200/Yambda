"""Шаг 5: графики для README (results/fig_*.png)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
BLUE, ORANGE, AQUA, YEL, MAG, GREEN, VIO, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY, INK, MUTED = "#9a9890", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#ecebe7", "axes.axisbelow": True, "figure.dpi": 130})
rl = json.loads((R / "rl.json").read_text())
base = rl["base"]
runs = {k: v for k, v in rl.items() if k != "base"}


def rel(run, key="reward_proxy"):
    rw = runs[run]["cfg"]["reward"]
    b = base[rw]["test"][key]
    return 100 * (runs[run]["test"][key] - b) / abs(b)


# 1) компромисс награда ↔ релевантность
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
bn = base["ratio"]["test"]["ndcg@10"]
betas = sorted([k for k in runs if k.startswith("ratio_b")], key=lambda k: float(k.split("_b")[1]))
for key, ls, lab in [("reward_proxy", "-", "proxy-RM"), ("reward_evaluator", "--", "LightGBM-оценщик")]:
    xs = [0] + [rel(k, key) for k in betas[::-1]]
    ys = [bn] + [runs[k]["test"]["ndcg@10"] for k in betas[::-1]]
    ax.plot(xs, ys, color=BLUE, lw=2 if ls == "-" else 1.3, ls=ls, marker="o", ms=6, label=f"досмотр по {lab}")
for k in betas:
    ax.annotate(f"β={k.split('_b')[1]}", (rel(k), runs[k]["test"]["ndcg@10"]), xytext=(6, 2), textcoords="offset points",
                color=MUTED, fontsize=8)
ax.annotate("base", (0, bn), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=9)
ax.set_xlabel("прирост предсказанного досмотра топ-10, % к базе"); ax.set_ylabel("NDCG@10 (test)")
ax.set_title("A. reward = ratio: сетка β (KL к базе)", loc="left", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="center left")
ax = axes[1]
cols = {"ratio": BLUE, "ratio_lift": ORANGE, "ratio_novel": AQUA, "listen_time": YEL, "like": MAG, "cont": GREEN, "engagement": VIO}
for k, v in runs.items():
    rw = v["cfg"]["reward"]
    if v["cfg"]["beta"] != 0.5:
        continue
    x = rel(k, "reward_evaluator")
    ax.scatter([x], [v["test"]["ndcg@10"]], s=70, color=cols[rw], marker="s" if "_h5" in k else "D", zorder=3)
    ax.annotate(k.replace("_b0.5", "").replace("_h5", " (5 шагов)"), (x, v["test"]["ndcg@10"]), xytext=(6, 3),
                textcoords="offset points", color=MUTED, fontsize=8)
ax.scatter([1], [bn], s=80, color=INK); ax.annotate("base", (1, bn), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("прирост «своей» награды по НЕЗАВИСИМОМУ оценщику, % (лог. шкала)")
ax.set_title("B. разные награды при β = 0.5", loc="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(R / "fig_tradeoff.png")

# 2) поведение
show = ["base"] + [k for k in ["ratio_b0.0", "ratio_b0.5", "ratio_b2.0", "ratio_lift_b0.5", "ratio_novel_b0.5",
                               "listen_time_b0.5", "like_b0.5", "engagement_b0.5"] if k in runs]
metrics = [("repeat_rate", "доля повторов"), ("top1_item_share_users", "юзеров с самым частым треком"),
           ("coverage", "покрытие словаря"), ("mean_length_sec", "средняя длина, с / 300")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
for ax, (m, title) in zip(axes, metrics):
    vals = [(base["ratio"]["test"] if s == "base" else runs[s]["test"])[m] for s in show]
    if m == "mean_length_sec":
        vals = [v / 300 for v in vals]
    ax.barh(range(len(show)), vals, color=[INK if s == "base" else BLUE for s in show], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left"); ax.set_xlim(0, 1.15); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(show)), show); axes[0].invert_yaxis()
fig.suptitle("Как меняется поведение рекомендаций (test, топ-10)", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(R / "fig_behaviour.png")

# 3) переоптимизация по ходу обучения
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for k, c in [("ratio_b0.0", RED), ("ratio_b0.5", BLUE), ("ratio_lift_b0.5", ORANGE)]:
    if k not in runs:
        continue
    lg = [r for r in runs[k]["log"] if "ndcg@10" in r]
    st = [0] + [r["step"] for r in lg]
    b0 = base[runs[k]["cfg"]["reward"]]["val"]
    axes[0].plot(st, [None] + [r["ndcg@10"] for r in lg], color=c, lw=2, marker="o", ms=5, label=k)
    axes[1].plot(st, [b0["reward_proxy"]] + [r["reward_proxy"] for r in lg], color=c, lw=2, marker="o", ms=5, label=f"{k}: proxy")
    axes[1].plot(st, [b0["reward_evaluator"]] + [r["reward_evaluator"] for r in lg], color=c, lw=1.5, ls="--", marker="x", ms=6, label=f"{k}: оценщик")
axes[0].set_title("val NDCG@10 по шагам RL", loc="left", color=INK, fontsize=10); axes[0].set_xlabel("шаг")
axes[1].set_title("награда топ-10: proxy-RM (—) и LightGBM-оценщик (--)", loc="left", color=INK, fontsize=10); axes[1].set_xlabel("шаг")
axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(R / "fig_rl_curves.png")

# 4) многошаговая генерация
if (R / "rollout.json").exists():
    ro = json.loads((R / "rollout.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    pal = [INK, RED, BLUE, ORANGE, AQUA, YEL, VIO, GREEN]
    steps = [1, 2, 3, 4, 5]
    for (n, v), c in zip(ro.items(), pal):
        axes[0].plot(steps, v["autoregressive"]["hit_by_step"], color=c, lw=2, marker="o", ms=5, label=f"{n}: autoregressive")
        if n == "base":
            axes[0].plot(steps, v["teacher_forced_hit_by_step"], color=c, lw=1.5, ls="--", label="base: teacher-forced (оракул)")
            axes[0].plot(steps, v["one_shot"]["hit_by_rank"], color=GRAY, lw=1.5, ls=":", label="base: one-shot топ-5, ранг k")
        axes[1].plot(steps, v["autoregressive"]["pred_ratio_gen_ctx_by_step"], color=c, lw=2, marker="o", ms=5, label=f"{n}: сгенер. контекст")
        axes[1].plot(steps, v["autoregressive"]["pred_ratio_real_ctx_by_step"], color=c, lw=1.2, ls="--")
    axes[0].set_title("доля попаданий в реальные след. 20 треков, по шагу", loc="left", color=INK, fontsize=10)
    axes[1].set_title("предсказанный досмотр: сгенер. контекст (—) vs реальный (--)", loc="left", color=INK, fontsize=10)
    for a in axes:
        a.set_xlabel("шаг генерации"); a.set_xticks(steps)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(R / "fig_rollout.png")
print("ok")
