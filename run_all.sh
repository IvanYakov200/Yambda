#!/usr/bin/env bash
# Полное воспроизведение (~3 ч на 2 CPU). Перед запуском: python prepare_raw.py /path/to/yambda/flat/50m
set -euo pipefail
python -m src.data                      # события, отклики (like/cont/repeat), словарь по train
python scripts/01_base.py 2000          # GPT-2, этап 1 (2000 шагов) + бейзлайны
cp results/base.json results/base_stage1.json
python scripts/01_base.py 2000 resume   # GPT-2, этап 2 (ещё 2000 шагов, warm restart)
python scripts/02_reward.py 1500        # reward model + LightGBM-оценщик + качество на отложенных данных
python scripts/03_rl.py                 # все RL-эксперименты (сетка β, 7 наград, многошаговый)
python scripts/04_rollout.py            # генерация 5 треков подряд
python scripts/05_figures.py
pytest -q tests/
