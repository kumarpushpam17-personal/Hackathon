"""
Plot generators for the README results section.

Produces two PNG files in ``results/``:

    reward_curve.png   — per-step training reward (from training_state.json)
    before_after.png   — per-task baseline vs trained score bar chart

Run after both ``baseline_scores.json`` and ``trained_scores.json``
exist::

    python training/plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
RESULTS = ROOT / "results"
BASELINE = REPO_ROOT / "baseline_scores.json"
TRAINED = REPO_ROOT / "trained_scores.json"
TRAIN_STATE = RESULTS / "training_state.json"


def _load_scores(path: Path) -> Dict[str, float]:
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())["scores"]


def plot_reward_curve() -> None:
    if not TRAIN_STATE.exists():
        print(f"[WARN] {TRAIN_STATE} not found — skipping reward_curve.png")
        return
    history = json.loads(TRAIN_STATE.read_text())
    rows = [h for h in history if "reward" in h and "step" in h]
    if not rows:
        print("[WARN] no reward entries in training_state.json")
        return

    steps = [h["step"] for h in rows]
    rewards = [h["reward"] for h in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, linewidth=2, color="#2563eb",
             label="GRPO training reward")
    if "loss" in rows[0]:
        plt.twinx().plot(
            steps, [h.get("loss", 0) for h in rows],
            linewidth=1, color="#9ca3af", linestyle="--", label="loss",
        )
    plt.xlabel("Training step")
    plt.ylabel("Mean episode reward")
    plt.title("Enterprise Contract Guardian — GRPO Training")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = RESULTS / "reward_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] wrote {out}")


def plot_before_after() -> None:
    baseline = _load_scores(BASELINE)
    trained = _load_scores(TRAINED)

    tasks = sorted(set(baseline) | set(trained))
    base_vals = [baseline.get(t, 0.0) for t in tasks]
    train_vals = [trained.get(t, 0.0) for t in tasks]

    x = np.arange(len(tasks))
    width = 0.4

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width / 2, base_vals, width, label="Baseline (untrained)",
            color="#9ca3af")
    plt.bar(x + width / 2, train_vals, width, label="GRPO-trained",
            color="#16a34a")
    plt.xticks(x, tasks, rotation=30, ha="right", fontsize=9)
    plt.ylabel("Episode score (0–1)")
    plt.ylim(0, 1.0)
    plt.title("Per-task score: baseline vs trained agent")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = RESULTS / "before_after.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] wrote {out}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_reward_curve()
    plot_before_after()


if __name__ == "__main__":
    main()
