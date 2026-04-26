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
BASELINE_7B = REPO_ROOT / "baseline_7b_scores.json"
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
    """Three-bar comparison: 72B baseline, 7B baseline, 7B + LoRA trained."""
    baseline72 = _load_scores(BASELINE)
    baseline7 = _load_scores(BASELINE_7B) if BASELINE_7B.exists() else None
    trained = _load_scores(TRAINED)

    tasks = list(trained.keys())  # preserve task order from trained_scores.json
    b72 = [baseline72.get(t, 0.0) for t in tasks]
    b7 = [baseline7.get(t, 0.0) for t in tasks] if baseline7 else None
    tr = [trained.get(t, 0.0) for t in tasks]

    x = np.arange(len(tasks))
    n_bars = 3 if baseline7 else 2
    width = 0.8 / n_bars

    plt.figure(figsize=(12, 6))
    if baseline7:
        plt.bar(x - width, b72, width, label="Qwen2.5-72B (untrained)",
                color="#6b7280")
        plt.bar(x,        b7,  width, label="Qwen2.5-7B  (untrained, same base)",
                color="#9ca3af")
        plt.bar(x + width, tr, width, label="Qwen2.5-7B + LoRA (GRPO-trained)",
                color="#16a34a")
    else:
        plt.bar(x - width / 2, b72, width, label="Baseline (Qwen2.5-72B)",
                color="#9ca3af")
        plt.bar(x + width / 2, tr, width, label="Trained 7B + LoRA",
                color="#16a34a")
    plt.xticks(x, tasks, rotation=30, ha="right", fontsize=9)
    plt.ylabel("Episode score (0–1)")
    plt.ylim(0, 1.0)
    plt.title(
        "Per-task score: untrained baselines vs GRPO-trained adapter\n"
        "GRPO unlocks `detect_breaking_changes` (0.01 → 0.67) — neither baseline can do it"
    )
    plt.legend(loc="upper right", fontsize=9)
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
