"""
Reward computation for the API Contract Validator Environment.

Provides partial-progress reward signals rather than binary end-of-episode
scoring.  The agent earns reward for each correctly identified violation
and receives penalties for false positives.
"""

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    """Detailed breakdown of a single-step reward."""

    reward: float
    is_correct: bool
    is_duplicate: bool
    is_false_positive: bool
    is_done_signal: bool
    explanation: str


# ── Per-step reward values ────────────────────────────────────────────────

CORRECT_VIOLATION_REWARD = 1.0
DUPLICATE_PENALTY = -0.1
FALSE_POSITIVE_PENALTY = -0.3
DONE_BONUS_MULTIPLIER = 0.5  # bonus = multiplier * (correct / total)


def compute_step_reward(
    *,
    is_correct: bool,
    is_duplicate: bool,
    is_done_signal: bool,
    correct_so_far: int,
    total_violations: int,
) -> RewardBreakdown:
    """Compute reward for a single agent step.

    Parameters
    ----------
    is_correct:
        Whether the reported violation matches a ground-truth violation.
    is_duplicate:
        Whether the agent already reported this violation.
    is_done_signal:
        Whether the agent submitted ``field_path='DONE'``.
    correct_so_far:
        Number of unique correct violations found *before* this step.
    total_violations:
        Total planted violations in the current scenario.

    Returns
    -------
    RewardBreakdown
        Contains the scalar reward and an explanation string.
    """
    if is_done_signal:
        completeness = correct_so_far / max(total_violations, 1)
        bonus = DONE_BONUS_MULTIPLIER * completeness
        return RewardBreakdown(
            reward=round(bonus, 4),
            is_correct=False,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=True,
            explanation=(
                f"Agent signalled DONE. "
                f"Completeness {correct_so_far}/{total_violations} "
                f"→ bonus {bonus:.2f}"
            ),
        )

    if is_duplicate:
        return RewardBreakdown(
            reward=DUPLICATE_PENALTY,
            is_correct=False,
            is_duplicate=True,
            is_false_positive=False,
            is_done_signal=False,
            explanation="Duplicate violation report — already submitted.",
        )

    if is_correct:
        return RewardBreakdown(
            reward=CORRECT_VIOLATION_REWARD,
            is_correct=True,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=False,
            explanation="Correct! Violation matches ground truth.",
        )

    # False positive
    return RewardBreakdown(
        reward=FALSE_POSITIVE_PENALTY,
        is_correct=False,
        is_duplicate=False,
        is_false_positive=True,
        is_done_signal=False,
        explanation="False positive — no matching violation in ground truth.",
    )


def compute_episode_score(correct_count: int, total_violations: int) -> float:
    """Compute the final normalised score for the episode.

    Returns a float in ``[0.0, 1.0]``.
    """
    if total_violations == 0:
        return 1.0
    return round(correct_count / total_violations, 4)
