"""
Reward computation for the API Contract Validator Environment.

Provides partial-progress reward signals rather than binary end-of-episode
scoring.  The reward function has several interesting properties:

  - Correct violation found          → +1.0  (primary incentive)
  - Path-only match (wrong type)     → +0.3  (proximity signal — learn location first)
  - HINT requested                   → -0.5  (expensive but informative)
  - Duplicate report                 → -0.1  (light penalty, track what you've found)
  - False positive                   → -0.3  (penalise guessing)
  - DONE signal                      → +0.5 × (found/total)  (completeness bonus)

The proximity reward creates a richer gradient: agents learn to locate the
right field first, then refine their violation classification.
"""

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    """Detailed breakdown of a single-step reward."""

    reward: float
    is_correct: bool
    is_path_match: bool
    is_duplicate: bool
    is_false_positive: bool
    is_done_signal: bool
    is_hint: bool
    explanation: str


# ── Per-step reward values ────────────────────────────────────────────────

CORRECT_VIOLATION_REWARD = 1.0
PATH_MATCH_REWARD = 0.3          # right field, wrong violation_type
HINT_PENALTY = -0.5              # cost of requesting a location hint
DUPLICATE_PENALTY = -0.1
FALSE_POSITIVE_PENALTY = -0.3
DONE_BONUS_MULTIPLIER = 0.5      # bonus = multiplier * (correct / total)


def compute_step_reward(
    *,
    is_correct: bool,
    is_path_match: bool = False,
    is_duplicate: bool,
    is_done_signal: bool,
    is_hint: bool = False,
    correct_so_far: int,
    total_violations: int,
) -> RewardBreakdown:
    """Compute reward for a single agent step.

    Parameters
    ----------
    is_correct:
        Whether the report fully matches a ground-truth violation (path + type).
    is_path_match:
        Whether the field_path matches a violation but violation_type is wrong.
    is_duplicate:
        Whether the agent already reported this violation.
    is_done_signal:
        Whether the agent submitted ``field_path='DONE'``.
    is_hint:
        Whether the agent submitted ``field_path='HINT'``.
    correct_so_far:
        Number of unique correct violations found before this step.
    total_violations:
        Total planted violations in the current scenario.

    Returns
    -------
    RewardBreakdown
        Contains the scalar reward and a human-readable explanation.
    """
    if is_hint:
        return RewardBreakdown(
            reward=HINT_PENALTY,
            is_correct=False,
            is_path_match=False,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=False,
            is_hint=True,
            explanation="Hint requested. -0.5 reward.",
        )

    if is_done_signal:
        completeness = correct_so_far / max(total_violations, 1)
        bonus = DONE_BONUS_MULTIPLIER * completeness
        return RewardBreakdown(
            reward=round(bonus, 4),
            is_correct=False,
            is_path_match=False,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=True,
            is_hint=False,
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
            is_path_match=False,
            is_duplicate=True,
            is_false_positive=False,
            is_done_signal=False,
            is_hint=False,
            explanation="Duplicate violation report — already submitted.",
        )

    if is_correct:
        return RewardBreakdown(
            reward=CORRECT_VIOLATION_REWARD,
            is_correct=True,
            is_path_match=False,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=False,
            is_hint=False,
            explanation="Correct! Violation matches ground truth.",
        )

    if is_path_match:
        return RewardBreakdown(
            reward=PATH_MATCH_REWARD,
            is_correct=False,
            is_path_match=True,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=False,
            is_hint=False,
            explanation=(
                "Correct field location! The field_path matches a violation, "
                "but the violation_type is wrong. Try again with the right type."
            ),
        )

    # False positive
    return RewardBreakdown(
        reward=FALSE_POSITIVE_PENALTY,
        is_correct=False,
        is_path_match=False,
        is_duplicate=False,
        is_false_positive=True,
        is_done_signal=False,
        is_hint=False,
        explanation="False positive — no matching violation in ground truth.",
    )


def compute_episode_score(correct_count: int, total_violations: int) -> float:
    """Compute the final normalised score for the episode.

    Returns a float strictly in ``(0.0, 1.0)`` — endpoints excluded — as
    required by the OpenEnv evaluation pipeline.
    """
    if total_violations == 0:
        return 0.5
    raw = correct_count / total_violations
    return round(max(0.0001, min(0.9999, raw)), 4)
