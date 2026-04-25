"""
Multi-phase reward computation for the API Contract Validator.

This module follows the OpenEnv "composable rubric" pattern: each reward
signal is an independent ``RubricComponent`` and the total step reward is
the sum of its components. Independent components have two key benefits:

  1. Reduces reward-hacking risk — an agent that maximises one component
     while degrading another shows up immediately in per-component logs.
  2. Gives RL training a richer gradient — partial progress on one
     component still produces signal even when others are zero.

Phase reward signals
--------------------

Phase 1 — Detection (legacy ``compute_step_reward`` helper kept for
back-compat with the existing tests; new code should use the rubric API):

  - correct violation         +1.0
  - proximity match           +0.3
  - hint requested            -0.5
  - duplicate report          -0.1
  - false positive            -0.3
  - DONE bonus                +0.5 * (correct / total)

Phase 2 — Impact Tracing:

  - correct consumer hit      +0.8 each
  - missed consumer           -0.5 each
  - false-flag consumer       -0.4 each
  - unknown service name      -0.2 each (sub-rule of false-flag)

Phase 3 — Fix & Verify:

  - fix passes ALL consumers  +2.0
  - fix breaks 1+ consumer    -1.0
  - malformed spec patch      -0.5
  - unacceptable strategy     -0.3

Cross-cutting:

  - format compliance         -0.2 for malformed action JSON
  - anti-hacking (spam)       -1.0 if total reports > 3 * planted violations
"""

from dataclasses import dataclass, field
from typing import List

from .fix_validator import FixValidationResult
from .impact_tracer import ImpactTraceResult


# ── Rubric primitives ────────────────────────────────────────────────────


@dataclass
class RubricComponent:
    """A single named reward signal."""

    name: str
    score: float
    explanation: str = ""


@dataclass
class Rubric:
    """Composition of independent reward signals.

    The ``total`` property sums every component. Components are kept
    individually so logs and training analysis can show which signal
    moved across episodes (the key requirement for the "Pipeline 10%"
    judging criterion).
    """

    components: List[RubricComponent] = field(default_factory=list)

    def add(self, name: str, score: float, explanation: str = "") -> "Rubric":
        self.components.append(
            RubricComponent(name=name, score=score, explanation=explanation)
        )
        return self

    @property
    def total(self) -> float:
        return sum(c.score for c in self.components)

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "components": [
                {
                    "name": c.name,
                    "score": round(c.score, 4),
                    "explanation": c.explanation,
                }
                for c in self.components
            ],
        }


# ── Phase 1 — detection (legacy scalar helper kept for backwards compat) ─


@dataclass
class RewardBreakdown:
    """Detailed breakdown of a single Phase 1 step reward.

    Kept for backwards compatibility with the existing Phase 1 tests and
    inference loop. New phases use ``Rubric`` directly.
    """

    reward: float
    is_correct: bool
    is_path_match: bool
    is_duplicate: bool
    is_false_positive: bool
    is_done_signal: bool
    is_hint: bool
    explanation: str


# Phase 1 reward constants
CORRECT_VIOLATION_REWARD = 1.0
PATH_MATCH_REWARD = 0.3
HINT_PENALTY = -0.5
DUPLICATE_PENALTY = -0.1
FALSE_POSITIVE_PENALTY = -0.3
DONE_BONUS_MULTIPLIER = 0.5

# Phase 2 reward constants
CORRECT_CONSUMER_REWARD = 0.8
MISSED_CONSUMER_PENALTY = -0.5
FALSE_FLAG_PENALTY = -0.4
UNKNOWN_SERVICE_PENALTY = -0.2

# Phase 3 reward constants
FIX_PASSES_ALL_REWARD = 2.0
FIX_BREAKS_CONSUMER_PENALTY = -1.0
MALFORMED_PATCH_PENALTY = -0.5
UNACCEPTABLE_STRATEGY_PENALTY = -0.3

# Cross-cutting
MALFORMED_ACTION_PENALTY = -0.2
SPAM_PENALTY = -1.0
SPAM_THRESHOLD_MULTIPLIER = 3


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
    """Compute a Phase 1 detection step reward (legacy scalar API)."""

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
        bonus = round(max(0.01, min(0.99, bonus)), 4)
        return RewardBreakdown(
            reward=bonus,
            is_correct=False,
            is_path_match=False,
            is_duplicate=False,
            is_false_positive=False,
            is_done_signal=True,
            is_hint=False,
            explanation=(
                f"Agent signalled DONE. Completeness "
                f"{correct_so_far}/{total_violations} → bonus {bonus:.2f}"
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
                "Correct field location! The field_path matches a "
                "violation, but the violation_type is wrong. Try again "
                "with the right type."
            ),
        )

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
    """Final Phase 1 normalised score, strictly in (0, 1)."""
    if total_violations == 0:
        return 0.5
    raw = correct_count / total_violations
    return round(max(0.01, min(0.99, raw)), 4)


# ── Phase 2 — impact tracing (Rubric API) ────────────────────────────────


def phase2_trace_rubric(result: ImpactTraceResult) -> Rubric:
    """Build a per-consumer Rubric from a Phase 2 impact-trace result."""
    rubric = Rubric()

    for hit in result.correct_hits:
        rubric.add(
            name=f"consumer_correct:{hit}",
            score=CORRECT_CONSUMER_REWARD,
            explanation=f"Correctly identified affected consumer '{hit}'.",
        )
    for missed in result.missed:
        rubric.add(
            name=f"consumer_missed:{missed}",
            score=MISSED_CONSUMER_PENALTY,
            explanation=f"Missed affected consumer '{missed}'.",
        )
    for flagged in result.false_flags:
        rubric.add(
            name=f"consumer_false_flag:{flagged}",
            score=FALSE_FLAG_PENALTY,
            explanation=(
                f"False-flagged unaffected consumer '{flagged}'."
            ),
        )
    for unknown in result.unknown_services:
        rubric.add(
            name=f"unknown_service:{unknown}",
            score=UNKNOWN_SERVICE_PENALTY,
            explanation=f"'{unknown}' is not a known service in this graph.",
        )
    return rubric


def phase2_episode_score(result: ImpactTraceResult) -> float:
    """Phase 2 final score = F1, clamped to (0.01, 0.99)."""
    return round(max(0.01, min(0.99, result.f1)), 4)


# ── Phase 3 — fix validation (Rubric API) ────────────────────────────────


def phase3_fix_rubric(result: FixValidationResult) -> Rubric:
    """Build a Rubric from a Phase 3 fix-validation result."""
    rubric = Rubric()

    if not result.is_well_formed:
        rubric.add(
            name="malformed_patch",
            score=MALFORMED_PATCH_PENALTY,
            explanation="; ".join(result.notes) or "Malformed spec patch.",
        )
        return rubric

    if not result.is_strategy_acceptable:
        rubric.add(
            name="strategy_unacceptable",
            score=UNACCEPTABLE_STRATEGY_PENALTY,
            explanation=(
                f"Strategy '{result.strategy}' is not appropriate for this "
                f"scenario."
            ),
        )

    if result.all_consumers_pass:
        rubric.add(
            name="fix_passes_all_consumers",
            score=FIX_PASSES_ALL_REWARD,
            explanation=(
                f"Fix using strategy '{result.strategy}' validates against "
                f"all {len(result.consumers_passing)} consumer(s)."
            ),
        )
    else:
        for consumer, reason in result.failure_reasons.items():
            rubric.add(
                name=f"fix_breaks_consumer:{consumer}",
                score=FIX_BREAKS_CONSUMER_PENALTY,
                explanation=(
                    f"Fix breaks consumer '{consumer}': {reason}."
                ),
            )

    return rubric


def phase3_episode_score(result: FixValidationResult) -> float:
    """Phase 3 final score: 0.99 if all consumers pass else proportional."""
    if not result.is_well_formed:
        return 0.01
    total = len(result.consumers_passing) + len(result.consumers_failing)
    if total == 0:
        return 0.01
    raw = len(result.consumers_passing) / total
    return round(max(0.01, min(0.99, raw)), 4)


# ── Cross-cutting signals ────────────────────────────────────────────────


def malformed_action_component() -> RubricComponent:
    """Penalty for an action JSON that fails schema validation."""
    return RubricComponent(
        name="malformed_action",
        score=MALFORMED_ACTION_PENALTY,
        explanation="Action did not match the expected schema.",
    )


def spam_penalty_component(reports: int, planted: int) -> RubricComponent | None:
    """Anti-hacking: agent reporting > 3× planted violations is spamming."""
    if planted <= 0:
        return None
    if reports > SPAM_THRESHOLD_MULTIPLIER * planted:
        return RubricComponent(
            name="spam_penalty",
            score=SPAM_PENALTY,
            explanation=(
                f"Reported {reports} violations against {planted} planted "
                f"— exceeds {SPAM_THRESHOLD_MULTIPLIER}× threshold."
            ),
        )
    return None
