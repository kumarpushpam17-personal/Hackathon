"""
API Contract Validator Environment Implementation.

The agent validates API payloads against OpenAPI specifications by
reporting violations one at a time.  The environment grades each
report against planted ground-truth violations and provides partial
reward signals.
"""

from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ValidatorAction, ValidatorObservation, ValidatorState
except (ImportError, ModuleNotFoundError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import ValidatorAction, ValidatorObservation, ValidatorState

from .rewards import (
    RewardBreakdown,
    compute_episode_score,
    compute_step_reward,
)
from .spec_generator import (
    AVAILABLE_TASKS,
    PlantedViolation,
    TaskScenario,
    generate_scenario_for_task,
)


def _normalise_path(path: str) -> str:
    """Lower-case and strip whitespace for fuzzy path matching."""
    return path.strip().lower().replace(" ", "")


def _find_matching_violation(
    reported_path: str,
    reported_type: str,
    ground_truth: List[PlantedViolation],
) -> Optional[PlantedViolation]:
    """Return the first ground-truth violation that matches the report.

    Matching is intentionally lenient: paths are compared after
    normalisation and the ``violation_type`` is checked with a
    substring match so agents don't need to produce the exact enum
    string.
    """
    norm_path = _normalise_path(reported_path)
    norm_type = reported_type.strip().lower()

    for violation in ground_truth:
        gt_path = _normalise_path(violation.field_path)
        gt_type = violation.violation_type.strip().lower()

        path_match = (norm_path == gt_path) or (
            norm_path in gt_path or gt_path in norm_path
        )
        type_match = (norm_type == gt_type) or (
            norm_type in gt_type or gt_type in norm_type
        )

        if path_match and type_match:
            return violation
    return None


class ValidatorEnvironment(Environment):
    """API Contract Validator — an OpenEnv RL environment.

    At the start of each episode the environment loads a task scenario
    containing an API spec, a payload, and a set of planted violations.
    The agent inspects the spec + payload and reports violations one per
    step.  The episode ends when the agent sends ``DONE`` or exhausts its
    step budget.

    Attributes
    ----------
    SUPPORTS_CONCURRENT_SESSIONS : bool
        ``True`` — each WebSocket connection gets its own environment
        instance with isolated state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = ValidatorState()
        self._scenario: Optional[TaskScenario] = None
        self._matched_paths: Set[str] = set()
        self._reported_violations: List[Dict[str, str]] = []
        self._task_index: int = 0

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ValidatorObservation:
        """Start a new episode.

        If ``task_name`` is passed in *kwargs* it selects a specific task;
        otherwise the environment cycles through all available tasks.
        """
        task_name = kwargs.get("task_name") or AVAILABLE_TASKS[
            self._task_index % len(AVAILABLE_TASKS)
        ]
        self._task_index += 1

        self._scenario = generate_scenario_for_task(task_name)
        self._matched_paths = set()
        self._reported_violations = []

        self._state = ValidatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._scenario.task_name,
            total_violations=len(self._scenario.violations),
            correct_reports=0,
            false_positives=0,
            duplicate_reports=0,
            score=0.0,
        )

        return ValidatorObservation(
            done=False,
            reward=0.0,
            task_name=self._scenario.task_name,
            task_description=self._scenario.task_description,
            api_spec=self._scenario.api_spec,
            payload=self._scenario.payload,
            violations_found=[],
            violations_remaining=len(self._scenario.violations),
            feedback="Episode started. Inspect the spec and payload, then report violations.",
            max_steps=self._scenario.max_steps,
        )

    # ── step ──────────────────────────────────────────────────────────

    def step(
        self,
        action: ValidatorAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ValidatorObservation:
        """Process one violation report from the agent."""
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        is_done_signal = action.field_path.strip().upper() == "DONE"

        # ── Handle DONE signal ────────────────────────────────────────
        if is_done_signal:
            breakdown = compute_step_reward(
                is_correct=False,
                is_duplicate=False,
                is_done_signal=True,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )
            self._state.score = compute_episode_score(
                self._state.correct_reports,
                self._state.total_violations,
            )
            return self._build_observation(
                reward=breakdown.reward,
                done=True,
                feedback=breakdown.explanation,
            )

        # ── Check for duplicate ───────────────────────────────────────
        norm_reported = _normalise_path(action.field_path)
        if norm_reported in self._matched_paths:
            breakdown = compute_step_reward(
                is_correct=False,
                is_duplicate=True,
                is_done_signal=False,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )
            self._state.duplicate_reports += 1
            return self._build_observation(
                reward=breakdown.reward,
                done=False,
                feedback=breakdown.explanation,
            )

        # ── Match against ground truth ────────────────────────────────
        matched = _find_matching_violation(
            action.field_path,
            action.violation_type,
            self._scenario.violations,
        )

        if matched is not None:
            self._matched_paths.add(_normalise_path(matched.field_path))
            self._state.correct_reports += 1
            self._reported_violations.append(
                {
                    "field_path": matched.field_path,
                    "violation_type": matched.violation_type,
                    "description": matched.description,
                }
            )
            breakdown = compute_step_reward(
                is_correct=True,
                is_duplicate=False,
                is_done_signal=False,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )
        else:
            self._state.false_positives += 1
            breakdown = compute_step_reward(
                is_correct=False,
                is_duplicate=False,
                is_done_signal=False,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )

        # ── Check if all violations found ─────────────────────────────
        all_found = self._state.correct_reports >= self._state.total_violations
        steps_exhausted = self._state.step_count >= self._scenario.max_steps
        done = all_found or steps_exhausted

        if done:
            self._state.score = compute_episode_score(
                self._state.correct_reports,
                self._state.total_violations,
            )

        feedback = breakdown.explanation
        if all_found:
            feedback += " All violations found — episode complete!"
        elif steps_exhausted:
            remaining = (
                self._state.total_violations - self._state.correct_reports
            )
            feedback += f" Step limit reached. {remaining} violation(s) missed."

        return self._build_observation(
            reward=breakdown.reward,
            done=done,
            feedback=feedback,
        )

    # ── state ─────────────────────────────────────────────────────────

    @property
    def state(self) -> ValidatorState:
        """Return current internal state (includes ground-truth counts)."""
        return self._state

    # ── helpers ────────────────────────────────────────────────────────

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        feedback: str,
    ) -> ValidatorObservation:
        """Construct an observation from current state."""
        assert self._scenario is not None
        remaining = self._state.total_violations - self._state.correct_reports
        return ValidatorObservation(
            done=done,
            reward=reward,
            task_name=self._scenario.task_name,
            task_description=self._scenario.task_description,
            api_spec=self._scenario.api_spec,
            payload=self._scenario.payload,
            violations_found=list(self._reported_violations),
            violations_remaining=max(remaining, 0),
            feedback=feedback,
            max_steps=self._scenario.max_steps,
        )
