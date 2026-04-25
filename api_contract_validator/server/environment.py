"""
API Contract Validator Environment Implementation.

The agent validates API payloads against OpenAPI specifications by
reporting violations one at a time.  The environment grades each
report against planted ground-truth violations and provides partial
reward signals.

Special field_path values:
  'DONE' — end the episode and collect the completeness bonus
  'HINT' — receive a location hint (costs -0.5 reward)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

logger = logging.getLogger(__name__)

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
    """Return the first ground-truth violation that matches both path and type.

    Matching is intentionally lenient: paths are compared after normalisation
    and violation_type uses substring matching.
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


def _find_path_only_match(
    reported_path: str,
    ground_truth: List[PlantedViolation],
    already_matched: Set[str],
    already_proximity: Set[str],
) -> Optional[PlantedViolation]:
    """Return a violation whose path matches but has not yet been fully matched.

    Used for the proximity reward: agent found the right field but wrong type.
    Ignores violations that have already been correctly reported OR already
    received a proximity reward (to prevent reward farming).
    """
    norm_path = _normalise_path(reported_path)

    for violation in ground_truth:
        gt_path = _normalise_path(violation.field_path)
        if gt_path in already_matched or gt_path in already_proximity:
            continue
        path_match = (norm_path == gt_path) or (
            norm_path in gt_path or gt_path in norm_path
        )
        if path_match:
            return violation
    return None


def _hint_section(field_path: str) -> str:
    """Extract the top-level section name from a field path.

    Examples:
        'customer.email'           → 'customer'
        'items[1].quantity'        → 'items'
        'billing.tax_rate'         → 'billing'
        'due_date'                 → 'due_date'
        'POST /products.price'     → 'POST /products'
    """
    path = field_path.strip()
    # Handle breaking-change paths like "POST /products.price"
    if path.startswith(("GET ", "POST ", "PUT ", "PATCH ", "DELETE ")):
        dot_idx = path.find(".")
        return path[:dot_idx] if dot_idx != -1 else path
    # Standard paths: split on first dot or bracket
    for i, ch in enumerate(path):
        if ch in (".", "["):
            return path[:i]
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ValidatorEnvironment(Environment):
    """API Contract Validator — an OpenEnv RL environment.

    At the start of each episode the environment loads a task scenario
    containing an API spec, a payload, and a set of planted violations.
    The agent inspects the spec + payload and reports violations one per
    step.  The episode ends when the agent sends ``DONE`` or exhausts its
    step budget.

    Special actions:
      field_path='DONE'  — end episode, collect completeness bonus
      field_path='HINT'  — receive a location hint, pay -0.5 reward

    Attributes
    ----------
    SUPPORTS_CONCURRENT_SESSIONS : bool
        True — each WebSocket connection gets its own isolated instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = ValidatorState()
        self._scenario: Optional[TaskScenario] = None
        self._matched_paths: Set[str] = set()
        self._proximity_paths: Set[str] = set()
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

        self._scenario = generate_scenario_for_task(task_name, seed=seed)
        self._matched_paths = set()
        self._proximity_paths = set()
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

        logger.info(json.dumps({
            "event": "episode_start",
            "episode_id": self._state.episode_id,
            "task": self._state.task_name,
            "total_violations": self._state.total_violations,
            "max_steps": self._scenario.max_steps,
            "ts": _now(),
        }))

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
        action: ValidatorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ValidatorObservation:
        """Process one violation report from the agent."""
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        signal = action.field_path.strip().upper()

        # ── HINT request ──────────────────────────────────────────────
        if signal == "HINT":
            remaining = [
                v for v in self._scenario.violations
                if _normalise_path(v.field_path) not in self._matched_paths
            ]
            if remaining:
                section = _hint_section(remaining[0].field_path)
                hint_msg = (
                    f"Hint: An undetected violation is in the '{section}' section. "
                    f"(-0.5 reward)"
                )
            else:
                hint_msg = "All violations have already been found. Submit DONE."

            breakdown = compute_step_reward(
                is_correct=False,
                is_path_match=False,
                is_duplicate=False,
                is_done_signal=False,
                is_hint=True,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )
            return self._build_observation(
                reward=breakdown.reward,
                done=False,
                feedback=hint_msg,
            )

        # ── DONE signal ───────────────────────────────────────────────
        if signal == "DONE":
            breakdown = compute_step_reward(
                is_correct=False,
                is_path_match=False,
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

        # ── Duplicate check ───────────────────────────────────────────
        norm_reported = _normalise_path(action.field_path)
        if norm_reported in self._matched_paths:
            breakdown = compute_step_reward(
                is_correct=False,
                is_path_match=False,
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

        # ── Full match (path + type) ──────────────────────────────────
        matched = _find_matching_violation(
            action.field_path,
            action.violation_type,
            self._scenario.violations,
        )

        if matched is not None:
            gt_path = _normalise_path(matched.field_path)
            self._matched_paths.add(gt_path)
            self._proximity_paths.discard(gt_path)
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
                is_path_match=False,
                is_duplicate=False,
                is_done_signal=False,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )

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
                remaining = self._state.total_violations - self._state.correct_reports
                feedback += f" Step limit reached. {remaining} violation(s) missed."

            return self._build_observation(
                reward=breakdown.reward,
                done=done,
                feedback=feedback,
            )

        # ── Proximity match (right path, wrong type) ──────────────────
        path_match = _find_path_only_match(
            action.field_path,
            self._scenario.violations,
            self._matched_paths,
            self._proximity_paths,
        )

        if path_match is not None:
            self._proximity_paths.add(_normalise_path(path_match.field_path))
            breakdown = compute_step_reward(
                is_correct=False,
                is_path_match=True,
                is_duplicate=False,
                is_done_signal=False,
                correct_so_far=self._state.correct_reports,
                total_violations=self._state.total_violations,
            )
            steps_exhausted = self._state.step_count >= self._scenario.max_steps
            if steps_exhausted:
                self._state.score = compute_episode_score(
                    self._state.correct_reports,
                    self._state.total_violations,
                )
            return self._build_observation(
                reward=breakdown.reward,
                done=steps_exhausted,
                feedback=breakdown.explanation,
            )

        # ── False positive ────────────────────────────────────────────
        self._state.false_positives += 1
        breakdown = compute_step_reward(
            is_correct=False,
            is_path_match=False,
            is_duplicate=False,
            is_done_signal=False,
            correct_so_far=self._state.correct_reports,
            total_violations=self._state.total_violations,
        )

        steps_exhausted = self._state.step_count >= self._scenario.max_steps
        done = steps_exhausted

        if done:
            self._state.score = compute_episode_score(
                self._state.correct_reports,
                self._state.total_violations,
            )

        feedback = breakdown.explanation
        if steps_exhausted:
            remaining = self._state.total_violations - self._state.correct_reports
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

        logger.debug(json.dumps({
            "event": "step",
            "episode_id": self._state.episode_id,
            "task": self._state.task_name,
            "step": self._state.step_count,
            "reward": round(reward, 4),
            "correct_so_far": self._state.correct_reports,
            "total_violations": self._state.total_violations,
            "done": done,
            "ts": _now(),
        }))

        if done:
            logger.info(json.dumps({
                "event": "episode_end",
                "episode_id": self._state.episode_id,
                "task": self._state.task_name,
                "score": round(self._state.score, 4),
                "steps": self._state.step_count,
                "correct": self._state.correct_reports,
                "total": self._state.total_violations,
                "false_positives": self._state.false_positives,
                "duplicates": self._state.duplicate_reports,
                "ts": _now(),
            }))

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
