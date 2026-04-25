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
    from ..models import (
        ACTION_PROPOSE_FIX,
        ACTION_REPORT_VIOLATION,
        ACTION_TRACE_IMPACT,
        ACTION_VALIDATE_FIX,
        ValidatorAction,
        ValidatorObservation,
        ValidatorState,
    )
except (ImportError, ModuleNotFoundError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import (
        ACTION_PROPOSE_FIX,
        ACTION_REPORT_VIOLATION,
        ACTION_TRACE_IMPACT,
        ACTION_VALIDATE_FIX,
        ValidatorAction,
        ValidatorObservation,
        ValidatorState,
    )

from .fix_validator import validate_fix
from .impact_tracer import trace_impact
from .rewards import (
    RewardBreakdown,
    compute_episode_score,
    compute_step_reward,
    phase2_episode_score,
    phase2_trace_rubric,
    phase3_episode_score,
    phase3_fix_rubric,
)
from .service_graph import (
    CASCADE_SCENARIO_IDS,
    CascadeScenario,
    consumer_specs_for_fix,
    get_cascade_scenario,
    public_observation,
)
from .spec_generator import (
    AVAILABLE_TASKS,
    PlantedViolation,
    TaskScenario,
    generate_scenario_for_task,
)


# ── Phase 2 / Phase 3 task names ─────────────────────────────────────────

PHASE2_TASKS: Set[str] = {"trace_downstream_blast_radius"}
PHASE3_TASKS: Set[str] = {"propose_backward_compat_fix"}
CASCADE_TASKS: Set[str] = {"multi_service_cascade_fix"}
ALL_TASKS: List[str] = (
    AVAILABLE_TASKS
    + sorted(PHASE2_TASKS)
    + sorted(PHASE3_TASKS)
    + sorted(CASCADE_TASKS)
)

PHASE_DETECTION = "detection"
PHASE_TRACING = "tracing"
PHASE_FIX = "fix_proposal"


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

        # Phase 2 / Phase 3 episode state
        self._cascade: Optional[CascadeScenario] = None
        self._phase: str = PHASE_DETECTION
        self._consumers_traced: Set[str] = set()
        self._last_fix_results: Dict[str, Any] = {}
        self._cascade_max_steps: int = 0

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ValidatorObservation:
        """Start a new episode.

        Dispatches to the right setup path based on ``task_name``:

          * Phase 1 detection tasks (default) → spec + payload
          * Phase 2 trace task → service graph + breaking change
          * Phase 3 fix task → detected violation + consumer specs
          * Cascade task → all three phases in one episode
        """
        task_name = kwargs.get("task_name") or AVAILABLE_TASKS[
            self._task_index % len(AVAILABLE_TASKS)
        ]
        self._task_index += 1

        # Reset shared episode bookkeeping
        self._matched_paths = set()
        self._proximity_paths = set()
        self._reported_violations = []
        self._consumers_traced = set()
        self._last_fix_results = {}
        self._cascade = None
        self._scenario = None

        if task_name in PHASE2_TASKS:
            return self._reset_phase2(task_name, seed, episode_id)
        if task_name in PHASE3_TASKS:
            return self._reset_phase3(task_name, seed, episode_id)
        if task_name in CASCADE_TASKS:
            return self._reset_cascade(task_name, seed, episode_id)
        return self._reset_phase1(task_name, seed, episode_id)

    # ── Phase 1 reset (unchanged behaviour) ──────────────────────────

    def _reset_phase1(
        self,
        task_name: str,
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> ValidatorObservation:
        self._phase = PHASE_DETECTION
        self._scenario = generate_scenario_for_task(task_name, seed=seed)

        self._state = ValidatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._scenario.task_name,
            phase=PHASE_DETECTION,
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
            "phase": self._phase,
            "total_violations": self._state.total_violations,
            "max_steps": self._scenario.max_steps,
            "ts": _now(),
        }))

        return ValidatorObservation(
            done=False,
            reward=0.0,
            task_name=self._scenario.task_name,
            task_description=self._scenario.task_description,
            phase=PHASE_DETECTION,
            api_spec=self._scenario.api_spec,
            payload=self._scenario.payload,
            violations_found=[],
            violations_remaining=len(self._scenario.violations),
            feedback="Episode started. Inspect the spec and payload, then report violations.",
            max_steps=self._scenario.max_steps,
        )

    # ── Phase 2 reset — impact tracing ───────────────────────────────

    def _reset_phase2(
        self,
        task_name: str,
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> ValidatorObservation:
        self._phase = PHASE_TRACING
        self._cascade = get_cascade_scenario(seed=seed)
        max_steps = 20
        self._cascade_max_steps = max_steps

        self._state = ValidatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            phase=PHASE_TRACING,
            total_consumers=len(self._cascade.consumers),
            consumers_correctly_traced=0,
            consumers_missed=len(self._cascade.ground_truth_affected),
            consumers_false_flagged=0,
            score=0.01,
        )

        logger.info(json.dumps({
            "event": "episode_start",
            "episode_id": self._state.episode_id,
            "task": task_name,
            "phase": self._phase,
            "scenario": self._cascade.scenario_id,
            "consumers": [c.name for c in self._cascade.consumers],
            "max_steps": max_steps,
            "ts": _now(),
        }))

        return ValidatorObservation(
            done=False,
            reward=0.0,
            task_name=task_name,
            task_description=(
                f"{self._cascade.description} Submit a single trace_impact "
                f"action listing every downstream service whose contract is "
                f"broken by the change."
            ),
            phase=PHASE_TRACING,
            service_graph=public_observation(self._cascade),
            consumers_traced=[],
            total_consumers=len(self._cascade.consumers),
            feedback=(
                "Phase 2 — Impact Tracing. Inspect the service graph and "
                "submit action_type='trace_impact' with affected_services."
            ),
            max_steps=max_steps,
        )

    # ── Phase 3 reset — fix proposal ─────────────────────────────────

    def _reset_phase3(
        self,
        task_name: str,
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> ValidatorObservation:
        self._phase = PHASE_FIX
        self._cascade = get_cascade_scenario(seed=seed)
        max_steps = 25
        self._cascade_max_steps = max_steps

        self._state = ValidatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            phase=PHASE_FIX,
            total_consumers=len(self._cascade.consumers),
            fix_attempts=0,
            fix_validated=False,
            score=0.01,
        )

        logger.info(json.dumps({
            "event": "episode_start",
            "episode_id": self._state.episode_id,
            "task": task_name,
            "phase": self._phase,
            "scenario": self._cascade.scenario_id,
            "acceptable_strategies": self._cascade.acceptable_fix_strategies,
            "max_steps": max_steps,
            "ts": _now(),
        }))

        return ValidatorObservation(
            done=False,
            reward=0.0,
            task_name=task_name,
            task_description=(
                f"{self._cascade.description} Submit propose_fix with a "
                f"fix_strategy and spec_patch that keeps every consumer "
                f"working."
            ),
            phase=PHASE_FIX,
            detected_violation=self._cascade.violation,
            consumer_specs=consumer_specs_for_fix(self._cascade),
            service_graph=public_observation(self._cascade),
            feedback=(
                "Phase 3 — Fix & Verify. Submit action_type='propose_fix' "
                f"with fix_strategy in "
                f"{self._cascade.acceptable_fix_strategies} and a "
                "spec_patch object."
            ),
            max_steps=max_steps,
        )

    # ── Cascade reset — full workflow ────────────────────────────────

    def _reset_cascade(
        self,
        task_name: str,
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> ValidatorObservation:
        """Full detect → trace → fix workflow in one episode.

        Starts in tracing phase since the violation is given to the agent
        upfront (cascade scenarios already include the breaking change).
        Phase 3 begins after the agent submits a successful trace_impact.
        """
        self._phase = PHASE_TRACING
        self._cascade = get_cascade_scenario(seed=seed)
        max_steps = 40
        self._cascade_max_steps = max_steps

        self._state = ValidatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            phase=PHASE_TRACING,
            total_consumers=len(self._cascade.consumers),
            consumers_correctly_traced=0,
            consumers_missed=len(self._cascade.ground_truth_affected),
            fix_attempts=0,
            fix_validated=False,
            score=0.01,
        )

        logger.info(json.dumps({
            "event": "episode_start",
            "episode_id": self._state.episode_id,
            "task": task_name,
            "phase": self._phase,
            "scenario": self._cascade.scenario_id,
            "max_steps": max_steps,
            "ts": _now(),
        }))

        return ValidatorObservation(
            done=False,
            reward=0.0,
            task_name=task_name,
            task_description=(
                "Multi-phase cascade: first trace_impact to identify "
                "affected consumers, then propose_fix with a backward-"
                "compatible migration. Episode ends when the fix passes "
                "all consumers or the step budget runs out."
            ),
            phase=PHASE_TRACING,
            service_graph=public_observation(self._cascade),
            detected_violation=self._cascade.violation,
            consumer_specs=consumer_specs_for_fix(self._cascade),
            total_consumers=len(self._cascade.consumers),
            feedback=(
                "Cascade episode started in Phase 2. Submit trace_impact "
                "first, then move on to propose_fix."
            ),
            max_steps=max_steps,
        )

    # ── step ──────────────────────────────────────────────────────────

    def step(
        self,
        action: ValidatorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ValidatorObservation:
        """Dispatch one agent action to the matching phase handler."""
        if self._scenario is None and self._cascade is None:
            raise RuntimeError("Call reset() before step().")

        # Phase 2 — single-step trace
        if (
            action.action_type == ACTION_TRACE_IMPACT
            and self._cascade is not None
        ):
            return self._step_trace_impact(action)

        # Phase 3 — fix proposal / validation
        if (
            action.action_type in (ACTION_PROPOSE_FIX, ACTION_VALIDATE_FIX)
            and self._cascade is not None
        ):
            return self._step_fix(action)

        # Default — Phase 1 detection (handles report_violation, DONE, HINT)
        if self._scenario is None:
            return self._build_observation_phase2(
                reward=-0.5,
                done=False,
                feedback=(
                    f"Action type '{action.action_type}' is not valid in "
                    f"phase '{self._phase}'."
                ),
            )

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

    # ── Phase 2 step — trace_impact ──────────────────────────────────

    def _step_trace_impact(
        self, action: ValidatorAction
    ) -> ValidatorObservation:
        """Grade a single trace_impact action against ground truth."""
        assert self._cascade is not None

        self._state.step_count += 1

        result = trace_impact(self._cascade, action.affected_services)
        rubric = phase2_trace_rubric(result)
        reward = rubric.total

        self._consumers_traced.update(result.correct_hits)
        self._state.consumers_correctly_traced = len(result.correct_hits)
        self._state.consumers_missed = len(result.missed)
        self._state.consumers_false_flagged = len(result.false_flags)

        # In a pure Phase-2 task, one trace ends the episode.
        # In cascade, a fully-correct trace transitions to Phase 3.
        is_cascade = self._state.task_name in CASCADE_TASKS
        all_correct = not result.missed and not result.false_flags
        steps_exhausted = self._state.step_count >= self._cascade_max_steps

        if is_cascade and all_correct and not steps_exhausted:
            self._phase = PHASE_FIX
            self._state.phase = PHASE_FIX
            done = False
            feedback = (
                "All consumers correctly traced. Phase 3 unlocked — submit "
                "propose_fix with a backward-compatible spec_patch."
            )
        else:
            done = True
            self._state.score = phase2_episode_score(result)
            feedback = (
                f"Phase 2 result — precision {result.precision:.2f}, "
                f"recall {result.recall:.2f}, f1 {result.f1:.2f}. "
                f"correct={result.correct_hits} | missed={result.missed} | "
                f"false-flagged={result.false_flags}"
            )

        if steps_exhausted and not done:
            done = True
            self._state.score = phase2_episode_score(result)
            feedback += " Step budget exhausted."

        return self._build_observation_phase2(
            reward=round(reward, 4),
            done=done,
            feedback=feedback,
            rubric_components=rubric.to_dict(),
        )

    # ── Phase 3 step — propose_fix / validate_fix ────────────────────

    def _step_fix(self, action: ValidatorAction) -> ValidatorObservation:
        """Grade a fix proposal against every consumer in the scenario."""
        assert self._cascade is not None

        self._state.step_count += 1
        self._state.fix_attempts += 1

        fix_result = validate_fix(
            self._cascade, action.fix_strategy, action.spec_patch
        )
        rubric = phase3_fix_rubric(fix_result)
        reward = rubric.total

        self._state.fix_validated = fix_result.all_consumers_pass
        self._state.fix_breaks_consumers = len(fix_result.consumers_failing)
        self._last_fix_results = {
            "strategy": fix_result.strategy,
            "consumers_passing": fix_result.consumers_passing,
            "consumers_failing": fix_result.consumers_failing,
            "failure_reasons": fix_result.failure_reasons,
            "notes": fix_result.notes,
        }

        steps_exhausted = self._state.step_count >= self._cascade_max_steps
        done = fix_result.all_consumers_pass or steps_exhausted

        if done:
            self._state.score = phase3_episode_score(fix_result)

        if fix_result.all_consumers_pass:
            feedback = (
                f"Fix accepted — strategy '{fix_result.strategy}' "
                f"validates against all "
                f"{len(fix_result.consumers_passing)} consumer(s). "
                f"Episode complete."
            )
        elif not fix_result.is_well_formed:
            feedback = (
                f"Malformed fix proposal: "
                f"{'; '.join(fix_result.notes) or 'see field requirements'}."
            )
        else:
            feedback = (
                f"Fix breaks {len(fix_result.consumers_failing)} consumer(s): "
                f"{fix_result.consumers_failing}. "
                f"Refine the spec_patch and try again."
            )
            if steps_exhausted:
                feedback += " Step budget exhausted."

        return self._build_observation_phase3(
            reward=round(reward, 4),
            done=done,
            feedback=feedback,
            fix_validation_results=self._last_fix_results,
            rubric_components=rubric.to_dict(),
        )

    # ── Phase 2 observation builder ──────────────────────────────────

    def _build_observation_phase2(
        self,
        *,
        reward: float,
        done: bool,
        feedback: str,
        rubric_components: Optional[Dict[str, Any]] = None,
    ) -> ValidatorObservation:
        assert self._cascade is not None
        logger.debug(json.dumps({
            "event": "step",
            "episode_id": self._state.episode_id,
            "task": self._state.task_name,
            "phase": self._phase,
            "step": self._state.step_count,
            "reward": reward,
            "done": done,
            "rubric": rubric_components,
            "ts": _now(),
        }))
        if done:
            logger.info(json.dumps({
                "event": "episode_end",
                "episode_id": self._state.episode_id,
                "task": self._state.task_name,
                "phase": self._phase,
                "score": round(self._state.score, 4),
                "steps": self._state.step_count,
                "consumers_correctly_traced": self._state.consumers_correctly_traced,
                "consumers_missed": self._state.consumers_missed,
                "consumers_false_flagged": self._state.consumers_false_flagged,
                "ts": _now(),
            }))
        return ValidatorObservation(
            done=done,
            reward=reward,
            task_name=self._state.task_name,
            task_description="",
            phase=self._phase,
            service_graph=public_observation(self._cascade),
            consumers_traced=sorted(self._consumers_traced),
            total_consumers=len(self._cascade.consumers),
            detected_violation=self._cascade.violation,
            consumer_specs=consumer_specs_for_fix(self._cascade),
            feedback=feedback,
            max_steps=self._cascade_max_steps,
        )

    # ── Phase 3 observation builder ──────────────────────────────────

    def _build_observation_phase3(
        self,
        *,
        reward: float,
        done: bool,
        feedback: str,
        fix_validation_results: Dict[str, Any],
        rubric_components: Optional[Dict[str, Any]] = None,
    ) -> ValidatorObservation:
        assert self._cascade is not None
        logger.debug(json.dumps({
            "event": "step",
            "episode_id": self._state.episode_id,
            "task": self._state.task_name,
            "phase": self._phase,
            "step": self._state.step_count,
            "reward": reward,
            "done": done,
            "fix_validation": fix_validation_results,
            "rubric": rubric_components,
            "ts": _now(),
        }))
        if done:
            logger.info(json.dumps({
                "event": "episode_end",
                "episode_id": self._state.episode_id,
                "task": self._state.task_name,
                "phase": self._phase,
                "score": round(self._state.score, 4),
                "steps": self._state.step_count,
                "fix_validated": self._state.fix_validated,
                "fix_attempts": self._state.fix_attempts,
                "ts": _now(),
            }))
        return ValidatorObservation(
            done=done,
            reward=reward,
            task_name=self._state.task_name,
            task_description="",
            phase=self._phase,
            service_graph=public_observation(self._cascade),
            consumers_traced=sorted(self._consumers_traced),
            total_consumers=len(self._cascade.consumers),
            detected_violation=self._cascade.violation,
            consumer_specs=consumer_specs_for_fix(self._cascade),
            fix_validation_results=fix_validation_results,
            feedback=feedback,
            max_steps=self._cascade_max_steps,
        )

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
            phase=PHASE_DETECTION,
            api_spec=self._scenario.api_spec,
            payload=self._scenario.payload,
            violations_found=list(self._reported_violations),
            violations_remaining=max(remaining, 0),
            feedback=feedback,
            max_steps=self._scenario.max_steps,
        )
