"""
API Contract Validator Environment Client.

Maintains a persistent WebSocket connection to the environment server.
Each client instance has its own isolated session.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import ValidatorAction, ValidatorObservation, ValidatorState
except (ImportError, ModuleNotFoundError):
    from models import ValidatorAction, ValidatorObservation, ValidatorState


class ValidatorEnv(
    EnvClient[ValidatorAction, ValidatorObservation, ValidatorState]
):
    """Client for the API Contract Validator Environment.

    Example::

        with ValidatorEnv(base_url="http://localhost:7860").sync() as env:
            result = env.reset()
            print(result.observation.task_name)
            result = env.step(ValidatorAction(
                field_path="email",
                violation_type="missing_required",
                description="Required field 'email' is missing.",
            ))
            print(result.observation.feedback)

    Example with Docker::

        env = ValidatorEnv.from_docker_image("api-contract-validator:latest")
        result = env.reset()
        ...
        env.close()
    """

    def _step_payload(self, action: ValidatorAction) -> Dict[str, Any]:
        """Convert action to JSON payload for the step message."""
        return {
            "action_type": action.action_type,
            # Phase 1
            "field_path": action.field_path,
            "violation_type": action.violation_type,
            "description": action.description,
            "suggested_fix": action.suggested_fix,
            # Phase 2
            "affected_services": list(action.affected_services),
            "reasoning": action.reasoning,
            # Phase 3
            "fix_strategy": action.fix_strategy,
            "spec_patch": dict(action.spec_patch),
            "rationale": action.rationale,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[ValidatorObservation]:
        """Parse the server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = ValidatorObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            phase=obs_data.get("phase", "detection"),
            api_spec=obs_data.get("api_spec", {}),
            payload=obs_data.get("payload", {}),
            violations_found=obs_data.get("violations_found", []),
            violations_remaining=obs_data.get("violations_remaining", 0),
            service_graph=obs_data.get("service_graph", {}),
            consumers_traced=obs_data.get("consumers_traced", []),
            total_consumers=obs_data.get("total_consumers", 0),
            detected_violation=obs_data.get("detected_violation", {}),
            consumer_specs=obs_data.get("consumer_specs", {}),
            fix_validation_results=obs_data.get("fix_validation_results", {}),
            feedback=obs_data.get("feedback", ""),
            max_steps=obs_data.get("max_steps", 0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ValidatorState:
        """Parse the state response into a typed ValidatorState."""
        return ValidatorState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            phase=payload.get("phase", "detection"),
            total_violations=payload.get("total_violations", 0),
            correct_reports=payload.get("correct_reports", 0),
            false_positives=payload.get("false_positives", 0),
            duplicate_reports=payload.get("duplicate_reports", 0),
            total_consumers=payload.get("total_consumers", 0),
            consumers_correctly_traced=payload.get("consumers_correctly_traced", 0),
            consumers_missed=payload.get("consumers_missed", 0),
            consumers_false_flagged=payload.get("consumers_false_flagged", 0),
            fix_attempts=payload.get("fix_attempts", 0),
            fix_validated=payload.get("fix_validated", False),
            fix_breaks_consumers=payload.get("fix_breaks_consumers", 0),
            score=payload.get("score", 0.0),
        )
