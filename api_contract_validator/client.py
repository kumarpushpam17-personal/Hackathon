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
            "field_path": action.field_path,
            "violation_type": action.violation_type,
            "description": action.description,
            "suggested_fix": action.suggested_fix,
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
            api_spec=obs_data.get("api_spec", {}),
            payload=obs_data.get("payload", {}),
            violations_found=obs_data.get("violations_found", []),
            violations_remaining=obs_data.get("violations_remaining", 0),
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
            total_violations=payload.get("total_violations", 0),
            correct_reports=payload.get("correct_reports", 0),
            false_positives=payload.get("false_positives", 0),
            duplicate_reports=payload.get("duplicate_reports", 0),
            score=payload.get("score", 0.0),
        )
