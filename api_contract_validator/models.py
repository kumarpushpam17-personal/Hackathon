"""
Data models for the API Contract Validator Environment.

Defines typed Action, Observation, and State models that form the
contract between the agent and the environment.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action — what the agent submits each step
# ---------------------------------------------------------------------------

class ValidatorAction(Action):
    """A single violation report submitted by the agent.

    The agent inspects an API spec + payload pair and reports one violation
    per step.  Submitting ``field_path="DONE"`` signals the agent is finished.
    """

    field_path: str = Field(
        ...,
        description=(
            "Dot-notation path to the violated field, e.g. 'user.email'. "
            "Use 'DONE' to signal no more violations to report."
        ),
    )
    violation_type: str = Field(
        ...,
        description=(
            "Category of violation: type_mismatch | missing_required | "
            "invalid_enum | format_error | extra_field | breaking_change"
        ),
    )
    description: str = Field(
        default="",
        description="Human-readable explanation of the violation.",
    )
    suggested_fix: str = Field(
        default="",
        description="Optional suggested correction for the violation.",
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees after each step
# ---------------------------------------------------------------------------

class ValidatorObservation(Observation):
    """What the environment returns after each agent action.

    Inherits ``done: bool`` and ``reward: Optional[float]`` from the
    ``Observation`` base class.
    """

    task_name: str = Field(
        default="",
        description="Identifier for the current task.",
    )
    task_description: str = Field(
        default="",
        description="Natural-language instructions for the agent.",
    )
    api_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="The OpenAPI specification (or spec diff for hard tasks).",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="The API request/response payload to validate.",
    )
    violations_found: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Violations the agent has correctly identified so far.",
    )
    violations_remaining: int = Field(
        default=0,
        description="Number of planted violations still undetected.",
    )
    feedback: str = Field(
        default="",
        description="Result of the agent's last submitted report.",
    )
    max_steps: int = Field(
        default=0,
        description="Maximum steps allowed for the current episode.",
    )


# ---------------------------------------------------------------------------
# State — internal environment state (includes ground-truth)
# ---------------------------------------------------------------------------

class ValidatorState(State):
    """Full environment state including ground-truth violations.

    Inherits ``episode_id: Optional[str]`` and ``step_count: int`` from
    the ``State`` base class.
    """

    task_name: str = ""
    total_violations: int = 0
    correct_reports: int = 0
    false_positives: int = 0
    duplicate_reports: int = 0
    score: float = 0.0
