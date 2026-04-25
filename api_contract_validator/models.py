"""
Data models for the API Contract Validator Environment.

Defines typed Action, Observation, and State models that form the
contract between the agent and the environment across three phases:

    Phase 1 — Detection      action_type='report_violation'
    Phase 2 — Impact Tracing action_type='trace_impact'
    Phase 3 — Fix & Verify   action_type='propose_fix' | 'validate_fix'
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ── Action types ──────────────────────────────────────────────────────────

ACTION_REPORT_VIOLATION = "report_violation"
ACTION_TRACE_IMPACT = "trace_impact"
ACTION_PROPOSE_FIX = "propose_fix"
ACTION_VALIDATE_FIX = "validate_fix"


# ---------------------------------------------------------------------------
# Action — what the agent submits each step
# ---------------------------------------------------------------------------

class ValidatorAction(Action):
    """A single agent action.

    The ``action_type`` field selects which phase the action belongs to:

      * ``report_violation`` (Phase 1, default) — uses ``field_path`` and
        ``violation_type``. Special ``field_path`` values: ``DONE`` ends the
        episode, ``HINT`` requests a location clue at -0.5 reward.
      * ``trace_impact`` (Phase 2) — uses ``affected_services`` and
        ``reasoning``.
      * ``propose_fix`` / ``validate_fix`` (Phase 3) — uses
        ``fix_strategy``, ``spec_patch``, ``rationale``.

    All fields are optional so a single dataclass can carry every action
    type. Phase 1 callers that only set ``field_path`` + ``violation_type``
    continue to work without modification.
    """

    action_type: str = Field(
        default=ACTION_REPORT_VIOLATION,
        description=(
            "One of 'report_violation' (Phase 1), 'trace_impact' (Phase 2), "
            "'propose_fix' (Phase 3), 'validate_fix' (Phase 3)."
        ),
    )

    # ── Phase 1 — detection ──────────────────────────────────────────
    field_path: str = Field(
        default="",
        description=(
            "Dot-notation path to the violated field, e.g. 'user.email'. "
            "Use 'DONE' to signal no more violations. "
            "Use 'HINT' to receive a location hint at -0.5 reward cost."
        ),
    )
    violation_type: str = Field(
        default="",
        description=(
            "Category of violation: type_mismatch | missing_required | "
            "invalid_enum | format_error | extra_field | breaking_change | "
            "cross_field_constraint"
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

    # ── Phase 2 — impact tracing ─────────────────────────────────────
    affected_services: List[str] = Field(
        default_factory=list,
        description=(
            "Phase 2 — names of downstream services the agent believes "
            "are impacted by the breaking change."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Phase 2 — brief justification for the impact assessment.",
    )

    # ── Phase 3 — fix & verify ───────────────────────────────────────
    fix_strategy: str = Field(
        default="",
        description=(
            "Phase 3 — one of: field_alias | version_bump | "
            "deprecation_window | dual_write | consumer_patch."
        ),
    )
    spec_patch: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phase 3 — JSON-shaped patch to apply to the producer spec.",
    )
    rationale: str = Field(
        default="",
        description="Phase 3 — why the proposed fix preserves backward compatibility.",
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees after each step
# ---------------------------------------------------------------------------

class ValidatorObservation(Observation):
    """Environment response after each agent action.

    Inherits ``done: bool`` and ``reward: Optional[float]`` from the
    ``Observation`` base class.
    """

    # ── universal ────────────────────────────────────────────────────
    task_name: str = Field(default="", description="Current task identifier.")
    task_description: str = Field(
        default="",
        description="Natural-language instructions for the agent.",
    )
    phase: str = Field(
        default="detection",
        description="Current episode phase: detection | tracing | fix_proposal.",
    )
    feedback: str = Field(
        default="",
        description="Result of the last submitted action.",
    )
    max_steps: int = Field(
        default=0,
        description="Maximum steps allowed for the current episode.",
    )

    # ── Phase 1 — detection ──────────────────────────────────────────
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

    # ── Phase 2 — impact tracing ─────────────────────────────────────
    service_graph: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Phase 2 — enterprise service graph: {producer: spec, "
            "consumers: {name: {spec_excerpt, fields_consumed}}}."
        ),
    )
    consumers_traced: List[str] = Field(
        default_factory=list,
        description="Phase 2 — affected services the agent has correctly identified.",
    )
    total_consumers: int = Field(
        default=0,
        description="Phase 2 — total number of services in the graph.",
    )

    # ── Phase 3 — fix & verify ───────────────────────────────────────
    detected_violation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phase 3 — the breaking change that needs a fix.",
    )
    consumer_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phase 3 — consumer specs to validate the fix against.",
    )
    fix_validation_results: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Phase 3 — per-consumer validation results from the last "
            "validate_fix call."
        ),
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
    phase: str = "detection"

    # Phase 1
    total_violations: int = 0
    correct_reports: int = 0
    false_positives: int = 0
    duplicate_reports: int = 0

    # Phase 2
    total_consumers: int = 0
    consumers_correctly_traced: int = 0
    consumers_missed: int = 0
    consumers_false_flagged: int = 0

    # Phase 3
    fix_attempts: int = 0
    fix_validated: bool = False
    fix_breaks_consumers: int = 0

    score: float = 0.01
