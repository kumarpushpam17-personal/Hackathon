"""
Phase 3 — backward-compatibility fix validation.

Five strategies are accepted, each grading rule is independent so the
reward layer can compose a rubric:

    field_alias        — keep the old field name as an alias to the new one
    version_bump       — expose the change behind a new API version
    deprecation_window — keep the old field, mark it deprecated, document removal
    dual_write         — emit both the old and new field for a transition period
    consumer_patch     — coordinate consumer updates (only valid when the
                          producer cannot retain backward compat, e.g. enum
                          narrowing where the old values are illegal upstream)

A fix is graded against every consumer in the scenario's service graph.
The result includes per-consumer pass/fail so the reward layer can
penalise partial fixes (breaks ≥1 consumer) without false-rewarding
fixes that happen to satisfy the easy consumer but break the hard one.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .service_graph import CascadeScenario, ConsumerDeclaration


VALID_STRATEGIES = {
    "field_alias",
    "version_bump",
    "deprecation_window",
    "dual_write",
    "consumer_patch",
}


@dataclass
class FixValidationResult:
    """Outcome of validating one fix proposal against all consumers."""

    strategy: str
    is_well_formed: bool
    is_strategy_acceptable: bool
    consumers_passing: List[str] = field(default_factory=list)
    consumers_failing: List[str] = field(default_factory=list)
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def all_consumers_pass(self) -> bool:
        return (
            self.is_well_formed
            and self.is_strategy_acceptable
            and len(self.consumers_failing) == 0
            and len(self.consumers_passing) > 0
        )


# ── Strategy-specific consumer checks ────────────────────────────────────


def _check_field_alias(
    scenario: CascadeScenario,
    spec_patch: Dict[str, Any],
    consumer: ConsumerDeclaration,
) -> tuple[bool, str]:
    """Pass if the patch reintroduces every consumed field as an alias."""
    aliases = spec_patch.get("aliases") or spec_patch.get("field_aliases") or {}
    if not isinstance(aliases, dict) or not aliases:
        return False, "patch missing 'aliases' map"

    if scenario.scenario_id == "user_email_rename":
        # Patch must alias the old name back to the new name
        if "email" not in aliases:
            return False, "no alias for 'email'"
        target = aliases["email"]
        if target != "email_address":
            return False, f"alias points to '{target}', expected 'email_address'"
        # Consumer passes if it still consumes 'email' (alias covers it)
        if "email" in consumer.fields_consumed:
            return True, ""
        return True, "consumer not affected"

    if scenario.scenario_id == "orders_status_narrowed":
        # Aliasing doesn't help an enum narrowing — reject for affected consumers
        affected = consumer.name in scenario.ground_truth_affected
        if affected:
            return False, "field_alias cannot restore removed enum values"
        return True, "consumer not affected"

    return False, "unknown scenario for field_alias"


def _check_version_bump(
    scenario: CascadeScenario,
    spec_patch: Dict[str, Any],
    consumer: ConsumerDeclaration,
) -> tuple[bool, str]:
    """Pass if the patch declares both v1 (legacy) and v2 endpoints."""
    versions = spec_patch.get("versions") or []
    if not isinstance(versions, list) or len(versions) < 2:
        return False, "patch missing two-version declaration"
    has_legacy = any("v1" in str(v).lower() or "1.0" in str(v) for v in versions)
    has_new = any("v2" in str(v).lower() or "2.0" in str(v) for v in versions)
    if not (has_legacy and has_new):
        return False, "patch must keep v1 alongside v2"
    return True, ""


def _check_deprecation_window(
    scenario: CascadeScenario,
    spec_patch: Dict[str, Any],
    consumer: ConsumerDeclaration,
) -> tuple[bool, str]:
    """Pass if the patch keeps the old field/enum and marks it deprecated."""
    if scenario.scenario_id == "user_email_rename":
        deprecated = spec_patch.get("deprecated_fields") or []
        if "email" not in deprecated:
            return False, "must list 'email' under deprecated_fields"
        return True, ""

    if scenario.scenario_id == "orders_status_narrowed":
        deprecated_values = spec_patch.get("deprecated_enum_values") or []
        for value in ("cancelled", "refunded"):
            if value not in deprecated_values:
                return False, f"must keep '{value}' as deprecated enum"
        return True, ""

    return False, "unknown scenario for deprecation_window"


def _check_dual_write(
    scenario: CascadeScenario,
    spec_patch: Dict[str, Any],
    consumer: ConsumerDeclaration,
) -> tuple[bool, str]:
    """Pass if the patch emits both old and new field names simultaneously."""
    fields = spec_patch.get("emit_fields") or []
    if scenario.scenario_id == "user_email_rename":
        if "email" not in fields or "email_address" not in fields:
            return False, "must emit both 'email' and 'email_address'"
        return True, ""
    return False, "dual_write not supported for this scenario"


def _check_consumer_patch(
    scenario: CascadeScenario,
    spec_patch: Dict[str, Any],
    consumer: ConsumerDeclaration,
) -> tuple[bool, str]:
    """Pass if the patch lists every truly-affected consumer to migrate."""
    migrate = spec_patch.get("consumers_to_migrate") or []
    if not isinstance(migrate, list):
        return False, "consumers_to_migrate must be a list"
    if consumer.name in scenario.ground_truth_affected:
        if consumer.name not in migrate:
            return False, "affected consumer missing from migration list"
        return True, ""
    return True, ""


_STRATEGY_CHECKERS = {
    "field_alias": _check_field_alias,
    "version_bump": _check_version_bump,
    "deprecation_window": _check_deprecation_window,
    "dual_write": _check_dual_write,
    "consumer_patch": _check_consumer_patch,
}


# ── Public entry point ────────────────────────────────────────────────────


def validate_fix(
    scenario: CascadeScenario,
    strategy: str,
    spec_patch: Dict[str, Any],
) -> FixValidationResult:
    """Validate a fix proposal against all consumers in the scenario."""
    notes: List[str] = []

    if not isinstance(spec_patch, dict):
        return FixValidationResult(
            strategy=strategy,
            is_well_formed=False,
            is_strategy_acceptable=False,
            notes=["spec_patch must be a JSON object"],
        )

    if not strategy or strategy not in VALID_STRATEGIES:
        return FixValidationResult(
            strategy=strategy,
            is_well_formed=False,
            is_strategy_acceptable=False,
            notes=[
                f"strategy '{strategy}' not in {sorted(VALID_STRATEGIES)}"
            ],
        )

    is_acceptable = strategy in scenario.acceptable_fix_strategies
    if not is_acceptable:
        notes.append(
            f"'{strategy}' not in acceptable strategies "
            f"{scenario.acceptable_fix_strategies} for this scenario"
        )

    checker = _STRATEGY_CHECKERS[strategy]
    passing: List[str] = []
    failing: List[str] = []
    reasons: Dict[str, str] = {}

    for consumer in scenario.consumers:
        ok, reason = checker(scenario, spec_patch, consumer)
        if ok:
            passing.append(consumer.name)
        else:
            failing.append(consumer.name)
            reasons[consumer.name] = reason

    return FixValidationResult(
        strategy=strategy,
        is_well_formed=True,
        is_strategy_acceptable=is_acceptable,
        consumers_passing=passing,
        consumers_failing=failing,
        failure_reasons=reasons,
        notes=notes,
    )
