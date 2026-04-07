"""
Tests for the API Contract Validator Environment.

Run from the api_contract_validator/ directory:
    pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.environment import ValidatorEnvironment
from server.spec_generator import generate_scenario_for_task, AVAILABLE_TASKS
from models import ValidatorAction


@pytest.fixture
def env():
    """Fresh environment for each test."""
    return ValidatorEnvironment()


# ── Task structure ─────────────────────────────────────────────────────────


def test_six_tasks_registered():
    assert len(AVAILABLE_TASKS) == 6
    expected = {
        "find_type_mismatches",
        "validate_nested_objects",
        "detect_breaking_changes",
        "validate_response_schema",
        "validate_cross_field_constraints",
        "validate_auth_request",
    }
    assert set(AVAILABLE_TASKS) == expected


def test_all_tasks_have_violations():
    for task_name in AVAILABLE_TASKS:
        scenario = generate_scenario_for_task(task_name)
        assert len(scenario.violations) >= 4, (
            f"{task_name} has only {len(scenario.violations)} violations"
        )
        assert scenario.max_steps >= len(scenario.violations), (
            f"{task_name}: max_steps({scenario.max_steps}) < violations({len(scenario.violations)})"
        )


# ── Reset behaviour ────────────────────────────────────────────────────────


def test_all_tasks_reset_cleanly(env):
    for task_name in AVAILABLE_TASKS:
        obs = env.reset(task_name=task_name)
        assert obs.task_name == task_name
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.violations_found == []
        assert obs.violations_remaining > 0


# ── Correct violation reward ───────────────────────────────────────────────


def test_correct_violation_gives_plus_one(env):
    scenario = generate_scenario_for_task("find_type_mismatches")
    env.reset(task_name="find_type_mismatches")

    first = scenario.violations[0]
    action = ValidatorAction(
        field_path=first.field_path,
        violation_type=first.violation_type,
        description="test",
    )
    result = env.step(action)
    assert result.reward == 1.0
    assert len(result.violations_found) == 1


# ── False positive penalty ─────────────────────────────────────────────────


def test_false_positive_gives_negative_reward(env):
    env.reset(task_name="find_type_mismatches")
    action = ValidatorAction(
        field_path="nonexistent_field_xyz_abc",
        violation_type="type_mismatch",
        description="fabricated",
    )
    result = env.step(action)
    assert result.reward == pytest.approx(-0.3)


# ── Duplicate penalty ──────────────────────────────────────────────────────


def test_duplicate_gives_small_penalty(env):
    scenario = generate_scenario_for_task("find_type_mismatches")
    env.reset(task_name="find_type_mismatches")

    first = scenario.violations[0]
    action = ValidatorAction(
        field_path=first.field_path,
        violation_type=first.violation_type,
        description="test",
    )
    result1 = env.step(action)
    assert result1.reward == 1.0

    result2 = env.step(action)  # duplicate
    assert result2.reward == pytest.approx(-0.1)


# ── DONE signal ────────────────────────────────────────────────────────────


def test_done_signal_ends_episode(env):
    env.reset(task_name="find_type_mismatches")
    action = ValidatorAction(field_path="DONE", violation_type="", description="")
    result = env.step(action)
    assert result.done is True
    assert result.reward >= 0.0


# ── HINT mechanic ──────────────────────────────────────────────────────────


def test_hint_costs_half_point(env):
    env.reset(task_name="find_type_mismatches")
    action = ValidatorAction(field_path="HINT", violation_type="", description="")
    result = env.step(action)
    assert result.reward == pytest.approx(-0.5)
    assert "Hint:" in result.feedback
    assert result.done is False


# ── Proximity reward ───────────────────────────────────────────────────────


def test_proximity_reward_for_correct_path_wrong_type(env):
    scenario = generate_scenario_for_task("find_type_mismatches")
    env.reset(task_name="find_type_mismatches")

    first = scenario.violations[0]
    action = ValidatorAction(
        field_path=first.field_path,
        violation_type="extra_field",  # wrong type on purpose
        description="proximity test",
    )
    result = env.step(action)
    assert result.reward == pytest.approx(0.3)


# ── Seed reproducibility ───────────────────────────────────────────────────


def test_seed_gives_same_scenario():
    for task_name in AVAILABLE_TASKS:
        s1 = generate_scenario_for_task(task_name, seed=42)
        s2 = generate_scenario_for_task(task_name, seed=42)
        assert [v.field_path for v in s1.violations] == [
            v.field_path for v in s2.violations
        ], f"{task_name}: seed=42 gave different results across calls"


def test_different_seeds_give_different_easy_scenarios():
    """Easy task pool should vary with different seeds."""
    paths_by_seed = set()
    for seed in range(8):
        s = generate_scenario_for_task("find_type_mismatches", seed=seed)
        key = tuple(sorted(v.field_path for v in s.violations))
        paths_by_seed.add(key)
    assert len(paths_by_seed) > 1, "Different seeds produced identical scenarios"


# ── Cross-field task ───────────────────────────────────────────────────────


def test_cross_field_task_has_seven_violations():
    scenario = generate_scenario_for_task("validate_cross_field_constraints")
    assert len(scenario.violations) == 7


def test_cross_field_violations_use_correct_type():
    scenario = generate_scenario_for_task("validate_cross_field_constraints")
    for v in scenario.violations:
        assert v.violation_type == "cross_field_constraint", (
            f"Expected cross_field_constraint, got {v.violation_type} for {v.field_path}"
        )


# ── Auth task ──────────────────────────────────────────────────────────────


def test_auth_task_has_six_violations():
    scenario = generate_scenario_for_task("validate_auth_request")
    assert len(scenario.violations) == 6


def test_auth_task_variants_differ():
    s_even = generate_scenario_for_task("validate_auth_request", seed=0)
    s_odd = generate_scenario_for_task("validate_auth_request", seed=1)
    paths_even = {v.field_path for v in s_even.violations}
    paths_odd = {v.field_path for v in s_odd.violations}
    assert paths_even != paths_odd, "Even and odd seed should give different auth scenarios"


# ── Easy pool expansion ────────────────────────────────────────────────────


def test_easy_pool_has_twelve_variants():
    from server.spec_generator import _EASY_POOL
    assert len(_EASY_POOL) == 12, f"Expected 12 pool entries, got {len(_EASY_POOL)}"
