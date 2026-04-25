"""
Baseline Inference Script for API Contract Validator Environment.

Runs an LLM agent against all tasks and produces scores.
Uses the OpenAI client for all LLM calls.

Environment variables:
    API_BASE_URL      — API endpoint (default: HF router)
    MODEL_NAME        — Model identifier (default: Qwen2.5-72B-Instruct)
    HF_TOKEN          — Hugging Face / API key (no default)
    LOCAL_IMAGE_NAME  — Docker image name when using from_docker_image()

STDOUT format follows the hackathon specification exactly:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file (if present) before reading any os.getenv values.
# .env is gitignored — keeps HF_TOKEN out of source control.
try:
    from dotenv import load_dotenv

    _ENV_FILE = Path(__file__).resolve().parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:
    pass  # python-dotenv not installed — fall back to OS env vars only

from openai import OpenAI

from client import ValidatorEnv
from models import ValidatorAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SCORES_OUT_PATH = os.getenv("SCORES_OUT_PATH")  # e.g. baseline_scores.json

BENCHMARK = "api_contract_validator"
PHASE1_TASKS = [
    "find_type_mismatches",
    "validate_nested_objects",
    "detect_breaking_changes",
    "validate_response_schema",
    "validate_cross_field_constraints",
    "validate_auth_request",
]
PHASE2_TASKS = ["trace_downstream_blast_radius"]
PHASE3_TASKS = ["propose_backward_compat_fix"]
CASCADE_TASKS = ["multi_service_cascade_fix"]
TASKS = PHASE1_TASKS + PHASE2_TASKS + PHASE3_TASKS + CASCADE_TASKS

MAX_STEPS_PER_TASK = {
    "find_type_mismatches": 10,
    "validate_nested_objects": 15,
    "detect_breaking_changes": 20,
    "validate_response_schema": 25,
    "validate_cross_field_constraints": 18,
    "validate_auth_request": 14,
    "trace_downstream_blast_radius": 20,
    "propose_backward_compat_fix": 25,
    "multi_service_cascade_fix": 40,
}
MAX_CONSECUTIVE_FAILURES = 3  # stop retrying same field after this many -0.3 rewards
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Structured stdout logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PHASE1 = textwrap.dedent("""\
You are an expert API contract validator. You will be given an OpenAPI \
specification and an API payload. Your job is to find ALL violations in the \
payload that do not conform to the spec.

Each turn you must respond with EXACTLY one JSON object (no markdown, no \
explanation outside the JSON):
{
    "action_type": "report_violation",
    "field_path": "<dot-notation path to the violated field, or 'DONE' if finished>",
    "violation_type": "<type_mismatch|missing_required|invalid_enum|format_error|extra_field|breaking_change|cross_field_constraint>",
    "description": "<brief explanation of the violation>",
    "suggested_fix": "<how to fix it>"
}

STRICT RULES:
1. Report ONE violation per turn. Be systematic — check every field.
2. field_path must be ONLY the path. NEVER put ':violation_type' inside field_path.
3. Paths: dot-notation 'customer.email', arrays 'items[1].quantity', breaking changes 'POST /path.field'.
4. violation_type choices:
   - type_mismatch: wrong data type (string vs integer, etc.)
   - missing_required: required field absent from payload
   - invalid_enum: value not in the allowed enum list
   - format_error: value violates format/pattern/min/max constraint
   - breaking_change: API v1→v2 change that breaks existing clients
   - cross_field_constraint: arithmetic/date/conditional rule across multiple fields
5. Do NOT repeat a violation already in 'Violations found so far'.
6. If last feedback was 'False positive' or negative reward, that field is WRONG — move to a different field.
7. When you have reported all violations, set field_path='DONE'.
8. You may set field_path='HINT' for a location clue at -0.5 reward cost.
""")

SYSTEM_PROMPT_PHASE2 = textwrap.dedent("""\
You are an enterprise API impact analyst. A producer microservice has made \
a breaking change to its API. You will see:
  * the breaking change (`violation`)
  * a list of consumer services with the fields each consumer depends on

Your job: identify EVERY downstream service whose contract is broken by \
the change. Submit a SINGLE action listing all affected consumers.

Respond with EXACTLY one JSON object:
{
    "action_type": "trace_impact",
    "affected_services": ["ServiceA", "ServiceB", ...],
    "reasoning": "<why these services are impacted>"
}

STRICT RULES:
1. Include a service ONLY if its declared `fields_consumed` overlaps with the \
   field affected by the breaking change.
2. For enum-narrowing changes, a consumer is affected only if it emits one \
   of the removed values.
3. Do NOT include services that consume unrelated fields — false flags are \
   penalised heavily.
4. Service names are case-insensitive but must match those in the consumer list.
""")

SYSTEM_PROMPT_PHASE3 = textwrap.dedent("""\
You are a senior platform engineer designing a backward-compatible \
migration. You see a breaking change and a list of consumer specs.

Your job: propose a fix (a `spec_patch`) that lets every consumer keep \
working without redeploying.

Respond with EXACTLY one JSON object:
{
    "action_type": "propose_fix",
    "fix_strategy": "<field_alias|version_bump|deprecation_window|dual_write|consumer_patch>",
    "spec_patch": { ... },
    "rationale": "<why this preserves backward compat>"
}

Strategy contracts (the patch must contain these keys):
  * field_alias        → spec_patch.aliases = { "<old_name>": "<new_name>", ... }
  * version_bump       → spec_patch.versions = ["v1.0", "v2.0"]   (must keep both)
  * deprecation_window → spec_patch.deprecated_fields = ["<old_field>"]
                          OR spec_patch.deprecated_enum_values = ["<old_val>"]
  * dual_write         → spec_patch.emit_fields = ["<old>", "<new>"]
  * consumer_patch     → spec_patch.consumers_to_migrate = [<every affected consumer>]

STRICT RULES:
1. Pick the strategy that fits the change. For enum narrowing, prefer \
   consumer_patch or version_bump (aliasing cannot restore enum values).
2. The patch must apply for EVERY affected consumer, not just one.
3. If the proposal fails, refine the patch and try again — do not repeat \
   the same failing patch.
""")

SYSTEM_PROMPT_CASCADE = SYSTEM_PROMPT_PHASE2 + "\n\n" + SYSTEM_PROMPT_PHASE3 + (
    "\n\nThe episode begins in Phase 2 (trace_impact). After every consumer "
    "is correctly traced, the environment switches to Phase 3 (propose_fix)."
)


def _system_prompt_for_phase(phase: str, task_name: str) -> str:
    if task_name in CASCADE_TASKS:
        return SYSTEM_PROMPT_CASCADE
    if phase == "tracing" or task_name in PHASE2_TASKS:
        return SYSTEM_PROMPT_PHASE2
    if phase == "fix_proposal" or task_name in PHASE3_TASKS:
        return SYSTEM_PROMPT_PHASE3
    return SYSTEM_PROMPT_PHASE1


def build_user_prompt(
    observation: Dict[str, Any],
    step: int,
    history: List[str],
) -> str:
    """Build a phase-aware user prompt from the current observation."""
    phase = observation.get("phase", "detection")
    task_name = observation.get("task_name", "")
    history_block = "\n".join(history[-5:]) if history else "None"

    if phase == "tracing" or task_name in PHASE2_TASKS:
        graph = observation.get("service_graph", {})
        return textwrap.dedent(f"""\
Step: {step}
Task: {task_name}
Phase: 2 — Impact Tracing
Instructions: {observation.get('task_description', '')}

Breaking change:
{json.dumps(graph.get('violation', observation.get('detected_violation', {})), indent=2)}

Consumers (each declares fields_consumed):
{json.dumps(graph.get('consumers', []), indent=2)}

Last feedback: {observation.get('feedback', '')}
Previous steps:
{history_block}

Respond with a single JSON object containing action_type='trace_impact'.
""")

    if phase == "fix_proposal" or task_name in PHASE3_TASKS:
        violation = observation.get("detected_violation", {})
        consumer_specs = observation.get("consumer_specs", {})
        last_results = observation.get("fix_validation_results", {})
        return textwrap.dedent(f"""\
Step: {step}
Task: {task_name}
Phase: 3 — Fix & Verify
Instructions: {observation.get('task_description', '')}

Breaking change to fix:
{json.dumps(violation, indent=2)}

Consumer specs (every consumer must keep working):
{json.dumps(consumer_specs, indent=2)}

Last fix attempt result: {json.dumps(last_results, indent=2) if last_results else 'None'}
Last feedback: {observation.get('feedback', '')}
Previous steps:
{history_block}

Respond with a single JSON object containing action_type='propose_fix'.
""")

    # Default: Phase 1 — Detection
    violations_found = observation.get("violations_found", [])
    if violations_found:
        found_lines = [
            f"  - {v['field_path']}: {v['violation_type']}"
            for v in violations_found
        ]
        found_summary = "\n".join(found_lines)
    else:
        found_summary = "None yet"

    return textwrap.dedent(f"""\
Step: {step}
Task: {task_name}
Phase: 1 — Detection
Instructions: {observation.get('task_description', '')}

API Specification:
{json.dumps(observation.get('api_spec', {}), indent=2)}

Payload to validate:
{json.dumps(observation.get('payload', {}), indent=2)}

Violations found so far:
{found_summary}

Violations remaining: {observation.get('violations_remaining', '?')}
Last feedback: {observation.get('feedback', '')}

Previous steps:
{history_block}

Respond with a single JSON object for the next violation (or DONE).
""")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


_VALID_VIOLATION_TYPES = {
    "type_mismatch", "missing_required", "invalid_enum",
    "format_error", "extra_field", "breaking_change", "cross_field_constraint",
}


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dict for any phase.

    Handles markdown fences and infers the action_type when the model
    omits it but includes phase-specific fields.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        upper = cleaned.upper()
        if "DONE" in upper:
            return {"action_type": "report_violation", "field_path": "DONE",
                    "violation_type": "", "description": "", "suggested_fix": ""}
        return {"action_type": "report_violation", "field_path": "DONE",
                "violation_type": "unknown",
                "description": f"Failed to parse: {cleaned[:100]}",
                "suggested_fix": ""}

    action_type = str(data.get("action_type", "")).strip()

    # Infer action_type when omitted
    if not action_type:
        if "affected_services" in data:
            action_type = "trace_impact"
        elif "fix_strategy" in data or "spec_patch" in data:
            action_type = "propose_fix"
        else:
            action_type = "report_violation"

    if action_type == "trace_impact":
        services = data.get("affected_services") or []
        if not isinstance(services, list):
            services = [str(services)]
        return {
            "action_type": "trace_impact",
            "affected_services": [str(s) for s in services],
            "reasoning": str(data.get("reasoning", "")),
        }

    if action_type in ("propose_fix", "validate_fix"):
        patch = data.get("spec_patch") or {}
        if not isinstance(patch, dict):
            patch = {}
        return {
            "action_type": action_type,
            "fix_strategy": str(data.get("fix_strategy", "")),
            "spec_patch": patch,
            "rationale": str(data.get("rationale", "")),
        }

    # Default: Phase 1 — report_violation
    field_path = str(data.get("field_path", "DONE"))
    violation_type = str(data.get("violation_type", "unknown"))
    if ":" in field_path:
        parts = field_path.split(":")
        if any(vt in parts[-1].lower() for vt in _VALID_VIOLATION_TYPES):
            field_path = parts[0].strip()
    return {
        "action_type": "report_violation",
        "field_path": field_path,
        "violation_type": violation_type,
        "description": str(data.get("description", "")),
        "suggested_fix": str(data.get("suggested_fix", "")),
    }


def _build_action(action_data: Dict[str, Any]) -> ValidatorAction:
    """Materialise a ValidatorAction from a parsed-LLM dict."""
    at = action_data.get("action_type", "report_violation")
    if at == "trace_impact":
        return ValidatorAction(
            action_type="trace_impact",
            affected_services=action_data.get("affected_services", []),
            reasoning=action_data.get("reasoning", ""),
        )
    if at in ("propose_fix", "validate_fix"):
        return ValidatorAction(
            action_type=at,
            fix_strategy=action_data.get("fix_strategy", ""),
            spec_patch=action_data.get("spec_patch", {}),
            rationale=action_data.get("rationale", ""),
        )
    return ValidatorAction(
        action_type="report_violation",
        field_path=action_data.get("field_path", "DONE"),
        violation_type=action_data.get("violation_type", ""),
        description=action_data.get("description", ""),
        suggested_fix=action_data.get("suggested_fix", ""),
    )


def _action_summary(action_data: Dict[str, Any]) -> str:
    at = action_data.get("action_type", "report_violation")
    if at == "trace_impact":
        return f"trace_impact:{','.join(action_data.get('affected_services', []))}"
    if at in ("propose_fix", "validate_fix"):
        return f"{at}:{action_data.get('fix_strategy','')}"
    return (
        f"{action_data.get('field_path','')}:"
        f"{action_data.get('violation_type','')}"
    )


def query_llm(
    client: OpenAI,
    observation: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Send the current observation to the LLM and return a parsed action."""
    phase = observation.get("phase", "detection")
    task_name = observation.get("task_name", "")
    user_prompt = build_user_prompt(observation, step, history)
    system_prompt = _system_prompt_for_phase(phase, task_name)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(raw_text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        # Safe fallback action for any phase
        if phase == "tracing" or task_name in PHASE2_TASKS:
            return {"action_type": "trace_impact", "affected_services": [],
                    "reasoning": f"LLM error: {exc}"}
        if phase == "fix_proposal" or task_name in PHASE3_TASKS:
            return {"action_type": "propose_fix", "fix_strategy": "",
                    "spec_patch": {}, "rationale": f"LLM error: {exc}"}
        return {"action_type": "report_violation", "field_path": "DONE",
                "violation_type": "", "description": f"LLM error: {exc}",
                "suggested_fix": ""}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_single_task(
    client: OpenAI,
    env: ValidatorEnv,
    task_name: str,
) -> Dict[str, Any]:
    """Run a single task episode and emit structured logs.

    Returns a dict with the per-task summary so the caller can aggregate
    a baseline_scores.json or trained_scores.json file.
    """
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 15)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    consecutive_failures = 0
    last_failed_path = ""
    is_phase1_task = task_name in PHASE1_TASKS

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs_dict = (
            result.observation.model_dump()
            if hasattr(result.observation, "model_dump")
            else result.observation.__dict__
        )

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Phase 1 only: trigger HINT after repeated failures on same field
            if is_phase1_task and consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                action_data = {
                    "action_type": "report_violation",
                    "field_path": "HINT",
                    "violation_type": "",
                    "description": "",
                    "suggested_fix": "",
                }
                consecutive_failures = 0
                last_failed_path = ""
            else:
                action_data = query_llm(client, obs_dict, step, history)

            action = _build_action(action_data)
            result = await env.step(action)
            obs_dict = (
                result.observation.model_dump()
                if hasattr(result.observation, "model_dump")
                else result.observation.__dict__
            )

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            action_str = _action_summary(action_data)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            # Phase 1 only: track stuck-on-same-field
            if (
                is_phase1_task
                and reward < 0
                and action_data.get("field_path") not in ("DONE", "HINT")
            ):
                fp = action_data.get("field_path", "")
                if fp == last_failed_path:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 1
                    last_failed_path = fp
            else:
                consecutive_failures = 0
                last_failed_path = ""

            history.append(
                f"Step {step}: {action_str} → reward {reward:+.2f}"
            )

            if done:
                break

        # Final score: trust env-side score (most accurate); fall back to
        # Phase 1 heuristic for back-compat with older Phase 1 evaluators.
        if is_phase1_task:
            if rewards:
                correct_count = sum(1 for r in rewards if r >= 1.0)
                total_violations = obs_dict.get("violations_remaining", 0) + len(
                    obs_dict.get("violations_found", [])
                )
                score = (
                    correct_count / total_violations
                    if total_violations > 0
                    else 0.0
                )
        else:
            # Phase 2/3 score is already computed by the environment.
            try:
                state = await env.state()
                score = getattr(state, "score", None) or 0.01
            except Exception:
                score = 0.01

        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": round(score, 4),
        "steps": steps_taken,
        "success": success,
        "rewards": [round(r, 4) for r in rewards],
    }


async def main() -> None:
    """Run the inference agent against all tasks and optionally save scores."""
    from datetime import datetime, timezone

    openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await ValidatorEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        env = ValidatorEnv(base_url=env_url)

    results: List[Dict[str, Any]] = []
    try:
        for task_name in TASKS:
            res = await run_single_task(openai_client, env, task_name)
            results.append(res)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

    if SCORES_OUT_PATH:
        scores_obj = {
            "model": MODEL_NAME,
            "benchmark": BENCHMARK,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "scores": {r["task"]: r["score"] for r in results},
            "details": results,
        }
        with open(SCORES_OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(scores_obj, fh, indent=2)
        print(f"[INFO] wrote {SCORES_OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
