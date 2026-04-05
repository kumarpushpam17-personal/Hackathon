"""
Baseline Inference Script for API Contract Validator Environment.

Runs an LLM agent against all three tasks and produces scores.
Uses the OpenAI client for all LLM calls.

Environment variables:
    API_BASE_URL   — API endpoint (default: HF router)
    MODEL_NAME     — Model identifier (default: Qwen2.5-72B-Instruct)
    HF_TOKEN       — Hugging Face / API key

STDOUT format follows the hackathon specification exactly:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import ValidatorEnv
from models import ValidatorAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "api_contract_validator"
TASKS = [
    "find_type_mismatches",
    "validate_nested_objects",
    "detect_breaking_changes",
    "validate_response_schema",
    "validate_cross_field_constraints",
]
MAX_STEPS_PER_TASK = {
    "find_type_mismatches": 10,
    "validate_nested_objects": 15,
    "detect_breaking_changes": 20,
    "validate_response_schema": 25,
    "validate_cross_field_constraints": 18,
}
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
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert API contract validator. You will be given an OpenAPI \
specification and an API payload. Your job is to find violations in the \
payload that do not conform to the spec.

Each turn you must respond with EXACTLY one JSON object (no markdown, no \
explanation outside the JSON) with these fields:
{
    "field_path": "<dot-notation path to the violated field, or 'DONE' if no more violations>",
    "violation_type": "<type_mismatch|missing_required|invalid_enum|format_error|extra_field|breaking_change|cross_field_constraint>",
    "description": "<brief explanation>",
    "suggested_fix": "<how to fix it>"
}

Rules:
- Report ONE violation per turn.
- Use dot-notation for nested paths: 'customer.email'
- Use bracket notation for arrays: 'items[1].quantity'
- For breaking changes use: 'METHOD /path.field' e.g. 'POST /products.price'
- When you have found all violations, respond with field_path set to 'DONE'.
- Do NOT repeat a violation you already reported.
- For cross-field constraints (arithmetic, date ordering, conditional requirements), use violation_type='cross_field_constraint'.
- You may submit field_path='HINT' to receive a location hint at a cost of -0.5 reward.
""")


def build_user_prompt(
    observation: Dict[str, Any],
    step: int,
    history: List[str],
) -> str:
    """Build the user prompt from the current observation."""
    violations_found = observation.get("violations_found", [])
    found_summary = "None yet"
    if violations_found:
        found_lines = []
        for v in violations_found:
            found_lines.append(f"  - {v['field_path']}: {v['violation_type']}")
        found_summary = "\n".join(found_lines)

    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""\
Step: {step}
Task: {observation.get('task_name', '')}
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


def parse_llm_response(text: str) -> Dict[str, str]:
    """Parse the LLM response into a violation dict.

    Handles cases where the model wraps JSON in markdown fences.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return {
            "field_path": str(data.get("field_path", "DONE")),
            "violation_type": str(data.get("violation_type", "unknown")),
            "description": str(data.get("description", "")),
            "suggested_fix": str(data.get("suggested_fix", "")),
        }
    except json.JSONDecodeError:
        # If the model just says "DONE" or similar
        upper = cleaned.upper()
        if "DONE" in upper:
            return {
                "field_path": "DONE",
                "violation_type": "",
                "description": "",
                "suggested_fix": "",
            }
        return {
            "field_path": "DONE",
            "violation_type": "unknown",
            "description": f"Failed to parse: {cleaned[:100]}",
            "suggested_fix": "",
        }


def query_llm(
    client: OpenAI,
    observation: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, str]:
    """Send the current observation to the LLM and return a parsed action."""
    user_prompt = build_user_prompt(observation, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
        return {
            "field_path": "DONE",
            "violation_type": "",
            "description": f"LLM error: {exc}",
            "suggested_fix": "",
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_single_task(
    client: OpenAI,
    env: ValidatorEnv,
    task_name: str,
) -> None:
    """Run a single task episode and emit structured logs."""
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 15)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else result.observation.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_data = query_llm(client, obs_dict, step, history)

            action = ValidatorAction(
                field_path=action_data["field_path"],
                violation_type=action_data["violation_type"],
                description=action_data["description"],
                suggested_fix=action_data["suggested_fix"],
            )

            result = await env.step(action)
            obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else result.observation.__dict__

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_data['field_path']}:{action_data['violation_type']}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} → reward {reward:+.2f}"
            )

            if done:
                break

        # Compute final score
        if rewards:
            correct_count = sum(1 for r in rewards if r >= 1.0)
            total_violations = obs_dict.get("violations_remaining", 0) + len(
                obs_dict.get("violations_found", [])
            )
            score = (
                correct_count / total_violations if total_violations > 0 else 0.0
            )
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    """Run the inference agent against all tasks."""
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await ValidatorEnv.from_docker_image(IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        env = ValidatorEnv(base_url=env_url)

    try:
        for task_name in TASKS:
            await run_single_task(openai_client, env, task_name)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
