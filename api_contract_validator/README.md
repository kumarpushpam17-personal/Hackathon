---
title: API Contract Validator
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# API Contract Validator — OpenEnv Environment

An OpenEnv RL environment where AI agents learn to validate API request/response payloads against OpenAPI specifications. Agents identify type mismatches, missing required fields, invalid enum values, and breaking changes between API versions.

## Why This Environment?

API contract violations are one of the **top causes of production incidents** in microservice architectures. Every API integration requires validating payloads against specs — a tedious, error-prone task that developers perform daily. This environment trains agents to automate this critical workflow.

**Real-world applications:**
- CI/CD pipeline contract testing
- API gateway validation
- SDK compatibility checking
- Migration safety audits between API versions

## How It Works

Each episode presents the agent with:
1. An **OpenAPI specification** defining expected types, required fields, and constraints
2. A **payload** containing planted violations

The agent inspects both and reports violations **one per step**. The environment grades each report against ground-truth violations and provides immediate feedback.

```
reset() → Agent sees spec + payload
step(violation_report) → Correct? +1.0 | False positive? -0.3 | Duplicate? -0.1
step(DONE) → Episode ends with completeness bonus
```

## Tasks (4 difficulty levels)

| Task | Difficulty | Violations | Max Steps | What the Agent Must Find |
|------|-----------|------------|-----------|--------------------------|
| `find_type_mismatches` | Easy | 4 | 10 | Type mismatches, missing required fields, invalid enums at the top level |
| `validate_nested_objects` | Medium | 7 | 15 | Violations inside nested objects and arrays — requires traversing deep structures |
| `detect_breaking_changes` | Hard | 9 | 20 | Breaking changes between two API spec versions — type changes, removed fields, narrowed enums, new requirements |
| `validate_response_schema` | Expert | 10 | 25 | Subtle format errors in an API response: invalid date formats, pattern mismatches, out-of-range numerics, and bad enum values scattered across nested objects and arrays |

### Randomised Episode Generation

All tasks support seed-based randomisation, making the environment suitable for **training** agents, not just evaluating them:

- `find_type_mismatches` — samples 4 violations from a pool of 8 (70 unique combinations)
- `validate_nested_objects` — two complete scenario variants (Order Service / Event Booking)
- `validate_response_schema` — two complete scenario variants with different violation sets
- Pass `seed` in the `reset()` call to select a deterministic episode

## Action Space

Each step the agent submits a `ValidatorAction`:

| Field | Type | Description |
|-------|------|-------------|
| `field_path` | `str` | Dot-notation path to the violated field (e.g., `customer.email`, `items[1].quantity`). Use `DONE` to end. |
| `violation_type` | `str` | One of: `type_mismatch`, `missing_required`, `invalid_enum`, `format_error`, `extra_field`, `breaking_change` |
| `description` | `str` | Human-readable explanation of the violation |
| `suggested_fix` | `str` | Optional suggested correction |

## Observation Space

After each step the agent receives a `ValidatorObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Current task identifier |
| `task_description` | `str` | Natural-language instructions |
| `api_spec` | `dict` | The OpenAPI specification (or version diff for hard task) |
| `payload` | `dict` | The payload to validate |
| `violations_found` | `list[dict]` | Violations correctly identified so far |
| `violations_remaining` | `int` | How many planted violations are still undetected |
| `feedback` | `str` | Result of the last submitted report |
| `max_steps` | `int` | Step budget for the episode |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward for the last action |

## Reward Function

The reward function provides **partial progress signals** — not binary end-of-episode scoring:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct violation found | +1.0 | Primary incentive — each discovery is rewarded |
| False positive | -0.3 | Penalises guessing without being too harsh |
| Duplicate report | -0.1 | Light penalty — agent should track what it already reported |
| DONE signal | +0.5 × (found/total) | Bonus proportional to completeness |

**Final score** = `correct_violations / total_violations` ∈ [0.0, 1.0]

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- `openenv-core` (`pip install openenv-core`)

### Local Development

```bash
git clone <your-repo-url>
cd api_contract_validator
pip install -e .

uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t api-contract-validator .
docker run -p 7860:7860 api-contract-validator
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"

python inference.py
```

## Validate Submission

```bash
openenv validate

# Full pre-submission check
./validate-submission.sh https://your-space.hf.space
```

## Baseline Scores

| Task | Model | Score | Steps |
|------|-------|-------|-------|
| `find_type_mismatches` | Qwen2.5-72B-Instruct | ~0.75 | 5–7 |
| `validate_nested_objects` | Qwen2.5-72B-Instruct | ~0.57 | 8–12 |
| `detect_breaking_changes` | Qwen2.5-72B-Instruct | ~0.44 | 12–18 |

*Scores are approximate and may vary with temperature/sampling.*

## Project Structure

```
api_contract_validator/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml             # Python project metadata
├── Dockerfile                 # Container definition
├── inference.py               # Baseline inference script
├── README.md                  # This file
├── models.py                  # Pydantic models (Action, Observation, State)
├── client.py                  # WebSocket client (EnvClient subclass)
├── __init__.py                # Package exports
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI wiring (create_app)
    ├── environment.py         # Core environment logic (reset/step/state)
    ├── spec_generator.py      # Task scenarios with planted violations
    ├── rewards.py             # Reward computation
    └── requirements.txt       # Server dependencies
```

## License

BSD-style — see LICENSE file.
