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

An OpenEnv RL environment where AI agents learn to validate API request/response
payloads against OpenAPI specifications. Agents identify type mismatches, missing
required fields, invalid enum values, format constraint violations, breaking API
changes, cross-field arithmetic errors, and authentication schema violations.

## Why This Environment?

API contract violations are one of the **top causes of production incidents** in
microservice architectures. Every API integration requires validating payloads
against specs — a tedious, error-prone task that developers perform daily. This
environment trains agents to automate this critical workflow.

**Real-world applications:**
- CI/CD pipeline contract testing
- API gateway request/response validation
- SDK compatibility checking between API versions
- OAuth2 and auth schema enforcement
- Migration safety audits

## How It Works

Each episode presents the agent with:
1. An **OpenAPI specification** defining expected types, required fields, and constraints
2. A **payload** containing planted violations

The agent inspects both and reports violations **one per step**. The environment
grades each report against ground-truth violations and provides immediate reward feedback.

```
reset()  →  Agent sees spec + payload with N hidden violations
step(violation_report)  →  Correct? +1.0 | Wrong path, right type? +0.3 | False positive? -0.3 | Duplicate? -0.1
step(HINT)  →  Receive a location clue at -0.5 cost
step(DONE)  →  Episode ends with completeness bonus +0.5 × (found/total)
```

## Tasks (6 difficulty levels)

| Task | Difficulty | Violations | Max Steps | What the Agent Must Find |
|------|-----------|------------|-----------|--------------------------|
| `find_type_mismatches` | Easy | 4 | 10 | Type mismatches, missing required fields, invalid enums at the top level. Sampled from a pool of 12 — 495 unique episode combinations |
| `validate_nested_objects` | Medium | 7 | 15 | Violations inside nested objects and arrays — requires traversing deep structures. 2 variants: Order Service / Event Booking |
| `detect_breaking_changes` | Hard | 9 | 20 | Breaking changes between two API spec versions — type changes, removed fields, narrowed enums, new required fields |
| `validate_response_schema` | Expert | 10 | 25 | Subtle format errors in an API response: invalid date formats, pattern mismatches, out-of-range numerics, bad enum values. 2 variants |
| `validate_cross_field_constraints` | Expert | 7 | 18 | Cross-field arithmetic and date ordering on Invoice API — line totals, subtotal sum, tax calculation, discount rules for trial accounts |
| `validate_auth_request` | Expert | 6 | 14 | OAuth2 token and API key management violations — invalid grant types, bad scopes, MFA token patterns, IP format, rate limits. 2 variants |

### Randomised Episode Generation

All tasks support seed-based randomisation, making the environment suitable for
**training** (varied seeds) as well as **evaluation** (fixed seeds):

- `find_type_mismatches` — samples 4 from a pool of 12 violations (495 unique combinations)
- `validate_nested_objects` — 2 complete scenario variants (Order Service / Event Booking)
- `validate_response_schema` — 2 complete scenario variants with different violation sets
- `validate_auth_request` — 2 complete scenario variants (OAuth2 / API key management)
- Pass `seed` in the `reset()` call to select a deterministic episode

## Action Space

Each step the agent submits a `ValidatorAction`:

| Field | Type | Description |
|-------|------|-------------|
| `field_path` | `str` | Dot-notation path to the violated field (e.g. `customer.email`, `items[1].quantity`). Special values: `DONE` to end episode, `HINT` for a location clue |
| `violation_type` | `str` | One of: `type_mismatch`, `missing_required`, `invalid_enum`, `format_error`, `extra_field`, `breaking_change`, `cross_field_constraint` |
| `description` | `str` | Human-readable explanation of the violation |
| `suggested_fix` | `str` | Optional suggested correction |

## Observation Space

After each step the agent receives a `ValidatorObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Current task identifier |
| `task_description` | `str` | Natural-language instructions for the agent |
| `api_spec` | `dict` | The OpenAPI specification (or version diff for hard task) |
| `payload` | `dict` | The API payload to validate |
| `violations_found` | `list[dict]` | Violations correctly identified so far |
| `violations_remaining` | `int` | Number of planted violations still undetected |
| `feedback` | `str` | Result of the last submitted report |
| `max_steps` | `int` | Step budget for this episode |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward for the last action |

## Reward Function

Partial progress signals — not binary end-of-episode scoring:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct violation (path + type match) | **+1.0** | Primary incentive |
| Proximity match (right path, wrong type) | **+0.3** | Encourages finding the right field first |
| HINT requested | **−0.5** | Informative but expensive |
| Duplicate report | **−0.1** | Light penalty — track what you already found |
| False positive | **−0.3** | Penalises guessing |
| DONE signal | **+0.5 × (found/total)** | Completeness bonus |

**Final score** = `correct_violations / total_violations` clamped to `(0.01, 0.99)`

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
```

## Baseline Scores

| Task | Model | Score | Steps |
|------|-------|-------|-------|
| `find_type_mismatches` | Qwen2.5-72B-Instruct | ~0.75 | 5–7 |
| `validate_nested_objects` | Qwen2.5-72B-Instruct | ~0.57 | 8–12 |
| `detect_breaking_changes` | Qwen2.5-72B-Instruct | ~0.44 | 12–18 |
| `validate_response_schema` | Qwen2.5-72B-Instruct | ~0.40 | 15–22 |
| `validate_cross_field_constraints` | Qwen2.5-72B-Instruct | ~0.43 | 10–16 |
| `validate_auth_request` | Qwen2.5-72B-Instruct | ~0.60 | 8–12 |

*Scores are approximate and may vary with temperature/sampling.*

## Project Structure

```
api_contract_validator/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Python project metadata
├── Dockerfile                # Container definition
├── inference.py              # Baseline inference script
├── README.md                 # This file
├── models.py                 # Pydantic models (Action, Observation, State)
├── client.py                 # WebSocket client (EnvClient subclass)
├── __init__.py               # Package exports
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI wiring (create_app)
    ├── environment.py        # Core environment logic (reset/step/state)
    ├── spec_generator.py     # Task scenarios with planted violations
    ├── rewards.py            # Reward computation
    └── requirements.txt      # Server dependencies
```

## License

BSD-style — see LICENSE file.
