# API Contract Validator — Claude Code Project Instructions

## Project Overview
This is a submission for the **Meta PyTorch Hackathon (Round 1)** hosted on Scaler.
- **Goal**: API Contract Validator — an OpenEnv RL environment where AI agents validate API payloads against OpenAPI specifications.
- **Dashboard**: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#assessment
- **GitHub Repo**: https://github.com/kumarpushpam17-personal/Hackathon
- **OpenEnv Course Reference**: `../openenv-course/` (cloned reference material — read-only)

---

## Directory Layout

```
hackathon/
├── CLAUDE.md                              # This file (project root)
├── openenv-course/                        # Reference course — read-only
│   ├── module-1/README.md                 # Why OpenEnv, the RL loop, architecture
│   ├── module-2/README.md                 # Using existing environments, policies
│   ├── module-3/README.md                 # Deploying to HF Spaces, openenv push
│   ├── module-4/README.md                 # Building your own environment (3-component pattern)
│   └── module-5/README.md                 # GRPO training with TRL + OpenEnv
└── api_contract_validator/                # ← THE SUBMISSION
    ├── openenv.yaml                       # OpenEnv manifest
    ├── pyproject.toml                     # Python metadata + dependencies
    ├── Dockerfile                         # HF Spaces deployment
    ├── inference.py                       # Baseline inference script (MUST stay named this)
    ├── README.md                          # HF Spaces README (has frontmatter)
    ├── models.py                          # ValidatorAction, ValidatorObservation, ValidatorState
    ├── client.py                          # ValidatorEnv (WebSocket EnvClient)
    ├── __init__.py                        # Package exports
    ├── uv.lock                            # Dependency lockfile
    └── server/
        ├── __init__.py
        ├── app.py                         # FastAPI wiring via create_app()
        ├── environment.py                 # ValidatorEnvironment (reset/step/state)
        ├── spec_generator.py              # 3 task scenarios with planted violations
        ├── rewards.py                     # Reward computation (partial progress)
        └── requirements.txt               # Server-side deps
```

---

## 5 Tasks

| Task | Difficulty | Violations | Max Steps | Description |
|------|-----------|------------|-----------|-------------|
| `find_type_mismatches` | Easy | 4 | 10 | Top-level type errors, missing fields, bad enums. Pool of 8 → samples 4 per episode |
| `validate_nested_objects` | Medium | 7 | 15 | Nested object + array violations. 2 complete variants (Order Service / Event Booking) |
| `detect_breaking_changes` | Hard | 9 | 20 | Breaking changes between API v1 and v2 |
| `validate_response_schema` | Expert | 10 | 25 | Format errors in API response: invalid dates, pattern mismatches, out-of-range values. 2 variants |
| `validate_cross_field_constraints` | Expert | 7 | 18 | Cross-field arithmetic and date constraints on Invoice API (due_date ordering, line totals, subtotal sum, tax calculation, discount rules) |

---

## Hackathon Non-Negotiables (Pre-Submission Checklist)

### inference.py MUST:
- Be named exactly `inference.py` inside `api_contract_validator/`
- Use **OpenAI Client** for all LLM calls (not any other SDK)
- Read env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Emit stdout in EXACT format (any deviation = wrong eval score):
  ```
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
  ```
- Complete in < 20 minutes on vcpu=2, 8GB RAM

### Environment MUST:
- Pass `openenv validate`
- `docker build && docker run -p 7860:7860` works without errors
- HF Space responds 200 on POST `/reset`
- 3+ tasks with graders returning scores in 0.0–1.0
- Reward provides partial progress (not just binary 0 or 1)

### README.md MUST have HF Spaces frontmatter:
```yaml
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
```

### openenv.yaml MUST exist with correct fields:
```yaml
spec_version: 1
name: api_contract_validator
type: space
runtime: fastapi
app: server.app:app
port: 7860
```

---

## Evaluation Weights
| Criterion | Weight | What judges look for |
|---|---|---|
| Real-world utility | 30% | Genuine task, immediate value for RL/agent community |
| Task & grader quality | 25% | 3+ tasks, deterministic graders, real difficulty range |
| Environment design | 20% | Clean state, sensible action/obs spaces, good reward shaping |
| Code quality & spec compliance | 15% | openenv validate, docker works, typed models, documented |
| Creativity & novelty | 10% | Novel domain, clever reward design |

---

## Key Files to Know

- **`server/spec_generator.py`** — Task scenarios with planted violations. Edit here to add/modify tasks.
- **`server/rewards.py`** — Reward logic: Correct = +1.0, false positive = -0.3, duplicate = -0.1, DONE bonus = 0.5 × completeness.
- **`server/environment.py`** — Core `ValidatorEnvironment`. Fuzzy path matching for grading agent reports.
- **`models.py`** — All Pydantic models: `ValidatorAction`, `ValidatorObservation`, `ValidatorState`.
- **`client.py`** — `ValidatorEnv` WebSocket client used in inference.py and by external training code.
- **`inference.py`** — Baseline agent script. Uses OpenAI client. Run this to get baseline scores.

---

## Local Development Commands

```bash
# Run server (from api_contract_validator/)
cd api_contract_validator
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Docker (from api_contract_validator/)
docker build -t api-contract-validator .
docker run -p 7860:7860 api-contract-validator

# Test server is live
curl -X POST http://localhost:7860/reset

# Run inference (server must be running separately)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

---

## What Still Needs to Be Done

1. **Deploy to HF Spaces** — Use `/hf-deploy` command for step-by-step guidance
2. **Get baseline scores** — Run inference.py against all 3 tasks, document results in README
3. **openenv validate** — Must pass before submission
4. **Submit HF Space URL** on the Scaler dashboard before deadline

---

## When Modifying This Project
1. After changing `spec_generator.py` → run `/test-env` to verify violations match
2. After changing `models.py` → update `client.py` parsing accordingly
3. After changing `environment.py` → re-run `openenv validate`
4. After changing `Dockerfile` → rebuild and test locally before deploying
5. After changing `inference.py` → verify stdout format matches spec exactly
6. For OpenEnv patterns → read `../openenv-course/module-4/README.md`
7. For deployment guidance → read `../openenv-course/module-3/README.md`

---

## OpenEnv Course Reference Guide

| Module | File | When to Read |
|--------|------|-------------|
| Why OpenEnv | `../openenv-course/module-1/README.md` | Understanding the RL loop and architecture |
| Using Environments | `../openenv-course/module-2/README.md` | Writing policies, type-safe models |
| Deploying | `../openenv-course/module-3/README.md` | HF Spaces deployment, openenv push |
| Building Environments | `../openenv-course/module-4/README.md` | 3-component pattern, models/server/client |
| GRPO Training | `../openenv-course/module-5/README.md` | Training LLMs with this environment |

---

## Code Style
- Python 3.10+, type hints everywhere
- Pydantic v2 models for all data structures
- Clear, descriptive function names
- Docstrings on public functions
- Keep environment deterministic (same seed → same violations)
