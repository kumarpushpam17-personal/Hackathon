# API Contract Validator — Claude Code Project Instructions

## Project Overview
This is a submission for the **Meta PyTorch Hackathon (Round 1)** hosted on Scaler.
- **Goal**: API Contract Validator — an OpenEnv RL environment where AI agents validate API payloads against OpenAPI specifications.
- **Dashboard**: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#assessment
- **OpenEnv Course Repo**: `openenv-course/` (cloned reference material)

## Architecture

```
hackathon/
├── CLAUDE.md                              # This file
├── openenv-course/                        # Reference — read-only course modules
└── api_contract_validator/                # ← THE SUBMISSION
    ├── openenv.yaml                       # OpenEnv manifest
    ├── pyproject.toml                     # Python metadata + dependencies
    ├── Dockerfile                         # HF Spaces deployment
    ├── inference.py                       # Baseline inference script
    ├── README.md                          # Documentation (with HF frontmatter)
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

## 3 Tasks

| Task | Difficulty | Violations | Description |
|------|-----------|------------|-------------|
| `find_type_mismatches` | Easy | 4 | Top-level type errors, missing fields, bad enums |
| `validate_nested_objects` | Medium | 7 | Nested object + array violations |
| `detect_breaking_changes` | Hard | 9 | Breaking changes between API v1 and v2 |

## Key Files to Know

- **`server/spec_generator.py`** — Where task scenarios are defined. Each task has an API spec, a payload with planted violations, and ground-truth `PlantedViolation` records. To add/modify tasks, edit here.
- **`server/rewards.py`** — Reward logic. Correct = +1.0, false positive = -0.3, duplicate = -0.1, DONE bonus = 0.5 × completeness.
- **`server/environment.py`** — Core `ValidatorEnvironment` class. Handles matching agent reports against ground truth using fuzzy path matching.
- **`models.py`** — All Pydantic models. `ValidatorAction` (what agent submits), `ValidatorObservation` (what agent sees), `ValidatorState` (internal state).
- **`inference.py`** — Baseline agent script. Uses OpenAI client. Emits `[START]`/`[STEP]`/`[END]` logs.

## Hackathon Non-Negotiables

### inference.py MUST:
- Be named exactly `inference.py` in project root
- Use **OpenAI Client** for LLM calls
- Read env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Emit stdout in exact format:
  ```
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
  ```
- Complete in < 20 minutes on vcpu=2, 8GB RAM

### Environment MUST:
- Pass `openenv validate`
- `docker build && docker run` works
- HF Space responds 200 on POST `/reset`
- 3+ tasks with graders returning 0.0–1.0
- Reward provides partial progress (not just binary)

## Evaluation Weights
| Criterion | Weight |
|---|---|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality & spec compliance | 15% |
| Creativity & novelty | 10% |

## When Modifying This Project
1. After changing `spec_generator.py`, run the environment tests to verify violations match
2. After changing `models.py`, update `client.py` parsing methods accordingly
3. After changing `environment.py`, re-run `openenv validate`
4. After changing `Dockerfile`, rebuild and test: `docker build -t test . && docker run -p 7860:7860 test`
5. After changing `inference.py`, verify stdout format matches spec exactly
6. Consult `openenv-course/` modules for OpenEnv patterns and conventions

## Code Style
- Python 3.10+, type hints everywhere
- Pydantic v2 models for all data structures
- Clear, descriptive function names
- Docstrings on all public functions (Google/NumPy style)
- Keep environment deterministic
