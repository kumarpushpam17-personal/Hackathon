# API Contract Validator — Claude Code Project Instructions

## Project Overview

We cleared **Round 1** of the **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology** (52,000+ developers) and are competing in the **Grand Finale on Apr 25–26, 2026** at Scaler School of Technology, Bangalore.

- **Round 1 submission**: API Contract Validator — an OpenEnv RL environment for validating API payloads against OpenAPI specs.
- **Finale vision**: **Enterprise Contract Guardian** — a multi-service RL environment where an agent detects breaking changes, traces downstream blast radius across microservices, and proposes backward-compatible migration plans.
- **Theme alignment**: Theme #3.1 (World Modeling → Professional Tasks) + ⭐ **Scaler AI Labs Bonus Prize** (Multi-App RL Environment for Enterprise Workflows).
- **Dashboard**: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#assessment
- **GitHub Repo**: https://github.com/kumarpushpam17-personal/Hackathon

### 📖 Reference Docs — Always Consult These

| Location | What's in it | When to read |
|---|---|---|
| `final_docs/themes.md` | All 5 finale themes + Scaler AI Labs bonus + judging criteria + minimum requirements | When deciding scope, validating theme fit, checking required deliverables |
| `final_docs/help_guide.md` | Official self-serve build guide (22 sections — env design, rewards, anti-hacking, training stack, 1-day plan) | Before every major design decision. §7 (rewards), §8 (anti-hacking), §15 (monitoring), §18 (execution plan) are highest-value |
| `final_docs/resource.md` | Official links — OpenEnv repo, HF Hub, tutorials, YouTube, reward-engineering papers | When looking up SDK/API usage or training examples |
| `openenv-course/` | Read-only cloned course — modules 1–5 cover OpenEnv architecture, building envs, deployment, GRPO training | When implementing env/client/training code |
| `FINALE_CHECKLIST.md` | Live to-win checklist — minimum requirements, judging-aligned tasks, risk tracker | Check/update at the start of every work session |

---

## Finale Vision — Enterprise Contract Guardian

### One-Line Story (for the 3-min pitch)
> "An engineer ships a 'small' API change Friday evening. Four downstream teams break on Monday. We built an environment that trains agents to catch this — not just spot the bad schema, but trace who it affects, propose a backward-compatible fix, and verify the fix doesn't cascade."

### Theme #3.1 Alignment
- ✅ Real interaction with tools, APIs, dynamic systems
- ✅ Partially observable world (agent discovers consumer graph)
- ✅ Multi-step orchestration (detect → trace → propose → validate)
- ✅ Enterprise workflow nuance (backward compatibility, versioning, migration safety)

### What We Add on Top of Round 1

1. **Service Graph** — simulated enterprise with 3–5 microservices, each with its own OpenAPI spec + consumer declarations.
2. **Long-horizon tasks** — episodes require multiple coordinated steps (not just "find the bug").
3. **New actions** — `trace_impact`, `propose_fix`, `validate_fix`, alongside existing violation-reporting actions.
4. **Independent reward functions** (per `help_guide.md §7`) — correctness, blast-radius accuracy, fix compatibility, format compliance, step efficiency, anti-hacking check.
5. **Trained checkpoint** — real reward curve via TRL + Unsloth GRPO (onsite, days 1–2).

---

## Judging Criteria (Finale)

Source: `final_docs/themes.md`. **All code decisions should trace back to one of these weights.**

| Weight | Criterion | Our plan to win it |
|:-:|---|---|
| **40%** | **Environment Innovation** | Multi-service enterprise workflow — not a linter. Blast-radius tracing + backward-compat fix is the differentiator. |
| **30%** | **Storytelling** | "Friday deploy breaks 4 teams" narrative. Visual service-graph demo. Concrete incident framing. |
| **20%** | **Showing Improvement in Rewards** | Pre-compute baseline on day 0. Train onsite day 1 with TRL+Unsloth+GRPO. Plot reward curve. Before/after inference side-by-side. |
| **10%** | **Reward & Pipeline Quality** | ≥ 5 independent reward functions (per `help_guide.md §7`). Anti-hacking checks. Deterministic grader. |

### Minimum Requirements (must-haves — per `themes.md`)
- [ ] Uses OpenEnv (latest release)
- [ ] Minimal training script using Unsloth or HF TRL in Colab
- [ ] Mini-blog on HuggingFace OR mini-video on YouTube (< 2 min)
- [ ] Environment hosted on Hugging Face Spaces (OpenEnv-compliant)

---

## Directory Layout

```
hackathon-api-contract-validator/          # ← git repo root (pushed to GitHub)
├── CLAUDE.md                              # This file (gitignored — internal AI instructions)
├── FINALE_CHECKLIST.md                    # Gitignored — internal to-win checklist
├── .gitignore
├── .claude/commands/                      # Gitignored — slash commands for this project
├── final_docs/                            # Gitignored — hackathon organiser reference docs
│   ├── themes.md                          # 5 themes + judging criteria + minimum requirements
│   ├── help_guide.md                      # Official build guide (22 sections)
│   └── resource.md                        # Links — OpenEnv, HF Hub, tutorials, papers
├── openenv-course/                        # Gitignored — reference course (read-only)
│   ├── module-1/README.md                 # Why OpenEnv, the RL loop, architecture
│   ├── module-2/README.md                 # Using existing environments, policies
│   ├── module-3/README.md                 # Deploying to HF Spaces, openenv push
│   ├── module-4/README.md                 # Building your own environment (3-component pattern)
│   └── module-5/README.md                 # GRPO training with TRL + OpenEnv
└── api_contract_validator/                # ← THE SUBMISSION (public)
    ├── openenv.yaml                       # OpenEnv manifest
    ├── pyproject.toml                     # Python metadata + dependencies
    ├── Dockerfile                         # HF Spaces deployment
    ├── inference.py                       # Baseline inference script (MUST stay named this)
    ├── README.md                          # HF Spaces README (has frontmatter)
    ├── models.py                          # ValidatorAction, ValidatorObservation, ValidatorState
    ├── client.py                          # ValidatorEnv (WebSocket EnvClient)
    ├── __init__.py                        # Package exports
    ├── uv.lock                            # Dependency lockfile
    ├── tests/
    │   └── test_environment.py            # pytest suite
    └── server/
        ├── __init__.py
        ├── app.py                         # FastAPI wiring via create_app()
        ├── environment.py                 # ValidatorEnvironment (reset/step/state)
        ├── spec_generator.py              # Phase 1 task scenarios with planted violations
        ├── service_graph.py               # [Finale] Phase 2 — enterprise service graph
        ├── impact_tracer.py               # [Finale] Phase 2 — consumer-impact ground truth
        ├── fix_validator.py               # [Finale] Phase 3 — cross-spec fix verification
        ├── rewards.py                     # Reward computation (multi-phase, independent signals)
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

> See `FINALE_CHECKLIST.md` for the complete, live to-win checklist.

**Short version (high-level finale roadmap)**:
1. Extend environment to multi-service (service graph + new actions) — see `FINALE_CHECKLIST.md` §Environment
2. Add 2–3 long-horizon tasks (detect → trace → propose → validate) — §Tasks
3. Add independent reward functions (≥ 5) per `help_guide.md §7` — §Rewards
4. Deploy to HF Spaces (required minimum) — §Deployment
5. Run baseline inference → save numbers for before/after comparison — §Training
6. Onsite days 1–2: GRPO training with TRL + Unsloth → reward curve — §Training
7. Record < 2 min mini-video + HF blog — §Deliverables
8. Prepare 3-min pitch + demo — §Pitch

---

## When Modifying This Project
1. After changing `spec_generator.py` → run `/test-env` to verify violations match
2. After changing `models.py` → update `client.py` parsing accordingly
3. After changing `environment.py` → re-run `openenv validate`
4. After changing `Dockerfile` → rebuild and test locally before deploying
5. After changing `inference.py` → verify stdout format matches spec exactly
6. For OpenEnv patterns → read `openenv-course/module-4/README.md`
7. For deployment guidance → read `openenv-course/module-3/README.md`

---

## OpenEnv Course Reference Guide

| Module | File | When to Read |
|--------|------|-------------|
| Why OpenEnv | `openenv-course/module-1/README.md` | Understanding the RL loop and architecture |
| Using Environments | `openenv-course/module-2/README.md` | Writing policies, type-safe models |
| Deploying | `openenv-course/module-3/README.md` | HF Spaces deployment, openenv push |
| Building Environments | `openenv-course/module-4/README.md` | 3-component pattern, models/server/client |
| GRPO Training | `openenv-course/module-5/README.md` | Training LLMs with this environment |

---

## Code Style
- Python 3.10+, type hints everywhere
- Pydantic v2 models for all data structures
- Clear, descriptive function names
- Docstrings on public functions
- Keep environment deterministic (same seed → same violations)
