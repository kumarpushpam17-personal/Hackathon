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

# Enterprise Contract Guardian — OpenEnv Environment

> **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale Submission**
> **Theme #3.1**: World Modeling → Professional Tasks · ⭐ **Scaler AI Labs bonus track**: Multi-App RL Environment for Enterprise Workflows

An OpenEnv RL environment that trains agents to do what senior platform engineers do when an API breaks in production: **detect the violation, trace which downstream services are affected, propose a backward-compatible fix, and verify the fix doesn't cascade**.

## The Story

> An engineer ships a "small" change to the Users API on Friday evening. It passes local tests. On Monday, **four downstream teams break** — the Orders service, the Billing pipeline, the Notification worker, and the Analytics ETL. The root cause: a single field renamed in one spec, with no awareness of who consumed it.
>
> This environment teaches agents the full workflow — not just "find the bug," but **reason about blast radius, propose fixes that preserve compatibility, and verify the migration across every consumer.**

## Why This Environment Matters (Theme #3.1 Alignment)

Per `themes.md` Theme #3.1: *"environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts."*

- ✅ **Real tools/APIs**: OpenAPI specs, payloads, consumer service graphs
- ✅ **Partially observable world**: agent discovers the consumer graph through queries
- ✅ **Persistent state**: violations found, consumers traced, fixes proposed build up across steps
- ✅ **Multi-step orchestration**: `detect → trace → propose → validate`
- ✅ **Enterprise workflow nuance**: versioning, deprecation, backward compatibility
- ✅ **Verifiable reward**: every step has a deterministic, objective grader

## Architecture: Phase 1 → Phase 2 → Phase 3

| Phase | What the agent does | Task examples |
|---|---|---|
| **Phase 1 — Detection** (inherited from Round 1) | Read one OpenAPI spec + payload, report violations | `find_type_mismatches`, `validate_nested_objects`, `detect_breaking_changes` |
| **Phase 2 — Impact Tracing** | Given a detected breaking change, identify all downstream consumers whose contracts are violated | `trace_downstream_blast_radius` |
| **Phase 3 — Fix & Verify** | Propose a backward-compatible migration; verify against every consumer spec | `propose_backward_compat_fix`, `multi_service_cascade_fix` |

**Real-world applications:**
- CI/CD contract gate that blocks a PR with predicted downstream impact
- Automated migration-plan generator for API versioning
- Enterprise API gateway pre-deployment safety check
- SDK compatibility auditor across microservices
- OAuth2/auth schema change impact analysis

## How It Works

Each episode places the agent inside a **simulated enterprise** with 3–5 microservices, each owning an OpenAPI spec and declaring which other services consume it.

```
Enterprise Service Graph
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ UsersService │ ─────▶ │ OrdersService │ ─────▶ │BillingService│
└──────────────┘        └──────────────┘        └──────────────┘
       │                        │
       ▼                        ▼
┌───────────────────┐    ┌──────────────────┐
│ NotificationsSvc  │    │  AnalyticsETL    │
└───────────────────┘    └──────────────────┘
```

### Episode Flow (Phase 2/3 tasks)

```
reset()
  →  Agent receives: a changed spec (producer) + partial service graph.

Phase 1 — Detection
  step(violation_report)    →  Correct? +1.0 | Proximity +0.3 | False positive -0.3 | Duplicate -0.1

Phase 2 — Impact Tracing
  step(trace_impact)        →  For each consumer correctly flagged: +reward; missed consumer: penalty

Phase 3 — Fix Proposal
  step(propose_fix)         →  Fix validates against ALL consumers: +big reward | breaks ≥1 consumer: penalty
  step(validate_fix)        →  Deterministic cross-spec check confirms/rejects the fix

step(DONE)  →  Completeness bonus = 0.5 × (correct_violations / total) × (consumers_traced / total) × fix_valid
```

Phase 1 tasks retain the simple single-spec flow (used as curriculum starters — `help_guide.md §6`).

## Tasks

### Phase 1 — Detection (curriculum starters, inherited from Round 1)

| Task | Difficulty | Violations | Max Steps | What the Agent Must Find |
|------|-----------|------------|-----------|--------------------------|
| `find_type_mismatches` | Easy | 4 | 10 | Type mismatches, missing required fields, invalid enums at the top level. Sampled from a pool of 12 — 495 unique episode combinations |
| `validate_nested_objects` | Medium | 7 | 15 | Violations inside nested objects and arrays — requires traversing deep structures. 2 variants: Order Service / Event Booking |
| `detect_breaking_changes` | Hard | 9 | 20 | Breaking changes between two API spec versions — type changes, removed fields, narrowed enums, new required fields |
| `validate_response_schema` | Expert | 10 | 25 | Subtle format errors in an API response: invalid date formats, pattern mismatches, out-of-range numerics, bad enum values. 2 variants |
| `validate_cross_field_constraints` | Expert | 7 | 18 | Cross-field arithmetic and date ordering on Invoice API — line totals, subtotal sum, tax calculation, discount rules for trial accounts |
| `validate_auth_request` | Expert | 6 | 14 | OAuth2 token and API key management violations — invalid grant types, bad scopes, MFA token patterns, IP format, rate limits. 2 variants |

### Phase 2 — Impact Tracing (finale — multi-service)

| Task | Difficulty | Max Steps | What the Agent Must Do |
|---|---|---|---|
| `trace_downstream_blast_radius` | Hard | 20 | Given a breaking change in a producer spec + a consumer service graph, identify every downstream service whose contract is violated. Graded on precision + recall against ground-truth consumer impact. |

### Phase 3 — Fix & Verify (finale — full workflow)

| Task | Difficulty | Max Steps | What the Agent Must Do |
|---|---|---|---|
| `propose_backward_compat_fix` | Expert | 25 | Given a detected breaking change, propose a migration (aliasing, deprecation, version bump). Graded by whether the fix validates against all consumer specs. |
| `multi_service_cascade_fix` | Expert | 40 | Full workflow: `detect → trace → propose → validate` in one episode, across 3–5 services. Sparse reward with per-phase sub-rewards. |

### Randomised Episode Generation

All tasks support seed-based randomisation, making the environment suitable for
**training** (varied seeds) as well as **evaluation** (fixed seeds):

- `find_type_mismatches` — samples 4 from a pool of 12 violations (495 unique combinations)
- `validate_nested_objects` — 2 complete scenario variants (Order Service / Event Booking)
- `validate_response_schema` — 2 complete scenario variants with different violation sets
- `validate_auth_request` — 2 complete scenario variants (OAuth2 / API key management)
- Pass `seed` in the `reset()` call to select a deterministic episode

## Action Space

Each step the agent submits a `ValidatorAction`. The action type it sends depends on the current episode phase.

### Detection actions (Phase 1)

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | `report_violation` |
| `field_path` | `str` | Dot-notation path to the violated field (e.g. `customer.email`, `items[1].quantity`). Special values: `DONE` to end episode, `HINT` for a location clue |
| `violation_type` | `str` | One of: `type_mismatch`, `missing_required`, `invalid_enum`, `format_error`, `extra_field`, `breaking_change`, `cross_field_constraint` |
| `description` | `str` | Human-readable explanation of the violation |
| `suggested_fix` | `str` | Optional suggested correction |

### Impact tracing actions (Phase 2 — finale)

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `trace_impact` |
| `affected_services` | `list[str]` | Names of downstream services the agent believes are impacted |
| `reasoning` | `str` | Brief justification for each entry |

### Fix-proposal actions (Phase 3 — finale)

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `propose_fix` or `validate_fix` |
| `fix_strategy` | `str` | One of: `field_alias`, `version_bump`, `deprecation_window`, `dual_write`, `consumer_patch` |
| `spec_patch` | `dict` | JSON patch to apply to the producer spec |
| `rationale` | `str` | Why this preserves backward compatibility |

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

Multiple **independent** reward signals (per `help_guide.md §7`) — reduces reward-hacking risk, provides rich training signal.

### Detection rewards (Phase 1)

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct violation (path + type match) | **+1.0** | Primary incentive |
| Proximity match (right path, wrong type) | **+0.3** | Encourages finding the right field first |
| HINT requested | **−0.5** | Informative but expensive |
| Duplicate report | **−0.1** | Light penalty — track what you already found |
| False positive | **−0.3** | Penalises guessing |
| DONE signal | **+0.5 × (found/total)** | Completeness bonus |

### Impact-tracing rewards (Phase 2)

| Event | Reward | Rationale |
|---|---|---|
| Correctly identified affected consumer | **+0.8** | Reward recall |
| Missed affected consumer | **−0.5** | Penalise under-reporting |
| False-flag unaffected consumer | **−0.4** | Penalise over-reporting |

### Fix-proposal rewards (Phase 3)

| Event | Reward | Rationale |
|---|---|---|
| Fix validates against ALL consumers | **+2.0** | Major incentive — this is the goal |
| Fix breaks 1+ consumer | **−1.0** | Must be backward compatible |
| Malformed spec patch | **−0.5** | Format compliance |
| Invalid strategy for this violation class | **−0.3** | Encourages strategy selection |

### Cross-cutting signals

| Signal | Reward | Rationale (`help_guide.md §7`) |
|---|---|---|
| **Step efficiency** | +0.05 per unused step at DONE | Discourages padding |
| **Format compliance** | −0.2 for malformed actions | Enforces schema |
| **Anti-hacking (spam)** | −1.0 if > 3× total violations reported | Prevents "report everything" exploit |

**Final episode score** = weighted blend of phase scores; see `server/rewards.py`.

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

## Training Results

> **Training**: GRPO via TRL + Unsloth · **Hardware**: HuggingFace Jobs T4 GPU

### Reward Curve

*Training plots will be embedded here after onsite training (Apr 25–26).*

<!-- After training, replace with:
![Reward Curve](results/reward_curve.png)
*Episode reward over training steps. Baseline (untrained) vs GRPO-trained agent. x-axis: training step, y-axis: episode reward (0–1).*

![Before vs After](results/before_after.png)
*Per-task score comparison. Baseline model (blue) vs trained checkpoint (green).*
-->

| Phase | WandB Run | Notebook |
|---|---|---|
| GRPO (Phase 1 + Phase 2/3) | *(link after training)* | [`training/grpo_colab.ipynb`](training/grpo_colab.ipynb) |

See [`training/README.md`](training/README.md) for the three ways to run the pipeline (Colab / HF Jobs / local).

### Baseline Scores (pre-training, Qwen2.5-72B-Instruct, recorded 2026-04-25)

| Task | Phase | Score | Steps | Success |
|---|---|---|---|---|
| `find_type_mismatches` | 1 | 0.75 | 4 | ✅ |
| `validate_nested_objects` | 1 | 0.99 | 12 | ✅ |
| `detect_breaking_changes` | 1 | **0.01** | 20 | ⛔ |
| `validate_response_schema` | 1 | 0.99 | 10 | ✅ |
| `validate_cross_field_constraints` | 1 | 0.86 | 8 | ✅ |
| `validate_auth_request` | 1 | 0.99 | 10 | ✅ |
| `trace_downstream_blast_radius` | 2 | 0.67 | 1 | ✅ |
| `propose_backward_compat_fix` | 3 | 0.99 | 1 | ✅ |
| `multi_service_cascade_fix` | 2+3 | 0.99 | 2 | ✅ |

Full per-step rewards in [`../baseline_scores.json`](../baseline_scores.json).

**Headroom for training**: `detect_breaking_changes` at 0.01 is the biggest opportunity — the 72B model finds the right field paths (proximity hits) but never predicts `violation_type='breaking_change'` correctly. Phase 2 trace is also under-shooting recall. After GRPO training the trained-model row will go alongside this table.

## Why This Matters

API contract violations are the **#1 cause of production incidents in microservice architectures**. Every platform team deals with this weekly. No existing RL environment teaches agents to reason about multi-service contract impact.

**Who benefits from an agent trained on this environment:**
- Platform / API gateway teams — pre-merge contract safety checks
- CI/CD pipelines — automated impact analysis before deploy
- API versioning toolchains — backward-compat migration planning
- Any engineering org operating ≥ 3 microservices

This is a genuinely underexplored domain in RL/LLM training — no prior benchmarks exist for multi-service API contract reasoning. A model trained here would be publishable as a research artifact.

## Links

| Resource | URL |
|---|---|
| HuggingFace Space (live env) | https://huggingface.co/spaces/pushpam14/api-contract-validator |
| Live env endpoint | https://pushpam14-api-contract-validator.hf.space |
| Health check | https://pushpam14-api-contract-validator.hf.space/health |
| Training Notebook (Colab) | [`training/grpo_colab.ipynb`](training/grpo_colab.ipynb) |
| GitHub repo | https://github.com/kumarpushpam17-personal/Hackathon |
| Demo Video / HF Blog | *(add after recording)* |
| WandB Training Run | *(add after training)* |

Quick test:

```bash
curl https://pushpam14-api-contract-validator.hf.space/health
# {"status":"healthy"}

curl -X POST https://pushpam14-api-contract-validator.hf.space/reset \
     -H "Content-Type: application/json" -d '{}'
```

## Project Structure

```
api_contract_validator/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Python project metadata
├── Dockerfile                # Container definition
├── inference.py              # Baseline inference script (OpenAI client, phase-aware)
├── README.md                 # This file
├── models.py                 # Pydantic models (Action, Observation, State — all 3 phases)
├── client.py                 # WebSocket client (EnvClient subclass)
├── __init__.py               # Package exports
├── results/                  # Training plots (.png) — committed, embedded above
├── tests/
│   └── test_environment.py   # 28 tests across all 3 phases
└── server/
    ├── app.py                # FastAPI wiring (create_app)
    ├── environment.py        # Core environment — multi-phase orchestration
    ├── logging_setup.py      # Structured JSON episode logging
    ├── spec_generator.py     # Phase 1 — task scenarios with planted violations
    ├── service_graph.py      # Phase 2 — simulated enterprise service graph
    ├── impact_tracer.py      # Phase 2 — ground-truth consumer-impact computation
    ├── fix_validator.py      # Phase 3 — cross-spec fix verification
    └── rewards.py            # Composable reward rubrics (multi-phase, independent signals)
```

## License

BSD-style — see LICENSE file.
