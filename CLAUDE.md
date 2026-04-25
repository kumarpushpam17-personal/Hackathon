# Enterprise Contract Guardian — Project Instructions

## Project Overview

We cleared **Round 1** of the **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology** (52,000+ developers) and are competing in the **Grand Finale on Apr 25–26, 2026** at Scaler School of Technology, Bangalore.

> ⏰ **SUBMISSION DEADLINE: 2:00 PM, April 26 (Day 2).** Changes after this are not considered. One submission per team.

- **Finale project**: **Enterprise Contract Guardian** — multi-service RL environment where an agent detects API breaking changes, traces downstream blast radius, and proposes backward-compatible fixes.
- **Theme**: #3.1 World Modeling → Professional Tasks + ⭐ **Scaler AI Labs Bonus** (Multi-App RL for Enterprise Workflows)
- **GitHub Repo**: https://github.com/kumarpushpam17-personal/Hackathon
- **Dashboard**: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#assessment

---

## 📖 Reference Docs — Always Consult These

| Location | What's in it | When to read |
|---|---|---|
| `final_docs/themes.md` | 5 themes + judging criteria + standout criteria + minimum requirements | Deciding scope, validating theme fit, checking deliverables |
| `final_docs/help_guide.md` | Build guide — env design, Rubric system, rewards, anti-hacking, HF Jobs, training, plots | Before every design decision. §7 (rubrics), §8 (anti-hacking), §15 (monitoring) are highest-value |
| `final_docs/resource.md` | All links — OpenEnv, HF Jobs, TRL examples, Unsloth, WandB, training notebooks | When looking up SDK usage or training examples |
| `final_docs/final_guidelines.md` | Gap analysis from opening ceremony slides — 9 actionable items not in other docs | Review at start of each session; drives the action plan |
| `openenv-course/` | Read-only course — modules 1–5: OpenEnv architecture, building envs, deployment, GRPO | When implementing env/client/training code |
| `FINALE_CHECKLIST.md` | Live to-win checklist — all judging-aligned tasks with tick boxes | Check/update at the start of every work session |

---

## Finale Vision — Enterprise Contract Guardian

### One-Line Pitch Story
> "An engineer ships a 'small' API change Friday evening. Four downstream teams break on Monday. We built an environment that trains agents to catch this — not just spot the bad schema, but trace who it affects, propose a backward-compatible fix, and verify the fix doesn't cascade."

### Why This Wins (judges' own innovation questions)
- **"Does this teach an LLM something it currently can't do well?"** → Yes. Multi-service API contract impact tracing is not a standard LLM skill.
- **"Is the domain underexplored in RL/LLM training?"** → Yes. Enterprise API workflow RL has no existing benchmarks.
- **"Could a researcher write a paper about this?"** → Yes. "Training LLMs for API Contract Impact Analysis in Enterprise Microservices."

### Theme #3.1 Alignment
- ✅ Real interaction with tools, APIs, dynamic systems
- ✅ Partially observable world (agent discovers consumer graph)
- ✅ Multi-step orchestration (detect → trace → propose → validate)
- ✅ Enterprise workflow nuance (backward compatibility, versioning, migration safety)

---

## Judging Criteria — Every Code Decision Maps Here

| Weight | Criterion | Our winning strategy |
|:-:|---|---|
| **40%** | **Environment Innovation** | Multi-service enterprise workflow. Blast-radius tracing + backward-compat fix is the differentiator. "Messy but ambitious beats polished but boring." |
| **30%** | **Storytelling** | "Friday deploy breaks 4 teams" narrative. Service-graph visual. Concrete incident framing. README answers 4 judge questions (Problem / Env / Results / Why it matters). |
| **20%** | **Showing Improvement in Rewards** | Baseline numbers recorded pre-training. GRPO onsite. Reward-curve `.png` committed to repo and embedded in README. Before/after table. |
| **10%** | **Reward & Pipeline** | Composable rubrics (OpenEnv Rubric system). ≥5 independent signals. Anti-hacking. Deterministic grader. |

### Minimum Requirements (all four are non-negotiable)
- [ ] Uses OpenEnv (latest release)
- [ ] Training script with Unsloth or HF TRL (Colab notebook, re-runnable by judges)
- [ ] Evidence of actual training — reward plots committed to repo + WandB link if used
- [ ] Mini-blog on HuggingFace OR video on YouTube (< 2 min) linked from README
- [ ] Environment on HuggingFace Spaces — README links to Space URL

---

## Directory Layout

```
hackathon-api-contract-validator/          # git repo root → GitHub
├── CLAUDE.md                              # This file (gitignored)
├── FINALE_CHECKLIST.md                    # Live to-win checklist (gitignored)
├── .gitignore
├── .claude/commands/                      # Slash commands (gitignored)
├── final_docs/                            # Organiser docs + gap analysis (gitignored)
│   ├── themes.md                          # 5 themes + judging + standout criteria
│   ├── help_guide.md                      # Build guide (rubrics, training, plots, HF Jobs)
│   ├── resource.md                        # All links
│   └── final_guidelines.md               # Gap analysis from opening ceremony slides
├── openenv-course/                        # Reference course (gitignored, read-only)
│   └── module-1…5/README.md
└── api_contract_validator/                # ← THE SUBMISSION (public on GitHub + HF)
    ├── openenv.yaml
    ├── pyproject.toml
    ├── Dockerfile
    ├── inference.py                       # Baseline script — MUST stay named this
    ├── README.md                          # HF Spaces README (judge-facing)
    ├── models.py                          # ValidatorAction, ValidatorObservation, ValidatorState
    ├── client.py                          # WebSocket client
    ├── __init__.py
    ├── uv.lock
    ├── results/                           # Training plots (.png) committed here
    │   ├── reward_curve.png               # [After training] Embedded in README
    │   └── before_after.png               # [After training] Baseline vs trained
    ├── tests/
    │   └── test_environment.py
    └── server/
        ├── app.py                         # FastAPI wiring
        ├── environment.py                 # reset/step/state — orchestrates all phases
        ├── spec_generator.py              # Phase 1 — detection task scenarios
        ├── service_graph.py               # [Finale] Phase 2 — enterprise service graph
        ├── impact_tracer.py               # [Finale] Phase 2 — consumer-impact ground truth
        ├── fix_validator.py               # [Finale] Phase 3 — cross-spec fix verification
        ├── rewards.py                     # Multi-phase reward computation (OpenEnv Rubric system)
        └── requirements.txt
```

---

## Tasks (9 across 3 phases)

### Phase 1 — Detection (Round 1, curriculum starters)
| Task | Difficulty | Violations | Max Steps |
|---|---|---|---|
| `find_type_mismatches` | Easy | 4 | 10 |
| `validate_nested_objects` | Medium | 7 | 15 |
| `detect_breaking_changes` | Hard | 9 | 20 |
| `validate_response_schema` | Expert | 10 | 25 |
| `validate_cross_field_constraints` | Expert | 7 | 18 |
| `validate_auth_request` | Expert | 6 | 14 |

### Phase 2 — Impact Tracing (Finale)
| Task | Difficulty | Max Steps |
|---|---|---|
| `trace_downstream_blast_radius` | Hard | 20 |

### Phase 3 — Fix & Verify (Finale)
| Task | Difficulty | Max Steps |
|---|---|---|
| `propose_backward_compat_fix` | Expert | 25 |
| `multi_service_cascade_fix` | Expert | 40 |

---

## Non-Negotiables Before Submission

### inference.py
- Named exactly `inference.py` inside `api_contract_validator/`
- Uses **OpenAI Client** only
- Reads: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Stdout format (exact — any deviation breaks eval):
  ```
  [START] task=<name> env=<benchmark> model=<model>
  [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
  ```
- Completes in < 20 min on vcpu=2, 8GB RAM

### Environment
- `openenv validate` passes
- `docker build && docker run -p 7860:7860` works
- HF Space responds 200 on `POST /reset`
- 3+ tasks, graders return 0.0–1.0
- Reward gives partial progress (not binary)
- No reserved MCP tool names used (`reset`, `step`, `state`, `close`)

### README.md frontmatter (HF Spaces)
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

### README.md content (judges read in 3–5 min)
Must answer four questions:
1. **Problem** — What capability gap are you targeting?
2. **Environment** — What does the agent see, do, get rewarded for?
3. **Results** — What changed after training? Show it with plots.
4. **Why it matters** — Who benefits, and why?

Must link to: HF Space URL · Training notebook · Blog/video · WandB run

---

## Key Files

| File | Purpose | Edit trigger |
|---|---|---|
| `server/spec_generator.py` | Phase 1 task scenarios + planted violations | Adding/changing Phase 1 tasks |
| `server/service_graph.py` | Phase 2 enterprise graph data | Adding/changing services |
| `server/impact_tracer.py` | Phase 2 ground-truth consumer impact | Changing blast-radius grading |
| `server/fix_validator.py` | Phase 3 cross-spec fix verification | Changing fix strategies |
| `server/rewards.py` | All reward computation (Rubric-based) | Changing any reward signal |
| `server/environment.py` | Core `reset`/`step`/`state`, phase orchestration | Changing episode flow |
| `models.py` | Pydantic models for all action/obs/state | Adding new action types |
| `client.py` | WebSocket client | After changing models.py |
| `inference.py` | Baseline agent script | After changing tasks |

---

## Local Development

```bash
# Run server
cd api_contract_validator
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Docker
docker build -t api-contract-validator .
docker run -p 7860:7860 api-contract-validator

# Test live
curl -X POST http://localhost:7860/reset

# Run baseline inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py

# Run HF Jobs training (onsite — use $30 HF credit)
hf jobs uv run --with trl --flavor t4-small -s HF_TOKEN -- training.py
```

---

## When Modifying This Project

1. After `spec_generator.py` → run `/test-env`
2. After `models.py` → update `client.py` parsing
3. After `environment.py` → run `openenv validate`
4. After `rewards.py` → run anti-hacking tests (spam detector must still fire)
5. After `Dockerfile` → rebuild + test locally before deploying
6. After `inference.py` → verify stdout format exactly
7. For OpenEnv patterns → `openenv-course/module-4/README.md`
8. For deployment → `openenv-course/module-3/README.md`
9. For training → `openenv-course/module-5/README.md` + `final_docs/resource.md`

---

## Code Style

- Python 3.10+, type hints everywhere
- Pydantic v2 for all data models
- Use `Environment` base class (not `MCPEnvironment`) unless MCP tool integration is needed
- Rewards via OpenEnv Rubric system (composable rubrics > monolithic scoring)
- Same seed → same violations (deterministic)
- Client code must never import server internals
