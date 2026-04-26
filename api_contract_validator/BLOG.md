# Enterprise Contract Guardian — training LLMs to reason about API blast radius

> This is the public mini-blog/writeup for our OpenEnv Hackathon submission.

*Submission to the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology Grand Finale (Apr 25–26, 2026). Theme #3.1: World Modeling → Professional Tasks. Scaler AI Labs Bonus Track.*

---

## TL;DR

We built **Enterprise Contract Guardian**, an OpenEnv environment that trains LLM agents to reason about API contract blast radius across microservices.

The key result: both untrained Qwen2.5-72B and untrained Qwen2.5-7B scored **0.01** on `detect_breaking_changes`. After **300 GRPO steps**, Qwen2.5-7B + LoRA scored **0.67** on the same task. This shows the environment taught a targeted capability that model scale alone did not solve.

The agent learns to:

- detect API contract violations,
- trace which downstream services break,
- propose backward-compatible fixes,
- and verify that the fix does not cascade into another outage.

---

## Who this is for

This writeup is for people interested in training LLM agents on realistic professional workflows rather than toy environments. If you care about API reliability, CI/CD gates, platform engineering, OpenEnv environments, or RL training with verifiable reward signals, this is the problem we are targeting.

By the end, you should understand:

- what the environment simulates,
- what actions the agent can take,
- how the reward signal teaches the behavior,
- what we trained with GRPO,
- and where the trained model improved.

## What we built

We built a hosted OpenEnv environment called **Enterprise Contract Guardian**.

Live environment: https://huggingface.co/spaces/pushpam14/api-contract-validator  
Primary HF Space writeup: https://huggingface.co/spaces/pushpam14/api-contract-validator/blob/main/BLOG.md  
Trained adapter: https://huggingface.co/pushpam14/api-contract-validator-grpo-7b  
Training run: https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k  
Source code: https://github.com/kumarpushpam17-personal/Hackathon  
GitHub README: https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/README.md

The end result is a training environment where an LLM agent interacts with a simulated enterprise API ecosystem and learns to reason about contract changes, downstream consumers, and backward-compatible fixes.

## Reviewer quick path

If you only have a few minutes:

1. Open the live environment: https://huggingface.co/spaces/pushpam14/api-contract-validator
2. Read the headline result in [Section 5](#5-results): `detect_breaking_changes` improves from **0.01 → 0.67** after GRPO.
3. Inspect the two committed plots: reward curve and three-way before/after comparison.
4. Verify training proof: https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/results/TRAINING_RUN_PROOF.md
5. Reproduce one live episode with the `curl` command in [Section 8](#8-try-it-yourself).

For full setup, action schema, reward tables, and local run instructions, use the GitHub README: https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/README.md

## What makes this submission strong

This is not a static prompt benchmark. It is a runnable OpenEnv environment with hidden ground truth, stateful episodes, objective rewards, real training, and public proof artifacts.

| Area | What is included |
|---|---|
| Environment depth | 9 tasks across detection, downstream impact tracing, and backward-compatible fix verification |
| Episode/data mix | Seeded synthetic enterprise API scenarios: OpenAPI specs, payloads, version diffs, consumer service graphs, and migration candidates |
| Reward richness | 14 independent reward signals covering correct findings, proximity, duplicates, false positives, missed consumers, malformed patches, broken consumers, and anti-spam |
| Training evidence | 300 GRPO steps on Qwen2.5-7B + LoRA, public WandB run, reward curve, training state JSON, full logs, and trained adapter on Hugging Face |
| Before/after evaluation | Three-way comparison: untrained Qwen2.5-72B, untrained Qwen2.5-7B, and trained Qwen2.5-7B + LoRA |

The environment uses generated, deterministic scenarios rather than a scraped external dataset. That is intentional: every episode has known ground truth, which makes the reward signal auditable and lets judges reproduce the same task with a fixed `seed`.

## The story in 30 seconds

It is Friday evening. A backend engineer makes what looks like a small API cleanup:

```diff
{
-  "email": "user@example.com"
+  "email_address": "user@example.com"
}
```

The producer service deploys successfully because its own tests pass. By Monday morning four downstream teams are paged: Orders cannot attach customer emails to receipts, Billing cannot send invoices, Notifications goes silent, Analytics quietly drops a field. The bug was not "the API changed." The bug was that **nobody traced who depended on that field** and nobody proposed a migration the old consumers could survive.

Today's LLMs are great at spotting individual schema violations. They are not great at reasoning across a microservice graph and producing migration patches that keep every consumer running. There is no RL benchmark for this. So we built one.

## 1) Theme fit: World Modeling / Professional Tasks

This submission targets **Theme #3.1: World Modeling → Professional Tasks**.

The environment simulates a partially observable enterprise API ecosystem. The agent sees OpenAPI specs, example payloads, consumer dependencies, and step feedback. It does not directly see the ground-truth blast radius. It must infer which consumers depend on which fields, update its belief after every action, and choose fixes that pass validation against every consumer contract.

This is not a static benchmark or a prompt-only eval. The training loop calls the environment's real `reset`, `step`, and `state` interfaces. Rewards come from the environment's own grader.

## 2) Environment design

An OpenEnv environment with three phases that mirror what a senior platform engineer actually does:

1. **Detect** — find the contract violation
2. **Trace** — identify which downstream consumers break (the "blast radius")
3. **Fix & verify** — propose a backward-compatible migration and validate it against every consumer's spec

Each episode places the agent inside a simulated enterprise (3–5 microservices, each owning an OpenAPI spec, each declaring which fields it consumes from upstream). When the producer ships a breaking change, the agent has to figure out who breaks — but the ground-truth answer is hidden. The agent must reason from the consumer declarations.

Then the agent has to propose a fix. Five backward-compat strategies are accepted: `field_alias`, `version_bump`, `deprecation_window`, `dual_write`, `consumer_patch`. The fix is validated against every consumer in the graph. If even one consumer would still break, the agent gets penalized.

### Agent interface

The agent interacts through normal OpenEnv-style calls:

- `reset(task_name, seed)` starts a task episode,
- `state()` returns the current observation,
- `step(action)` submits an action and receives reward,
- `close()` ends the session.

The action space is intentionally small and inspectable: report a violation, trace an impacted consumer, propose a fix, or mark the task done. This keeps the environment easy to run while still requiring non-trivial reasoning.

## 3) Reward signal — composable rubrics, 14 independent components

The reward is not a single pass-fail score. It is an OpenEnv Rubric composed of fourteen independent signals — correct violations, correct consumers, missed consumers, false flags, malformed patches, broken consumers, anti-spam, and more. Each is logged separately so we can see exactly which signal drives training.

```text
Phase 1 (detection): correct +1.0 | proximity +0.3 | duplicate -0.1 | false positive -0.3 | hint -0.5 | done bonus
Phase 2 (tracing):   correct consumer +0.8 | missed -0.5 | false flag -0.4 | unknown service -0.2
Phase 3 (fix):       fix passes ALL consumers +2.0 | breaks consumer -1.0 | malformed -0.5 | unacceptable strategy -0.3
Cross-cutting:       malformed action -0.2 | spam (>3× violations) -1.0
```

14 signals, all independent, hard to game without actually solving the task.

## 4) Training setup — GRPO via TRL + Unsloth

We trained Qwen2.5-7B-Instruct (4-bit) with LoRA r=16 for 300 GRPO steps on a single Hugging Face Jobs L4 GPU.

The important detail: the reward function is the **environment's own grader**, called step by step. This is not supervised fine-tuning on a static dataset. The model samples actions, sends them to the OpenEnv environment, receives reward, and GRPO updates the LoRA adapter from that feedback.

[Public WandB run: `grpo-7b-l4-300steps-v3`](https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k)

## 5) Results

We compared three configurations, all at the same inference temperature (0.7):

| Task | Qwen-72B | Qwen-7B | **Qwen-7B + LoRA** |
|---|---:|---:|---:|
| `find_type_mismatches` | 0.75 | 0.75 | 0.75 |
| `validate_nested_objects` | 0.99 | 0.57 | 0.57 |
| **`detect_breaking_changes`** | **0.01** | **0.01** | **0.67** |
| `validate_response_schema` | 0.99 | 0.70 | 0.30 |
| `validate_cross_field_constraints` | 0.99 | 0.43 | 0.29 |
| `validate_auth_request` | 0.99 | 0.83 | 0.33 |
| `trace_downstream_blast_radius` | 0.67 | 0.99 | 0.99 |
| `propose_backward_compat_fix` | 0.99 | 0.99 | 0.99 |
| `multi_service_cascade_fix` | 0.99 | 0.99 | 0.99 |

**The headline**: on `detect_breaking_changes`, both untrained models — including the **10× larger 72B** — score 0.01. They earn the +0.3 proximity reward repeatedly (they know *where* the breaking change is) but never predict `violation_type='breaking_change'` correctly. After 300 GRPO steps targeting our environment's reward, the trained 7B+LoRA scores **0.67**.

That is **+66 percentage points on a task where pure scale gave nothing.** This is RL training value, isolated from model size.

### Training curve and before/after plot

The training curve and three-way before/after comparison are committed in the GitHub repo so reviewers do not need access to a local notebook.

![Reward curve](https://raw.githubusercontent.com/kumarpushpam17-personal/Hackathon/main/api_contract_validator/results/reward_curve.png)

![Before vs after](https://raw.githubusercontent.com/kumarpushpam17-personal/Hackathon/main/api_contract_validator/results/before_after.png)

### Proof and artifacts

The training run is backed by public, reproducible artifacts:

| Artifact | Link |
|---|---|
| Public WandB run | https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k |
| Training proof summary | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/results/TRAINING_RUN_PROOF.md |
| Full training log | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/results/training_full_log.txt |
| Training state JSON | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/results/training_state.json |
| Trained scores | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/trained_scores.json |
| 7B baseline scores | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/baseline_7b_scores.json |
| 72B baseline scores | https://github.com/kumarpushpam17-personal/Hackathon/blob/main/baseline_72b_v2_scores.json |
| Trained LoRA adapter | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b |

## 6) The honest trade-off

GRPO heavily reinforced the high-reward action patterns from training (Phase 2/3 episodes give +2.0 fix rewards vs Phase 1's +1.0 per violation). The trained model now over-applies these patterns to Phase 1 tasks where they don't fit, causing regressions on `validate_response_schema`, `validate_cross_field_constraints`, and `validate_auth_request`. With task-balanced training and a "don't repeat" reward signal, this would close. But the headroom-task win is real and reproducible.

## 7) Why this matters

API contract violations are the **#1 cause of production incidents in microservice architectures**. Every platform team deals with this weekly. Three groups benefit from an agent trained on this environment:

- **Platform / API gateway teams** — pre-merge contract gates that predict downstream impact
- **CI/CD pipelines** — automated impact analysis before deploy
- **API versioning toolchains** — backward-compat migration planning

There is no existing RL benchmark for multi-service contract reasoning. This is genuinely a publishable artifact — researchers training LLMs for enterprise workflows now have a benchmark to compete on.

## 8) Try it yourself

- **Live environment**: https://huggingface.co/spaces/pushpam14/api-contract-validator
- **Primary HF Space writeup**: https://huggingface.co/spaces/pushpam14/api-contract-validator/blob/main/BLOG.md
- **Trained adapter**: https://huggingface.co/pushpam14/api-contract-validator-grpo-7b
- **WandB run**: https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k
- **GitHub**: https://github.com/kumarpushpam17-personal/Hackathon
- **GitHub README**: https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/README.md
- **Story doc + technical guide**: https://github.com/kumarpushpam17-personal/Hackathon/blob/main/api_contract_validator/ENTERPRISE_CONTRACT_GUARDIAN_STORY.md

```bash
# Try the live environment in 10 seconds
curl https://pushpam14-api-contract-validator.hf.space/health
# {"status":"healthy"}

curl -X POST https://pushpam14-api-contract-validator.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name":"trace_downstream_blast_radius","seed":1}'
```

## Final takeaway

The important result is not that one model became better at every task. It did not. The important result is narrower and more useful: an OpenEnv reward signal taught a 7B model a specific enterprise reasoning behavior that neither the untrained 7B nor the much larger 72B model could perform.

That is exactly why this environment matters. It creates a repeatable training loop for API blast-radius reasoning: detect the contract break, trace affected consumers, propose a migration, and verify the result.

## Acknowledgements

Built with [`openenv-core`](https://github.com/meta-pytorch/openenv), [`trl`](https://huggingface.co/docs/trl), [`unsloth`](https://github.com/unslothai/unsloth), Hugging Face Spaces, Hugging Face Jobs, Hugging Face Hub, and the OpenEnv composable rubric pattern.

Thanks to Meta PyTorch and Scaler School of Technology for the hackathon, Hugging Face for the hosting/training infrastructure, and the OpenEnv team for the framework.
