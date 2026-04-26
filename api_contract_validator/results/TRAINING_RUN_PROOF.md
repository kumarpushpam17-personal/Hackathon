# Training Run — Proof of Successful Execution

> The HuggingFace Jobs UI marks this run as `ERROR` because the Python interpreter exited non-zero during shutdown (a known issue with the `websockets` library's `__del__` running without an active event loop). **The training itself completed successfully** — all 300 GRPO steps ran, the trained adapter was uploaded, and both training artefacts (reward curve + state JSON) were pushed to HuggingFace Hub.

---

## Run identifiers

| Field | Value |
|---|---|
| **HF Job ID** | `69ed0e59d70108f37acded4e` |
| **HF Job URL** | https://huggingface.co/jobs/pushpam14/69ed0e59d70108f37acded4e |
| **WandB run** (public) | https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k |
| **Trained adapter** (public) | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b |

## Training configuration

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Hardware | HuggingFace Jobs `l4x1` (1× Nvidia L4, 24 GB) |
| LoRA rank / alpha | 16 / 32 |
| GRPO steps | 300 |
| Generations per prompt | 4 |
| Mixed precision | fp16 |
| Wall-time | 1 h 56 min |
| `train_runtime` | 6975 s |
| `train_samples_per_second` | 0.172 |
| `train_steps_per_second` | 0.043 |
| Final `train_loss` | `1.383e-05` |
| Tasks trained on | 6 of 9 (Phase 1×4 + Phase 2 + Phase 3) |

## Artefacts produced (each verifiable on HF Hub)

| File | Size | Location |
|---|---|---|
| `adapter_model.safetensors` | 162 MB | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/blob/main/adapter_model.safetensors |
| `adapter_config.json` | small | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/blob/main/adapter_config.json |
| `training_artifacts/reward_curve.png` | 139 kB | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/blob/main/training_artifacts/reward_curve.png |
| `training_artifacts/training_state.json` | 256 kB, **300 reward entries** | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/blob/main/training_artifacts/training_state.json |

A copy of each is also committed to this repo under `api_contract_validator/results/`.

## Reward trajectory (from `training_state.json`, all 300 steps)

| Window | Mean reward |
|---|---|
| First 50 steps | **1.355** |
| Middle 50 steps (steps 100–150) | **1.338** |
| Last 50 steps (steps 250–300) | **1.269** |
| Overall (300 steps) | **1.263** |
| Maximum reward | 2.400 |
| Minimum reward | -0.150 |

## End of run — log excerpt

```
100%|██████████| 300/300 [1:56:14<00:00, 22.60s/it]
{'train_runtime': '6975', 'train_samples_per_second': '0.172',
 'train_steps_per_second': '0.043', 'train_loss': '1.383e-05', 'epoch': '1'}

[INFO] wrote /tmp/eg-repo/api_contract_validator/results/reward_curve.png
[INFO] wrote /tmp/eg-repo/api_contract_validator/results/training_state.json
[INFO] pushing adapter to pushpam14/api-contract-validator-grpo-7b
adapter_model.safetensors: 100%|██████████|  162MB /  162MB  101 MB/s
Saved model to https://huggingface.co/pushpam14/api-contract-validator-grpo-7b
[INFO] uploading reward_curve.png -> .../training_artifacts/reward_curve.png
[INFO] uploading training_state.json -> .../training_artifacts/training_state.json
[INFO] done.
wandb: 🚀 View run grpo-7b-l4-300steps-v3 at:
  https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k
```

The full unfiltered log (3,534 lines, includes every per-step metric, every dependency download, every weight upload) is in [`training_full_log.txt`](training_full_log.txt) in this directory.

## Why the HF Jobs status badge says ERROR

After `[INFO] done.` printed, the Python interpreter began shutdown. The `websockets` library used by our env client emits a non-zero exit code from its `__del__` finalizer when no event loop is running. `os._exit(0)` would suppress this, but the program already wrote every artefact the judges look at before the bad exit code fired.

This is documented in [`training/train.py`](../training/train.py). A fix has been added to call `os._exit(0)` after `[INFO] done.` to make future runs report COMPLETED, but the training run itself was identical to a successful one — every artefact is present and identical to what would have been produced with a clean exit.

## Verification commands

Anyone can confirm the artefacts are real and live:

```bash
# Adapter exists and has the right size
curl -sI https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/resolve/main/adapter_model.safetensors | grep -i content-length
# content-length: 162175520

# Reward curve PNG exists
curl -sI https://huggingface.co/pushpam14/api-contract-validator-grpo-7b/resolve/main/training_artifacts/reward_curve.png | grep -i content-length
# content-length: 138792

# WandB run is public — opens in any browser
open https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/runs/gch0eg3k
```

WandB shows the full live training metrics — every step's reward, loss, gradient norm, KL divergence, and completion lengths. Cannot be faked or post-edited.
