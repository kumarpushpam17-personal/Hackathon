# Training — Enterprise Contract Guardian

Re-runnable training pipeline for the API Contract Validator environment, using GRPO from TRL with LoRA adapters via Unsloth.

## Files

| File | Purpose |
|---|---|
| `baseline.py` | Run the untrained model on every task; write `baseline_scores.json` |
| `train.py` | GRPO training loop — connects to env, rolls out, trains LoRA |
| `plot.py` | Build `reward_curve.png` and `before_after.png` for the README |
| `grpo_colab.ipynb` | One-click Colab notebook (open in Colab → Runtime → Run all) |

## Three ways to run training

### A. Colab notebook (easiest, free GPU)

Open `grpo_colab.ipynb` in Colab. Set `HF_TOKEN` and `WANDB_API_KEY` in the secrets pane. Hit **Runtime → Run all**.

### B. HF Jobs (best for the onsite — uses your $30 credit)

```bash
hf jobs uv run \
    --with trl --with unsloth --with openenv-core --with wandb \
    --flavor t4-small \
    -s HF_TOKEN -s WANDB_API_KEY \
    -- python training/train.py
```

### C. Local GPU

```bash
pip install trl unsloth wandb matplotlib datasets
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
export ENV_URL="http://localhost:7860"   # or your HF Space URL

# 1. Start the env server in another terminal
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. Baseline
python training/baseline.py

# 3. Train
python training/train.py

# 4. Inference with trained adapter
export MODEL_NAME="<your-username>/api-contract-validator-grpo"
export SCORES_OUT_PATH="trained_scores.json"
python inference.py

# 5. Plots
python training/plot.py
```

## Key environment variables

| Variable | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Required. Used for both inference (router) and Hub push |
| `WANDB_API_KEY` | — | Optional. If set, training logs go to WandB |
| `BASE_MODEL` | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | Small enough for T4 |
| `ENV_URL` | `http://localhost:7860` | Local server or deployed HF Space |
| `MAX_STEPS` | `200` | GRPO steps. ~45 min on T4 |
| `NUM_GENERATIONS` | `4` | Completions per prompt for relative ranking |
| `LORA_R` | `16` | LoRA rank |
| `PUSH_TO_HUB` | — | `<username>/<repo>` — push trained adapter |

## What the judges look at

After running, **commit** these to the repo:

```
baseline_scores.json                            # repo root
trained_scores.json                             # repo root
api_contract_validator/results/reward_curve.png
api_contract_validator/results/before_after.png
```

The README's "Training Results" section reads from these files. The plots are evidence of the "Improvement in Rewards" 20% criterion.
