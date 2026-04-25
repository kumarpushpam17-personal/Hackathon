# Training — Enterprise Contract Guardian

Re-runnable training pipeline for the API Contract Validator environment, using GRPO from TRL with LoRA adapters via Unsloth.

## Files

| File | Purpose |
|---|---|
| `baseline.py` | Run the untrained model on every task; write `baseline_scores.json` |
| `train.py` | GRPO training loop — connects to env, rolls out, trains LoRA |
| `plot.py` | Build `reward_curve.png` and `before_after.png` for the README |
| `grpo_colab.ipynb` | One-click Colab notebook (open in Colab → Runtime → Run all) |

## Recommended path — HF Jobs (best for the finale)

HF Jobs runs in the cloud, doesn't disconnect, and bills against your $30 hackathon credit. Three runs total cost ~$3 of $60 if you have credits across two accounts.

### Run 1 — Smoke test (~$0.30, 5 min)

Verifies the pipeline works end-to-end before committing to a long run.

```bash
hf jobs uv run \
    --with "trl" --with "unsloth" --with "openenv-core[core]>=0.2.2" \
    --with "wandb" --with "matplotlib" --with "datasets" --with "openai" \
    --flavor t4-small \
    -s HF_TOKEN -s WANDB_API_KEY \
    -e BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
    -e MAX_STEPS=10 \
    -e WANDB_RUN=smoke-test \
    -- python training/train.py
```

If this errors, **don't proceed**. Fix the error, re-run smoke test until it returns clean.

### Run 2 — Main training, **Qwen2.5-7B on L4** (~$2.40, ~2 hours)

Best balance of model size, speed, and cost for our $60 budget. L4 has 24 GB which fits Qwen2.5-7B with 4-bit quantisation + LoRA r=16.

```bash
hf jobs uv run \
    --with "trl" --with "unsloth" --with "openenv-core[core]>=0.2.2" \
    --with "wandb" --with "matplotlib" --with "datasets" --with "openai" \
    --flavor l4x1 \
    -s HF_TOKEN -s WANDB_API_KEY \
    -e BASE_MODEL=unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
    -e MAX_STEPS=300 \
    -e NUM_GENERATIONS=4 \
    -e LORA_R=16 \
    -e LORA_ALPHA=32 \
    -e WANDB_PROJECT=openenv-contract-guardian \
    -e WANDB_RUN=grpo-7b-l4-300steps \
    -e PUSH_TO_HUB=pushpam14/api-contract-validator-grpo-7b \
    -- python training/train.py
```

### Run 3 — Insurance run on second account (~$0.40, ~45 min)

Use your second HF account in parallel as a safety net. Smaller model = faster, more dramatic improvement curve. If Run 2 produces a beautiful curve we ship that; if Run 2 has issues we ship this one.

```bash
# Use your SECOND HF account's token here
hf jobs uv run \
    --with "trl" --with "unsloth" --with "openenv-core[core]>=0.2.2" \
    --with "wandb" --with "matplotlib" --with "datasets" --with "openai" \
    --flavor t4-small \
    -s HF_TOKEN -s WANDB_API_KEY \
    -e BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
    -e MAX_STEPS=200 \
    -e WANDB_RUN=grpo-1.5b-t4-200steps \
    -e PUSH_TO_HUB=YOUR_SECOND_ACCOUNT/api-contract-validator-grpo-1.5b \
    -- python training/train.py
```

### Hardware ↔ model size cheatsheet

| Flavor | VRAM | $/hr | Fits (4-bit + LoRA) |
|---|---|---|---|
| `t4-small` | 16 GB | $0.40 | Up to 3B |
| `l4x1` | 24 GB | $0.80 | Up to 8B comfortably |
| `a10g-large` | 24 GB | $1.50 | Up to 8B, faster than L4 |
| `a100-large` | 80 GB | $3.50 | 14B fp16 or 70B 4-bit |
| `h100x1` | 80 GB | $4.50 | Same as A100 but ~2× faster |

For our env, **`l4x1` + Qwen2.5-7B is the sweet spot.**

### Monitoring the run

```bash
# Watch the job logs in real time
hf jobs logs <job-id> --follow

# List recent jobs
hf jobs list

# WandB run will be auto-linked in the job logs — bookmark that URL for the README
```

## Alternative: Colab notebook (if HF Jobs is unavailable)

Open `grpo_colab.ipynb` in Colab. Set `HF_TOKEN` and `WANDB_API_KEY` in the secrets pane. Hit **Runtime → Run all**. Free T4, but disconnects after 3 hours and only fits the 1.5B model.

## Alternative: Local GPU

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
| `BASE_MODEL` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 7B fits on L4 (24 GB) with 4-bit |
| `ENV_URL` | `http://localhost:7860` | Local server or deployed HF Space |
| `MAX_STEPS` | `300` | GRPO steps. ~2 hours on L4 |
| `NUM_GENERATIONS` | `4` | Completions per prompt for relative ranking |
| `LORA_R` | `16` | LoRA rank |
| `PUSH_TO_HUB` | — | `<username>/<repo>` — push trained adapter |

## What the judges look at

The hackathon's "Improvement in Rewards" 20% criterion explicitly asks for a **before vs after comparison**. Per the official Q&A:

> "You're expected to show before vs after behavior. Run inference using both models and include the comparison (metrics, rewards, or outputs) in the README."

So after training, you must commit ALL FOUR of these:

```
baseline_scores.json                                     # already committed (the "before")
trained_scores.json                                      # post-training inference output
api_contract_validator/results/reward_curve.png          # GRPO training curve
api_contract_validator/results/before_after.png          # bar chart comparison
```

## Post-training commit checklist

Once the Colab notebook finishes, run this on your laptop:

```bash
cd ~/work/hackathon/hackathon-api-contract-validator

# 1. Pull the four artifacts down from Colab into local repo
#    (Colab File pane → right-click → Download for each)
#
#    Place them at:
#      ./trained_scores.json
#      ./api_contract_validator/results/reward_curve.png
#      ./api_contract_validator/results/before_after.png

# 2. Edit api_contract_validator/README.md
#    - Replace each "_(after training)_" placeholder in the Before vs After
#      table with the score from trained_scores.json
#    - Uncomment the two `<!-- ![](results/...) -->` lines so the plots render
#    - Add the WandB run URL in the Links section (and to the WandB row in
#      the Training Results table)

# 3. Run validation
PYTHONPATH=api_contract_validator python3 -m pytest \
    api_contract_validator/tests/test_environment.py -q

# 4. Commit + push
git add trained_scores.json \
        api_contract_validator/results/*.png \
        api_contract_validator/README.md
git commit -m "Add post-training results: trained scores + reward plots"
git push
```

That single push is the "after" half of the before-vs-after evidence judges grade.
