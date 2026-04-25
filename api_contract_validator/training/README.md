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
