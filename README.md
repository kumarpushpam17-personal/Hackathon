# Enterprise Contract Guardian

OpenEnv Hackathon submission for training LLM agents to reason about API contract blast radius across microservices.

The actual OpenEnv environment package is here:

**[`api_contract_validator/`](api_contract_validator/)**

Start with the full submission README:

**[`api_contract_validator/README.md`](api_contract_validator/README.md)**

## Quick Links

| Resource | Link |
|---|---|
| Live Hugging Face Space | https://huggingface.co/spaces/pushpam14/api-contract-validator |
| Live endpoint | https://pushpam14-api-contract-validator.hf.space |
| HF mini-blog writeup | [`api_contract_validator/BLOG.md`](api_contract_validator/BLOG.md) |
| Training proof + logs | [`api_contract_validator/results/TRAINING_RUN_PROOF.md`](api_contract_validator/results/TRAINING_RUN_PROOF.md) |
| Trained adapter | https://huggingface.co/pushpam14/api-contract-validator-grpo-7b |
| WandB training report | https://wandb.ai/pushpamsubscriptions-inn/openenv-contract-guardian/reports/Enterprise-Contract-Guardian-GRPO-training-Qwen-7B-LoRA-300-steps---VmlldzoxNjY3MTAxMA?accessToken=3dhumexjta1umyk04rq6dx47iww4t25utt3j0x7063b7pvzzibp8jah29grhlwpb |

## Headline Result

On `detect_breaking_changes`, both untrained Qwen2.5-72B and untrained Qwen2.5-7B scored `0.01`. After 300 GRPO steps, Qwen2.5-7B + LoRA scored `0.67`.

That is the core result: the OpenEnv reward signal taught a targeted API blast-radius reasoning behavior that model scale alone did not solve.
