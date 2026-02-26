#!/bin/bash
cd /root/Ukiyo-e_NER
source .venv/bin/activate
export $(grep -v "^#" .env | xargs)
export WANDB_DISABLED=true PYTHONPATH=/root/Ukiyo-e_NER

echo "[$(date)] Starting two-stage training..."
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh --config scripts/local_llm/config_llama_8b_ja.json

echo "[$(date)] Starting fewshot-only training..."
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh --config scripts/local_llm/config_llama_8b_ja_fewshot_only.json

echo "[$(date)] All done!"
