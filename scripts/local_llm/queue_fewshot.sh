#!/bin/bash
cd /root/Ukiyo-e_NER

# Wait for current training (PID 3238631) to finish
echo "[$(date)] Waiting for two-stage training to finish..."
while kill -0 3238631 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Two-stage training finished, starting few-shot only..."
source .venv/bin/activate
export $(grep -v "^#" .env | xargs)
CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true PYTHONPATH=/root/Ukiyo-e_NER \
    bash scripts/run.sh --config scripts/local_llm/config_llama_8b_ja_fewshot_only.json
