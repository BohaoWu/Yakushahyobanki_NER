#!/bin/bash
cd /root/Ukiyo-e_NER
source .venv/bin/activate
export $(grep -v "^#" .env | xargs)
export WANDB_DISABLED=true PYTHONPATH=/root/Ukiyo-e_NER

echo "[$(date)] Starting FR one-stage zero-shot..."
bash scripts/run.sh --config scripts/multilingual/config_newseye_fr_zero_shot.json

echo "[$(date)] Starting FR two-stage zero-shot..."
bash scripts/run.sh --config scripts/multilingual/config_newseye_fr_two_stage_zero_shot.json

echo "[$(date)] All FR done!"
