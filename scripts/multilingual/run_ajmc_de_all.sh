#!/bin/bash
cd /root/Ukiyo-e_NER
source .venv/bin/activate
export $(grep -v "^#" .env | xargs)
export WANDB_DISABLED=true PYTHONPATH=/root/Ukiyo-e_NER

echo "[$(date)] Starting AJMC DE one-stage zero-shot (English BERT added)..."
bash scripts/run.sh --config scripts/multilingual/config_ajmc_de_zero_shot.json

echo "[$(date)] Starting AJMC DE two-stage zero-shot..."
bash scripts/run.sh --config scripts/multilingual/config_ajmc_de_two_stage_zero_shot.json

echo "[$(date)] All AJMC DE done!"
