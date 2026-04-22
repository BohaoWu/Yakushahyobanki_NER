#!/bin/bash
# ============================================================================
# Run all Minna NER training experiments
#
# Usage:
#   bash scripts/minna/run_all.sh                # Run all stages × all configs
#   bash scripts/minna/run_all.sh direct         # Direct only
#   bash scripts/minna/run_all.sh one_stage      # One-stage MLM only
#   bash scripts/minna/run_all.sh two_stage      # Two-stage (synth × 2 LLMs × N values)
#   bash scripts/minna/run_all.sh three_stage    # Three-stage (MLM + synth × 2 LLMs × N values)
#   bash scripts/minna/run_all.sh chatgpt        # Direct + One + Two + Three (ChatGPT only)
#   bash scripts/minna/run_all.sh claude         # Direct + One + Two + Three (Claude only)
#   bash scripts/minna/run_all.sh minimal        # Direct + One + Two/Three N=20 (both LLMs)
#
# Environment:
#   GPU=0  CUDA_VISIBLE_DEVICES (default: 0)
#   N_VALUES="5,10,20,50,100"  Top-N values to run (default: all)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE=${1:-"all"}
GPU=${GPU:-0}
N_VALUES=${N_VALUES:-"5,10,20,50,100"}

export WANDB_DISABLED=true

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
}

run_direct() {
    print_header "[Direct] No pretraining, 7 models"
    CUDA_VISIBLE_DEVICES=$GPU bash scripts/run.sh \
        --config scripts/minna/config_minna_direct.json --gpu $GPU
}

run_one_stage() {
    print_header "[One-stage] MLM pretrain only, 7 models"
    CUDA_VISIBLE_DEVICES=$GPU bash scripts/run.sh \
        --config scripts/minna/config_minna_one_stage.json --gpu $GPU
}

run_two_stage() {
    local provider=$1
    local n=$2
    local syn_dir="dataset/minna_synthetic_topn${n}_uniform_${provider}"
    if [ ! -d "$syn_dir" ]; then
        echo -e "${YELLOW}Skipping two-stage ${provider} N=${n}: synthetic data not found at ${syn_dir}${NC}"
        return
    fi
    print_header "[Two-stage] ${provider} N=${n}"
    CUDA_VISIBLE_DEVICES=$GPU bash scripts/run.sh \
        --config scripts/minna/config_minna_two_stage_${provider}.json --topn $n --gpu $GPU
}

run_three_stage() {
    local provider=$1
    local n=$2
    local syn_dir="dataset/minna_synthetic_topn${n}_uniform_${provider}"
    if [ ! -d "$syn_dir" ]; then
        echo -e "${YELLOW}Skipping three-stage ${provider} N=${n}: synthetic data not found at ${syn_dir}${NC}"
        return
    fi
    print_header "[Three-stage] MLM + ${provider} N=${n}"
    CUDA_VISIBLE_DEVICES=$GPU bash scripts/run.sh \
        --config scripts/minna/config_minna_three_stage_${provider}.json --topn $n --gpu $GPU
}

START_TIME=$(date +%s)

case "$MODE" in
    direct)
        run_direct
        ;;
    one_stage)
        run_one_stage
        ;;
    two_stage)
        IFS=',' read -ra N_LIST <<< "$N_VALUES"
        for provider in chatgpt claude; do
            for n in "${N_LIST[@]}"; do
                run_two_stage "$provider" "$n"
            done
        done
        ;;
    three_stage)
        IFS=',' read -ra N_LIST <<< "$N_VALUES"
        for provider in chatgpt claude; do
            for n in "${N_LIST[@]}"; do
                run_three_stage "$provider" "$n"
            done
        done
        ;;
    chatgpt)
        run_direct
        run_one_stage
        IFS=',' read -ra N_LIST <<< "$N_VALUES"
        for n in "${N_LIST[@]}"; do
            run_two_stage chatgpt "$n"
            run_three_stage chatgpt "$n"
        done
        ;;
    claude)
        run_direct
        run_one_stage
        IFS=',' read -ra N_LIST <<< "$N_VALUES"
        for n in "${N_LIST[@]}"; do
            run_two_stage claude "$n"
            run_three_stage claude "$n"
        done
        ;;
    minimal)
        run_direct
        run_one_stage
        run_two_stage chatgpt 20
        run_two_stage claude 20
        run_three_stage chatgpt 20
        run_three_stage claude 20
        ;;
    all|"")
        run_direct
        run_one_stage
        IFS=',' read -ra N_LIST <<< "$N_VALUES"
        for provider in chatgpt claude; do
            for n in "${N_LIST[@]}"; do
                run_two_stage "$provider" "$n"
                run_three_stage "$provider" "$n"
            done
        done
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Valid modes: all | direct | one_stage | two_stage | three_stage | chatgpt | claude | minimal"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "\n${GREEN}========================================================================${NC}"
echo -e "${GREEN}All Minna training completed in $((ELAPSED / 60))m $((ELAPSED % 60))s${NC}"
echo -e "${GREEN}========================================================================${NC}"
