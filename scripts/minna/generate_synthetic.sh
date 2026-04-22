#!/bin/bash
# ============================================================================
# Generate synthetic NER data for Minna dataset using ChatGPT and Claude
#
# Usage:
#   bash scripts/minna/generate_synthetic.sh                  # Generate all (5 N values × 2 LLMs = 10 datasets)
#   bash scripts/minna/generate_synthetic.sh chatgpt          # Only ChatGPT
#   bash scripts/minna/generate_synthetic.sh claude           # Only Claude
#   bash scripts/minna/generate_synthetic.sh chatgpt 20       # Only ChatGPT N=20
#   bash scripts/minna/generate_synthetic.sh claude 5,10,20   # Claude with custom N values
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

source .venv/bin/activate

# Default config
PROVIDERS=${1:-"chatgpt,claude"}
N_VALUES=${2:-"5,10,20,50,100"}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-2000}

# Models
CHATGPT_MODEL="gpt-4o-mini"
CLAUDE_MODEL="claude-3-haiku-20240307"

# Workers
CHATGPT_WORKERS=10
CLAUDE_WORKERS=5

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

generate_one() {
    local provider=$1
    local n=$2
    local model
    local workers
    local extra_args=""
    if [ "$provider" == "chatgpt" ]; then
        model="$CHATGPT_MODEL"
        workers="$CHATGPT_WORKERS"
        local provider_arg="openai"
    else
        model="$CLAUDE_MODEL"
        workers="$CLAUDE_WORKERS"
        local provider_arg="claude"
        extra_args="--use-batch"  # Use Message Batches API for Claude (bypass rate limits)
    fi

    local output_dir="dataset/minna_synthetic_topn${n}_uniform_${provider}"

    if [ -f "${output_dir}/generated_ner_uniform_multi.json" ]; then
        echo -e "${YELLOW}Skipping (already exists): ${output_dir}${NC}"
        return
    fi

    echo -e "\n${BLUE}========================================================================${NC}"
    echo -e "${GREEN}Generating: ${provider} N=${n} → ${output_dir}${NC}"
    echo -e "${BLUE}========================================================================${NC}"

    python3 src/data_generation/generate_balanced.py \
        --source dataset/minna \
        --output "$output_dir" \
        --uniform-multi \
        --top-n "$n" \
        --total-samples "$TOTAL_SAMPLES" \
        --provider "$provider_arg" \
        --model "$model" \
        --lang ja_minna \
        --workers "$workers" \
        $extra_args
}

# Parse providers and N values
IFS=',' read -ra PROVIDER_LIST <<< "$PROVIDERS"
IFS=',' read -ra N_LIST <<< "$N_VALUES"

START_TIME=$(date +%s)

for provider in "${PROVIDER_LIST[@]}"; do
    for n in "${N_LIST[@]}"; do
        generate_one "$provider" "$n"
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "\n${GREEN}========================================================================${NC}"
echo -e "${GREEN}All synthetic data generation completed in $((ELAPSED / 60))m $((ELAPSED % 60))s${NC}"
echo -e "${GREEN}========================================================================${NC}"
