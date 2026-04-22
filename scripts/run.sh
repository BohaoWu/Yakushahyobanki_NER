#!/bin/bash

# ====================================================================================================
# Configuration-driven unified training script
# Configure experiment parameters by modifying the config JSON files
# ====================================================================================================

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${SCRIPT_DIR}/config_quick_test.json"

# Parse command line arguments
SHOW_HELP=false
DRY_RUN=false
GPU_OVERRIDE=""
CORPUS_SIZE_OVERRIDE=""
CORPUS_STEP=""
CORPUS_MAX=""
TOPN_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpu)
            GPU_OVERRIDE="$2"
            shift 2
            ;;
        --corpus-size)
            CORPUS_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --corpus-step)
            CORPUS_STEP="$2"
            shift 2
            ;;
        --corpus-max)
            CORPUS_MAX="$2"
            shift 2
            ;;
        --topn)
            TOPN_OVERRIDE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown argument '$1'${NC}"
            echo "Use --help to see usage information"
            exit 1
            ;;
    esac
done

# Display help information
if [ "$SHOW_HELP" = true ]; then
    cat << EOF
${GREEN}Configuration-driven unified training script${NC}

Usage:
  ./scripts/run.sh [options]

Options:
  --help, -h            Display this help information
  --config FILE         Specify configuration file (default: scripts/config_quick_test.json)
  --gpu DEVICE          Force use specific GPU (sequential execution, skips auto-scheduling)
  --corpus-size SIZE    Specify MLM pre-training corpus size (single run)
  --topn N              Specify topN value for template configs (replaces \${topn} placeholder)
  --corpus-step STEP    Specify corpus step size, automatically run multiple experiments (requires --corpus-max)
  --corpus-max MAX      Specify maximum corpus size (used together with --corpus-step)
  --dry-run             Only display commands to be executed, without actually running them

Configuration file:
  The configuration file uses JSON format and contains the following main sections:
  - experiment_name: Experiment name
  - model: Model configuration (name, CRF, LoRA, etc.)
  - data: Dataset and training epochs
  - training: Training parameters (batch size, learning rate, etc.)
  - mlm_pretrain: MLM pre-training configuration
  - output: Output directory
  - GPU auto-scheduling: automatically detects free GPUs

Examples:
  # Quick test (default config)
  ./scripts/run.sh

  # Use specific configuration file
  ./scripts/run.sh --config scripts/config_one_stage_slm.json

  # Use template config with topN value
  ./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 5

  # Specify GPU and single corpus size
  ./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --corpus-size 40000 --gpu 1

  # Batch run MLM corpus size experiments (step 20k, 0 to 480k)
  ./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --corpus-step 20000 --corpus-max 480000

  # Only view the commands to be executed
  ./scripts/run.sh --dry-run

More information:
  See scripts/README.md and scripts/COMMANDS.md
EOF
    exit 0
fi

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file does not exist: $CONFIG_FILE${NC}"
    exit 1
fi

# Handle corpus step batch run
if [ -n "$CORPUS_STEP" ]; then
    if [ -z "$CORPUS_MAX" ]; then
        echo -e "${RED}Error: --corpus-max must also be specified when using --corpus-step${NC}"
        exit 1
    fi

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}MLM corpus size batch experiment${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Config: ${CONFIG_FILE}"
    echo -e "Step size: ${CORPUS_STEP}"
    echo -e "Maximum: ${CORPUS_MAX}"
    [ -n "$GPU_OVERRIDE" ] && echo -e "GPU device: ${GPU_OVERRIDE}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Generate timestamp (shared by all experiments)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_BASE_DIR="experiments/mlm_corpus_step${CORPUS_STEP}_${TIMESTAMP}"

    # Calculate number of experiments
    CORPUS_SIZE=0
    EXPERIMENT_COUNT=0
    while [ $CORPUS_SIZE -le $CORPUS_MAX ]; do
        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        CORPUS_SIZE=$((CORPUS_SIZE + CORPUS_STEP))
    done

    echo -e "Will run ${EXPERIMENT_COUNT} experiments (0, ${CORPUS_STEP}, $((CORPUS_STEP * 2)), ..., ${CORPUS_MAX})"
    echo ""

    # Record total start time
    TOTAL_START=$(date +%s)
    SUCCESS_COUNT=0
    FAILED_COUNT=0

    # Loop through experiments
    CORPUS_SIZE=0
    CURRENT_EXP=1
    while [ $CORPUS_SIZE -le $CORPUS_MAX ]; do
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Experiment ${CURRENT_EXP}/${EXPERIMENT_COUNT}: corpus_size=${CORPUS_SIZE}${NC}"
        echo -e "${GREEN}========================================${NC}"

        # Build command arguments
        RUN_ARGS="--config $CONFIG_FILE --corpus-size $CORPUS_SIZE"
        [ -n "$GPU_OVERRIDE" ] && RUN_ARGS="$RUN_ARGS --gpu $GPU_OVERRIDE"
        [ "$DRY_RUN" = true ] && RUN_ARGS="$RUN_ARGS --dry-run"

        # Record individual experiment start time
        EXP_START=$(date +%s)

        # Run experiment
        if bash "$0" $RUN_ARGS; then
            EXP_END=$(date +%s)
            EXP_ELAPSED=$((EXP_END - EXP_START))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo -e "${GREEN}✓ Experiment ${CURRENT_EXP}/${EXPERIMENT_COUNT} completed (elapsed: $((EXP_ELAPSED / 60))m$((EXP_ELAPSED % 60))s)${NC}"
        else
            EXP_END=$(date +%s)
            EXP_ELAPSED=$((EXP_END - EXP_START))
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo -e "${RED}✗ Experiment ${CURRENT_EXP}/${EXPERIMENT_COUNT} failed (elapsed: $((EXP_ELAPSED / 60))m$((EXP_ELAPSED % 60))s)${NC}"
        fi

        echo ""
        CORPUS_SIZE=$((CORPUS_SIZE + CORPUS_STEP))
        CURRENT_EXP=$((CURRENT_EXP + 1))
    done

    # Calculate total elapsed time
    TOTAL_END=$(date +%s)
    TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
    HOURS=$((TOTAL_ELAPSED / 3600))
    MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
    SECONDS=$((TOTAL_ELAPSED % 60))

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Batch experiments completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Succeeded: ${SUCCESS_COUNT}/${EXPERIMENT_COUNT}"
    echo -e "Failed: ${FAILED_COUNT}/${EXPERIMENT_COUNT}"
    echo -e "Total elapsed: ${HOURS}h${MINUTES}m${SECONDS}s"
    echo -e "${GREEN}========================================${NC}"

    exit 0
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq is not installed, will use Python to parse JSON${NC}"
    USE_PYTHON=true
else
    USE_PYTHON=false
fi

# Function to read JSON configuration
read_config() {
    local key="$1"
    if [ "$USE_PYTHON" = true ]; then
        python3 -c "import json; print(json.load(open('$CONFIG_FILE'))$key)" 2>/dev/null || echo ""
    else
        jq -r "$key // empty" "$CONFIG_FILE" 2>/dev/null || echo ""
    fi
}

# Function to find a free GPU (lowest memory usage below threshold)
# Returns GPU ID or empty string if none available
find_free_gpu() {
    local threshold=${1:-80}
    if ! command -v nvidia-smi &> /dev/null; then
        echo ""
        return
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
    python3 -c "
import sys
best_gpu, best_usage = None, 100.0
for line in sys.stdin:
    parts = line.strip().split(', ')
    if len(parts) != 3:
        continue
    idx, used, total = parts
    usage = float(used) / float(total) * 100
    if usage < $threshold and usage < best_usage:
        best_gpu, best_usage = idx.strip(), usage
print(best_gpu if best_gpu is not None else '')
" 2>/dev/null || echo ""
}

# Get the number of available GPUs
get_num_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return
    fi
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

# Handle ${topn} template variable
if [ -n "$TOPN_OVERRIDE" ]; then
    echo -e "${BLUE}Using topN value: ${TOPN_OVERRIDE}${NC}"
    TEMP_TOPN_CONFIG=$(mktemp)
    sed "s/\${topn}/$TOPN_OVERRIDE/g" "$CONFIG_FILE" > "$TEMP_TOPN_CONFIG"
    CONFIG_FILE="$TEMP_TOPN_CONFIG"
elif grep -q '\${topn}' "$CONFIG_FILE"; then
    echo -e "${RED}Error: Config contains \${topn} placeholder but --topn not specified${NC}"
    echo -e "Usage: ./scripts/run.sh --config <config_file> --topn <value>"
    echo -e "Example: ./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 5"
    exit 1
fi

# Switch to project root directory
cd "$PROJECT_ROOT"

# Read configuration
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}Configuration-driven unified training script${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

EXPERIMENT_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['experiment_name'])")
DESCRIPTION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('description', ''))")

echo -e "${GREEN}Experiment name:${NC} $EXPERIMENT_NAME"
if [ -n "$DESCRIPTION" ]; then
    echo -e "${GREEN}Description:${NC} $DESCRIPTION"
fi
echo ""

# Check configuration format (supports models array)
NUM_MODELS=$(python3 -c "import json; models=json.load(open('$CONFIG_FILE')).get('models', []); print(len(models) if models else 0)")

if [ "$NUM_MODELS" -eq 0 ]; then
    echo -e "${RED}Error: Configuration file must contain a 'models' array${NC}"
    exit 1
fi

echo -e "${GREEN}Detected $NUM_MODELS model(s)${NC}"
echo ""

# Read global configuration (as defaults)
GLOBAL_DATASETS=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE'))['data']; print(','.join(data['datasets']) if isinstance(data['datasets'], list) and isinstance(data['datasets'][0], str) else '')")
GLOBAL_EPOCHS=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE'))['data']; print(','.join(map(str, data['epochs'])) if isinstance(data['epochs'], list) and isinstance(data['epochs'][0], int) else '')")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['batch_size'])")
LEARNING_RATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['learning_rate'])")
SEED=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['seed'])")
MLM_ENABLED=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['mlm_pretrain']['enabled'])")
BASE_OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output']['dir'])")
SAVE_MODEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output'].get('save_model', True))")
LOG_FILE_BASE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['advanced'].get('log_file', ''))")
SPLIT_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['data'].get('split_mode', 'default'))")
EARLY_STOPPING_PATIENCE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training'].get('early_stopping_patience', 0))")

# Detect GPU environment
NUM_GPUS=$(get_num_gpus)

# Handle ${timestamp} placeholder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR//\$\{timestamp\}/$TIMESTAMP}"

# Display global configuration summary
echo -e "${BLUE}Global configuration summary:${NC}"
if [ -n "$GLOBAL_DATASETS" ]; then
    echo -e "  Default datasets: ${YELLOW}$GLOBAL_DATASETS${NC}"
    echo -e "  Default training epochs: ${YELLOW}$GLOBAL_EPOCHS${NC}"
else
    echo -e "  Dataset configuration: ${YELLOW}Configured individually per model${NC}"
fi
echo -e "  Batch Size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  Learning rate: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  Random seed: ${YELLOW}$SEED${NC}"
[ "$MLM_ENABLED" = "True" ] && echo -e "  MLM pre-training: ${GREEN}Enabled${NC}"
echo -e "  Base output directory: ${YELLOW}$BASE_OUTPUT_DIR${NC}"
if [ -n "$GPU_OVERRIDE" ]; then
    echo -e "  GPU device: ${YELLOW}$GPU_OVERRIDE (manual)${NC}"
elif [ "$NUM_GPUS" -gt 0 ]; then
    echo -e "  GPU: ${GREEN}Auto-scheduling across $NUM_GPUS GPU(s)${NC}"
else
    echo -e "  GPU: ${YELLOW}No GPU detected, using CPU${NC}"
fi
echo ""

# ============================================================================
# Phase 1: Build all training commands
# ============================================================================
ALL_RESULT_DIRS=""
declare -a TASK_CMDS
declare -a TASK_NAMES
declare -a TASK_OUTPUT_DIRS

echo -e "${BLUE}Building training commands...${NC}"
echo ""

for MODEL_IDX in $(seq 0 $((NUM_MODELS - 1))); do
    # Read current model configuration and parse model name
    MODEL_NAME_OR_KEY=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX]['name'])")
    CUSTOM_MODEL_ID=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('model_id', ''))")
    USE_CRF=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('use_crf', False))")
    USE_TRANSFORMER_CRF=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('use_transformer_crf', False))")
    USE_LORA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('use_lora', False))")
    USE_4BIT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('use_4bit', False))")
    REMOTE_LLM_PROVIDER=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('provider', ''))")
    N_FEW_SHOT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('n_few_shot', 5))")
    FEW_SHOT_SELECTION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('few_shot_selection', 'stratified'))")

    # Check if the current model has its own independent dataset configuration
    MODEL_DATASETS=$(python3 << EOF
import json
config = json.load(open('$CONFIG_FILE'))
data = config['data']
datasets = data['datasets']
num_models = len(config.get('models', []))
model_idx = $MODEL_IDX
is_nested = (
    isinstance(datasets, list) and
    len(datasets) == num_models and
    len(datasets) > 0 and
    isinstance(datasets[0], list)
)
if is_nested:
    print(','.join(datasets[model_idx]))
else:
    if isinstance(datasets, list) and len(datasets) > 0:
        print(','.join(datasets))
    else:
        print('')
EOF
)

    MODEL_EPOCHS=$(python3 << EOF
import json
config = json.load(open('$CONFIG_FILE'))
data = config['data']
epochs = data['epochs']
num_models = len(config.get('models', []))
model_idx = $MODEL_IDX
is_nested = (
    isinstance(epochs, list) and
    len(epochs) == num_models and
    len(epochs) > 0 and
    isinstance(epochs[0], list)
)
if is_nested:
    print(','.join(map(str, epochs[model_idx])))
else:
    if isinstance(epochs, list) and len(epochs) > 0:
        print(','.join(map(str, epochs)))
    else:
        print('')
EOF
)

    if [ -z "$MODEL_DATASETS" ]; then
        MODEL_DATASETS="$GLOBAL_DATASETS"
    fi
    if [ -z "$MODEL_EPOCHS" ]; then
        MODEL_EPOCHS="$GLOBAL_EPOCHS"
    fi

    # Parse model name
    MODEL_INFO=$(python3 << EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.config.config import MODEL_CONFIGS
model_input = "$MODEL_NAME_OR_KEY"
if '/' in model_input:
    model_name = model_input
    model_key = model_input.split('/')[-1].replace('-', '_')
elif model_input in MODEL_CONFIGS:
    model_name = MODEL_CONFIGS[model_input]['name']
    model_key = model_input
else:
    model_name = model_input
    model_key = model_input.replace('/', '_').replace('-', '_')
print(f"{model_name}|{model_key}")
EOF
)

    MODEL_NAME=$(echo "$MODEL_INFO" | cut -d'|' -f1)
    MODEL_KEY=$(echo "$MODEL_INFO" | cut -d'|' -f2)

    # Set output directory
    if [ "$NUM_MODELS" -eq 1 ]; then
        OUTPUT_DIR="$BASE_OUTPUT_DIR"
    else
        if [ -n "$CUSTOM_MODEL_ID" ]; then
            OUTPUT_DIR="$BASE_OUTPUT_DIR/${CUSTOM_MODEL_ID}"
        elif [ "$MODEL_IDX" -eq 0 ]; then
            OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_KEY}"
        else
            OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_KEY}_${MODEL_IDX}"
        fi
    fi

    # Build training command (without CUDA_VISIBLE_DEVICES - added by scheduler)
    CMD="PYTHONPATH=\"$PROJECT_ROOT\" python3 src/core/train.py"
    CMD="$CMD --model \"$MODEL_NAME\""
    CMD="$CMD --datasets \"$MODEL_DATASETS\""
    CMD="$CMD --epochs \"$MODEL_EPOCHS\""
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --learning-rate $LEARNING_RATE"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""

    if [ "$SPLIT_MODE" != "default" ]; then
        CMD="$CMD --split-mode $SPLIT_MODE"
    fi

    if [ "$EARLY_STOPPING_PATIENCE" != "0" ] && [ -n "$EARLY_STOPPING_PATIENCE" ]; then
        CMD="$CMD --early-stopping-patience $EARLY_STOPPING_PATIENCE"
    fi

    if [ "$USE_CRF" = "True" ]; then
        CMD="$CMD --use-crf"
    fi

    if [ "$USE_TRANSFORMER_CRF" = "True" ]; then
        CMD="$CMD --use-transformer-crf"
    fi

    if [ "$USE_LORA" = "True" ]; then
        CMD="$CMD --use-lora"
        LORA_R=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('lora_r', 16))")
        LORA_ALPHA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('lora_alpha', 32))")
        LORA_DROPOUT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['models'][$MODEL_IDX].get('lora_dropout', 0.05))")
        CMD="$CMD --lora-r $LORA_R --lora-alpha $LORA_ALPHA"
    fi

    if [ "$USE_4BIT" = "True" ]; then
        CMD="$CMD --use-4bit"
    fi

    if [ -n "$REMOTE_LLM_PROVIDER" ]; then
        CMD="$CMD --remote-llm-provider $REMOTE_LLM_PROVIDER"
        CMD="$CMD --n-few-shot $N_FEW_SHOT"
        CMD="$CMD --few-shot-selection $FEW_SHOT_SELECTION"
    fi

    if [ "$MLM_ENABLED" = "True" ]; then
        MLM_CORPUS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['mlm_pretrain']['corpus'])")
        MLM_EPOCHS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['mlm_pretrain']['epochs'])")
        MLM_CORPUS_SIZE=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['mlm_pretrain'].get('corpus_size', 'None'))")
        MLM_BATCH_SIZE=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['mlm_pretrain'].get('batch_size', 'None'))")
        MLM_LEARNING_RATE=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['mlm_pretrain'].get('learning_rate', 'None'))")
        MLM_PROBABILITY=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['mlm_pretrain'].get('mlm_probability', 'None'))")

        if [ -n "$CORPUS_SIZE_OVERRIDE" ]; then
            MLM_CORPUS_SIZE="$CORPUS_SIZE_OVERRIDE"
        fi

        CMD="$CMD --mlm-pretrain --mlm-corpus \"$MLM_CORPUS\" --mlm-epochs $MLM_EPOCHS"
        [ "$MLM_CORPUS_SIZE" != "None" ] && CMD="$CMD --mlm-corpus-size $MLM_CORPUS_SIZE"
        [ "$MLM_BATCH_SIZE" != "None" ] && CMD="$CMD --mlm-batch-size $MLM_BATCH_SIZE"
        [ "$MLM_LEARNING_RATE" != "None" ] && CMD="$CMD --mlm-learning-rate $MLM_LEARNING_RATE"
        [ "$MLM_PROBABILITY" != "None" ] && CMD="$CMD --mlm-probability $MLM_PROBABILITY"
    fi

    if [ "$SAVE_MODEL" = "False" ]; then
        CMD="$CMD --no-save-model"
    fi

    # Store command in task arrays
    TASK_CMDS+=("$CMD")
    TASK_NAMES+=("$MODEL_KEY")
    TASK_OUTPUT_DIRS+=("$OUTPUT_DIR")

    echo -e "  [$((MODEL_IDX + 1))/$NUM_MODELS] ${YELLOW}$MODEL_KEY${NC} -> $OUTPUT_DIR"
done

NUM_TASKS=${#TASK_CMDS[@]}
echo ""
echo -e "${GREEN}Built $NUM_TASKS training task(s)${NC}"
echo ""

# ============================================================================
# Dry run: display all commands and exit
# ============================================================================
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}======================================${NC}"
    echo -e "${YELLOW}Dry run mode: displaying commands${NC}"
    echo -e "${BLUE}======================================${NC}"
    for i in $(seq 0 $((NUM_TASKS - 1))); do
        echo ""
        echo -e "${GREEN}[$((i + 1))/$NUM_TASKS] ${TASK_NAMES[$i]}${NC}"
        if [ -n "$GPU_OVERRIDE" ]; then
            echo -e "${YELLOW}CUDA_VISIBLE_DEVICES=$GPU_OVERRIDE ${TASK_CMDS[$i]}${NC}"
        else
            echo -e "${YELLOW}CUDA_VISIBLE_DEVICES=<auto> ${TASK_CMDS[$i]}${NC}"
        fi
    done
    echo ""
    [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG"
    exit 0
fi

# ============================================================================
# Confirm execution
# ============================================================================
if [ -t 0 ]; then
    if [ -n "$GPU_OVERRIDE" ]; then
        read -p "Start sequential training of $NUM_TASKS models on GPU $GPU_OVERRIDE? (y/n) " -n 1 -r
    elif [ "$NUM_GPUS" -gt 1 ]; then
        read -p "Start parallel training of $NUM_TASKS models across $NUM_GPUS GPUs? (y/n) " -n 1 -r
    else
        read -p "Start training $NUM_TASKS models? (y/n) " -n 1 -r
    fi
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG"
        exit 0
    fi
else
    echo -e "${YELLOW}Non-interactive environment, auto-continuing${NC}"
fi

# ============================================================================
# Phase 2: GPU Queue Scheduler
# ============================================================================
mkdir -p logs
SUCCESS_COUNT=0
FAILED_COUNT=0

if [ -n "$GPU_OVERRIDE" ]; then
    # ---- Manual GPU mode: sequential execution on specified GPU ----
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${GREEN}Sequential execution on GPU $GPU_OVERRIDE${NC}"
    echo -e "${BLUE}======================================${NC}"

    for i in $(seq 0 $((NUM_TASKS - 1))); do
        echo ""
        echo -e "${BLUE}======================================${NC}"
        echo -e "${GREEN}Training model $((i + 1))/$NUM_TASKS: ${TASK_NAMES[$i]}${NC}"
        echo -e "${BLUE}======================================${NC}"

        FULL_CMD="CUDA_VISIBLE_DEVICES=$GPU_OVERRIDE ${TASK_CMDS[$i]}"
        echo -e "${YELLOW}$FULL_CMD${NC}"
        echo ""

        eval $FULL_CMD
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✓ ${TASK_NAMES[$i]} training completed!${NC}"
            ALL_RESULT_DIRS="$ALL_RESULT_DIRS ${TASK_OUTPUT_DIRS[$i]}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "${RED}✗ ${TASK_NAMES[$i]} training failed (exit code: $EXIT_CODE)${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done

elif [ "$NUM_GPUS" -le 1 ]; then
    # ---- Single GPU or CPU: sequential execution with auto-detect ----
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${GREEN}Sequential execution (single GPU/CPU)${NC}"
    echo -e "${BLUE}======================================${NC}"

    for i in $(seq 0 $((NUM_TASKS - 1))); do
        echo ""
        echo -e "${BLUE}======================================${NC}"
        echo -e "${GREEN}Training model $((i + 1))/$NUM_TASKS: ${TASK_NAMES[$i]}${NC}"
        echo -e "${BLUE}======================================${NC}"

        GPU_ID=$(find_free_gpu)
        if [ -n "$GPU_ID" ]; then
            FULL_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID ${TASK_CMDS[$i]}"
        else
            FULL_CMD="${TASK_CMDS[$i]}"
        fi
        echo -e "${YELLOW}$FULL_CMD${NC}"
        echo ""

        eval $FULL_CMD
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✓ ${TASK_NAMES[$i]} training completed!${NC}"
            ALL_RESULT_DIRS="$ALL_RESULT_DIRS ${TASK_OUTPUT_DIRS[$i]}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "${RED}✗ ${TASK_NAMES[$i]} training failed (exit code: $EXIT_CODE)${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done

else
    # ---- Multi-GPU: parallel queue scheduler ----
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${GREEN}Parallel GPU queue scheduler ($NUM_GPUS GPUs)${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""

    # Disable exit-on-error for scheduler (we handle errors manually)
    set +e

    declare -A GPU_PIDS      # gpu_id -> pid
    declare -A GPU_TASK_IDX   # gpu_id -> task index
    declare -A GPU_LOG_FILES  # gpu_id -> log file
    NEXT_TASK=0
    COMPLETED=0

    while [ $COMPLETED -lt $NUM_TASKS ]; do
        # Check for completed jobs
        for gpu_id in "${!GPU_PIDS[@]}"; do
            pid=${GPU_PIDS[$gpu_id]}
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished
                wait "$pid"
                exit_code=$?
                task_idx=${GPU_TASK_IDX[$gpu_id]}
                task_name=${TASK_NAMES[$task_idx]}
                log_file=${GPU_LOG_FILES[$gpu_id]}

                COMPLETED=$((COMPLETED + 1))

                if [ $exit_code -eq 0 ]; then
                    echo -e "${GREEN}✓ [$COMPLETED/$NUM_TASKS] $task_name training completed (GPU $gpu_id)${NC}"
                    ALL_RESULT_DIRS="$ALL_RESULT_DIRS ${TASK_OUTPUT_DIRS[$task_idx]}"
                    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                else
                    echo -e "${RED}✗ [$COMPLETED/$NUM_TASKS] $task_name training failed (GPU $gpu_id, exit code: $exit_code)${NC}"
                    echo -e "  Log: ${YELLOW}$log_file${NC}"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi

                unset GPU_PIDS[$gpu_id]
                unset GPU_TASK_IDX[$gpu_id]
                unset GPU_LOG_FILES[$gpu_id]
            fi
        done

        # Assign tasks to free GPUs
        while [ $NEXT_TASK -lt $NUM_TASKS ]; do
            FREE_GPU=$(find_free_gpu)
            if [ -z "$FREE_GPU" ]; then
                break
            fi

            # Check if this GPU already has a running task
            if [ -n "${GPU_PIDS[$FREE_GPU]+x}" ]; then
                break
            fi

            task_name=${TASK_NAMES[$NEXT_TASK]}
            LOG_FILE="logs/${EXPERIMENT_NAME}_${task_name}.log"
            FULL_CMD="CUDA_VISIBLE_DEVICES=$FREE_GPU ${TASK_CMDS[$NEXT_TASK]}"

            echo -e "${BLUE}[Task $((NEXT_TASK + 1))/$NUM_TASKS] ${GREEN}$task_name${NC} -> GPU $FREE_GPU"
            echo -e "  Log: ${YELLOW}$LOG_FILE${NC}"

            # Launch in background
            eval $FULL_CMD > "$LOG_FILE" 2>&1 &
            GPU_PIDS[$FREE_GPU]=$!
            GPU_TASK_IDX[$FREE_GPU]=$NEXT_TASK
            GPU_LOG_FILES[$FREE_GPU]=$LOG_FILE

            NEXT_TASK=$((NEXT_TASK + 1))
        done

        # If there are still running jobs, wait a bit before checking again
        if [ ${#GPU_PIDS[@]} -gt 0 ]; then
            sleep 10
        fi
    done

    # Re-enable exit-on-error
    set -e
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}All model training completed!${NC}"
echo -e "  Succeeded: ${GREEN}$SUCCESS_COUNT${NC}  Failed: ${RED}$FAILED_COUNT${NC}"
echo -e "${BLUE}======================================${NC}"

# After all model training completed, run evaluation
EVAL_ENABLED=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('evaluation', {}).get('enabled', False))")

if [ "$EVAL_ENABLED" = "True" ]; then
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${GREEN}Starting evaluation and plotting...${NC}"
    echo -e "${BLUE}======================================${NC}"

    # Read evaluation configuration
    EVAL_OUTPUT_DIR=$(python3 -c "import json; d=json.load(open('$CONFIG_FILE')).get('evaluation', {}).get('output_dir', 'evaluation_report'); print(d.replace('\${timestamp}', '$TIMESTAMP'))")
    COMPARE_WITH=$(python3 -c "import json; import sys; cw=json.load(open('$CONFIG_FILE')).get('evaluation', {}).get('compare_with', []); print(' '.join(cw) if cw else '')")

    # Collect all training result files
    RESULT_FILES=""
    for result_dir in $ALL_RESULT_DIRS; do
        if [ -f "$result_dir/result.json" ]; then
            RESULT_FILES="$RESULT_FILES $result_dir/result.json"
        fi
    done

    # Add additional comparison result files
    if [ -n "$COMPARE_WITH" ]; then
        for compare_dir in $COMPARE_WITH; do
            if [ -f "$compare_dir/result.json" ]; then
                RESULT_FILES="$RESULT_FILES $compare_dir/result.json"
            fi
        done
    fi

    # Call evaluation.py
    python3 << EOF
import sys
sys.path.insert(0, 'src/core')
from train import HistoryDocumentResult
from evaluation import ResultEvaluator

result_files = "$RESULT_FILES".split()
print(f"Loading {len(result_files)} result files...")

results = []
for f in result_files:
    try:
        results.append(HistoryDocumentResult.load(f))
        print(f"  ✓ {f}")
    except Exception as e:
        print(f"  ✗ {f}: {e}")

if results:
    evaluator = ResultEvaluator(results)
    evaluator.print_summary()
    evaluator.create_full_report("$EVAL_OUTPUT_DIR")
else:
    print("No results loaded successfully")
EOF

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Evaluation completed!${NC}"
        echo -e "Report saved to: ${YELLOW}$EVAL_OUTPUT_DIR${NC}"
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
    fi
fi

# Clean up temporary config files
[ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG"

echo ""
echo -e "${GREEN}Done!${NC}"

exit ${EXIT_CODE:-0}
