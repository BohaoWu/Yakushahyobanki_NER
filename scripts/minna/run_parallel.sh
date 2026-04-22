#!/bin/bash
# ============================================================================
# Run all Minna NER experiments in parallel using GPU 0 + GPU 1
#
# Pipeline:
#   1. Start synth gen for ChatGPT and Claude in PARALLEL (different APIs)
#   2. GPU 0 worker: Direct → One-stage → ChatGPT Two-stage → ChatGPT Three-stage
#   3. GPU 1 worker: Claude Two-stage → Claude Three-stage (waits for synth)
#
# Usage:
#   bash scripts/minna/run_parallel.sh                    # Run everything
#   bash scripts/minna/run_parallel.sh --no-three-stage   # Skip three-stage
#   bash scripts/minna/run_parallel.sh --status           # Check running jobs
#   bash scripts/minna/run_parallel.sh --stop             # Stop all jobs
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_BASE="logs/minna_parallel"
PID_FILE="$LOG_BASE/pids.txt"

# ----- Status / Stop commands -----
if [ "$1" = "--status" ]; then
    if [ ! -f "$PID_FILE" ]; then
        echo "No active run found"
        exit 0
    fi
    echo "Active processes:"
    while IFS=$'\t' read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  $name [PID $pid] running"
        else
            echo "  $name [PID $pid] DONE/CRASHED"
        fi
    done < "$PID_FILE"
    echo
    echo "Logs in: $LOG_BASE/"
    ls -la "$LOG_BASE/"*.log 2>/dev/null
    exit 0
fi

if [ "$1" = "--stop" ]; then
    if [ ! -f "$PID_FILE" ]; then
        echo "No active run found"
        exit 0
    fi
    echo "Stopping all processes..."
    while IFS=$'\t' read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing $name [PID $pid]"
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "Done"
    exit 0
fi

# ----- Parse args -----
SKIP_THREE_STAGE=false
N_VALUES="5 10 20 50 100"
for arg in "$@"; do
    case "$arg" in
        --no-three-stage) SKIP_THREE_STAGE=true ;;
        --n-values) shift; N_VALUES="$2"; shift ;;
    esac
done

# Setup
mkdir -p "$LOG_BASE"
> "$PID_FILE"

# Load env (for API keys)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export WANDB_DISABLED=true

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_pid() {
    echo -e "$1\t$2" >> "$PID_FILE"
}

wait_for_synth() {
    # $1 = provider (chatgpt|claude), $2 = N value
    local file="dataset/minna_synthetic_topn${2}_uniform_${1}/generated_ner_uniform_multi.json"
    while [ ! -f "$file" ]; do sleep 30; done
}

# ============================================================
# Phase 1: Start synthetic data generation in parallel
# ============================================================
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}Starting Minna parallel pipeline${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo "Log dir: $LOG_BASE"
echo "Skip three-stage: $SKIP_THREE_STAGE"
echo "N values: $N_VALUES"
echo

echo -e "${GREEN}[1/3] Starting synthetic data generation (ChatGPT + Claude in parallel)${NC}"

# Convert space-separated N_VALUES to comma-separated for generate_synthetic.sh
N_VALUES_COMMA=$(echo "$N_VALUES" | tr ' ' ',')

nohup bash scripts/minna/generate_synthetic.sh chatgpt "$N_VALUES_COMMA" > "$LOG_BASE/synth_chatgpt.log" 2>&1 &
SYNTH_CHATGPT_PID=$!
log_pid "synth_chatgpt" "$SYNTH_CHATGPT_PID"

nohup bash scripts/minna/generate_synthetic.sh claude "$N_VALUES_COMMA" > "$LOG_BASE/synth_claude.log" 2>&1 &
SYNTH_CLAUDE_PID=$!
log_pid "synth_claude" "$SYNTH_CLAUDE_PID"

echo "  ChatGPT synth gen: PID $SYNTH_CHATGPT_PID"
echo "  Claude synth gen:  PID $SYNTH_CLAUDE_PID"

# ============================================================
# Phase 2: GPU 0 worker
# ============================================================
echo -e "${GREEN}[2/3] Starting GPU 0 worker (ChatGPT branch)${NC}"

cat > "$LOG_BASE/_gpu0_worker.sh" << WORKER_EOF
#!/bin/bash
set -e
cd "$PROJECT_ROOT"
source .venv/bin/activate
export WANDB_DISABLED=true

START=\$(date +%s)
echo "[GPU0] === Direct ===" \$(date)
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh --config scripts/minna/config_minna_direct.json --gpu 0

echo "[GPU0] === One-stage MLM ===" \$(date)
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh --config scripts/minna/config_minna_one_stage.json --gpu 0

# Two-stage ChatGPT (wait for each synth as needed)
for N in $N_VALUES; do
    echo "[GPU0] Waiting for chatgpt N=\$N synth..."
    while [ ! -f "dataset/minna_synthetic_topn\${N}_uniform_chatgpt/generated_ner_uniform_multi.json" ]; do
        sleep 30
    done
    echo "[GPU0] === Two-stage chatgpt N=\$N ===" \$(date)
    CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh \\
        --config scripts/minna/config_minna_two_stage_chatgpt.json --topn \$N --gpu 0
done

if [ "$SKIP_THREE_STAGE" = "false" ]; then
    for N in $N_VALUES; do
        echo "[GPU0] === Three-stage chatgpt N=\$N ===" \$(date)
        CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh \\
            --config scripts/minna/config_minna_three_stage_chatgpt.json --topn \$N --gpu 0
    done
fi

END=\$(date +%s)
echo "[GPU0] DONE in \$((END - START))s = \$(((END - START) / 60))m"
WORKER_EOF
chmod +x "$LOG_BASE/_gpu0_worker.sh"

nohup bash "$LOG_BASE/_gpu0_worker.sh" > "$LOG_BASE/gpu0.log" 2>&1 &
GPU0_PID=$!
log_pid "gpu0_worker" "$GPU0_PID"
echo "  GPU 0 worker:      PID $GPU0_PID"

# ============================================================
# Phase 3: GPU 1 worker
# ============================================================
echo -e "${GREEN}[3/3] Starting GPU 1 worker (Claude branch)${NC}"

cat > "$LOG_BASE/_gpu1_worker.sh" << WORKER_EOF
#!/bin/bash
set -e
cd "$PROJECT_ROOT"
source .venv/bin/activate
export WANDB_DISABLED=true

START=\$(date +%s)

# Two-stage Claude (wait for each synth as needed)
for N in $N_VALUES; do
    echo "[GPU1] Waiting for claude N=\$N synth..."
    while [ ! -f "dataset/minna_synthetic_topn\${N}_uniform_claude/generated_ner_uniform_multi.json" ]; do
        sleep 30
    done
    echo "[GPU1] === Two-stage claude N=\$N ===" \$(date)
    CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh \\
        --config scripts/minna/config_minna_two_stage_claude.json --topn \$N --gpu 1
done

if [ "$SKIP_THREE_STAGE" = "false" ]; then
    for N in $N_VALUES; do
        echo "[GPU1] === Three-stage claude N=\$N ===" \$(date)
        CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh \\
            --config scripts/minna/config_minna_three_stage_claude.json --topn \$N --gpu 1
    done
fi

END=\$(date +%s)
echo "[GPU1] DONE in \$((END - START))s = \$(((END - START) / 60))m"
WORKER_EOF
chmod +x "$LOG_BASE/_gpu1_worker.sh"

nohup bash "$LOG_BASE/_gpu1_worker.sh" > "$LOG_BASE/gpu1.log" 2>&1 &
GPU1_PID=$!
log_pid "gpu1_worker" "$GPU1_PID"
echo "  GPU 1 worker:      PID $GPU1_PID"

echo
echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}All jobs launched in background${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo "Monitor progress:"
echo "  bash scripts/minna/run_parallel.sh --status"
echo "  tail -f $LOG_BASE/gpu0.log"
echo "  tail -f $LOG_BASE/gpu1.log"
echo "  tail -f $LOG_BASE/synth_chatgpt.log"
echo "  tail -f $LOG_BASE/synth_claude.log"
echo
echo "Stop all jobs:"
echo "  bash scripts/minna/run_parallel.sh --stop"
