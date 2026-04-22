# Minna NER Experiments

Training scripts and configs for the **Minna** dataset (1855 Ansei Edo Earthquake records).

## Dataset

- **Source**: `dataset/minna/` (built from honkoku-data + ansei2 annotations)
- **Splits**: train=296, val=37, test=37 (zero-shot entity split)
- **Entity types**: location, damage, person, datetime (4 types)

## Files

| File | Purpose |
|------|---------|
| `config_minna_direct.json` | **Direct** — train only on real data |
| `config_minna_one_stage.json` | **One-stage** — MLM pretrain → fine-tune |
| `config_minna_two_stage_chatgpt.json` | **Two-stage** — ChatGPT synth → fine-tune |
| `config_minna_two_stage_claude.json` | **Two-stage** — Claude synth → fine-tune |
| `config_minna_three_stage_chatgpt.json` | **Three-stage** — MLM + ChatGPT synth → fine-tune |
| `config_minna_three_stage_claude.json` | **Three-stage** — MLM + Claude synth → fine-tune |
| `generate_synthetic.sh` | Generate synthetic data via ChatGPT/Claude |
| `run_all.sh` | Run training experiments (single GPU, sequential) |
| `run_parallel.sh` | **Recommended**: Auto-orchestrated 2-GPU parallel pipeline |

## MLM Corpus

`corpus/corpus_minna.txt` (created from minna train+val + honkoku-data v1 raw files):
- 8,451 lines, 5.4M characters
- Same MLM hyperparameters as yakusha (epochs=3, batch=16, lr=5e-5, mlm_prob=0.15)

## Pipeline

### Step 1: Generate synthetic data

```bash
# Generate ALL (5 N values × 2 LLMs = 10 datasets, ~2-3 hours)
bash scripts/minna/generate_synthetic.sh

# Only ChatGPT
bash scripts/minna/generate_synthetic.sh chatgpt

# Only N=20 for both LLMs (~25 minutes)
bash scripts/minna/generate_synthetic.sh chatgpt,claude 20

# Custom: Claude N=10,20,50
bash scripts/minna/generate_synthetic.sh claude 10,20,50
```

Outputs to `dataset/minna_synthetic_topn{N}_uniform_{chatgpt|claude}/generated_ner_uniform_multi.json`.

### Step 2: Run training

```bash
# Run everything: Direct + Two-stage × 5N × 2LLM (~5 hours total)
bash scripts/minna/run_all.sh

# Just Direct (~13 minutes)
bash scripts/minna/run_all.sh direct

# Direct + Two-stage Top-20 (both LLMs) (~1 hour)
bash scripts/minna/run_all.sh minimal

# Direct + ChatGPT only (5 N values)
bash scripts/minna/run_all.sh chatgpt

# Custom N values
N_VALUES="10,20,50" bash scripts/minna/run_all.sh chatgpt

# Run on GPU 1
GPU=1 bash scripts/minna/run_all.sh
```

## Hyperparameters

- num_epochs: **50** (with early stopping patience=5)
- batch_size: 32
- learning_rate: 2e-5
- seed: 42
- Models: BERT, BERT-Large, RoBERTa, LUKE, ALBERT, DeBERTa, BiLSTM (all without CRF)

## Time Estimates

| Task | Time |
|------|------|
| Generate ChatGPT × 1 N value | ~10-15 min |
| Generate Claude × 1 N value | ~15-25 min |
| Direct (7 models) | ~13 min |
| One-stage MLM (7 models) | ~50 min |
| Two-stage 1 config (7 models) | ~25 min |
| Three-stage 1 config (7 models) | ~70 min |
| **Full pipeline (single GPU)** | **~21 hours** |
| **Full pipeline (2 GPU parallel)** | **~10-11 hours** |
| **No three-stage (2 GPU)** | **~5-6 hours** |
| **Minimal pipeline** | **~1.5 hours** |

## Parallel Execution (Recommended)

```bash
# Launch full pipeline (in background, 2 GPUs)
bash scripts/minna/run_parallel.sh

# Skip three-stage (faster, ~5-6h instead of ~10-11h)
bash scripts/minna/run_parallel.sh --no-three-stage

# Check status
bash scripts/minna/run_parallel.sh --status

# Stop all jobs
bash scripts/minna/run_parallel.sh --stop

# Watch logs
tail -f logs/minna_parallel/gpu0.log
tail -f logs/minna_parallel/gpu1.log
```
