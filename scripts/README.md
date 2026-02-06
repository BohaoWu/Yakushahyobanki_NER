# Scripts

Training scripts and configuration files for Yakushahyoubanki NER experiments.

## Quick Start

```bash
cd /root/Yakushahyoubanki_NER

# Quick test (default config, validates all code paths)
./scripts/run.sh

# Use specific config
./scripts/run.sh --config scripts/config_one_stage_slm.json

# Template config with topN value
./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 5

# Preview commands without executing
./scripts/run.sh --dry-run
```

## Directory Structure

```
scripts/
├── README.md
├── run.sh                          # Main training script
├── config_quick_test.json          # Quick validation test (default)
├── config_one_stage_slm.json       # One-stage SLM training
├── chatgpt/                        # ChatGPT synthetic data configs
│   ├── config_two_stage_slm.json
│   └── config_three_stage_slm.json
├── claude/                         # Claude synthetic data configs
│   ├── config_two_stage_slm_claude.json
│   └── config_three_stage_slm_claude.json
├── mlm_finetuning/                 # MLM pretrain + fine-tune configs
│   └── config_mlm_finetuning_100000.json
└── llm_inference/                  # LLM zero/few-shot inference configs
    └── config_llm_zero_shot.json
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `config_quick_test.json` | Quick validation: BERT, BERT+CRF, BiLSTM with MLM + two-stage, 1 epoch |
| `config_one_stage_slm.json` | One-stage training on few-shot data |
| `chatgpt/config_two_stage_slm.json` | Two-stage: ChatGPT synthetic -> few-shot (template, use `--topn`) |
| `chatgpt/config_three_stage_slm.json` | Three-stage: MLM -> ChatGPT synthetic -> few-shot (template, use `--topn`) |
| `claude/config_two_stage_slm_claude.json` | Two-stage: Claude synthetic -> few-shot (template, use `--topn`) |
| `claude/config_three_stage_slm_claude.json` | Three-stage: MLM -> Claude synthetic -> few-shot (template, use `--topn`) |
| `mlm_finetuning/config_mlm_finetuning_100000.json` | Two-stage: MLM pretrain -> few-shot (no synthetic data) |
| `llm_inference/config_llm_zero_shot.json` | ChatGPT/Claude zero-shot & few-shot inference |

## run.sh Options

```
--help, -h            Display help
--config FILE         Configuration file (default: scripts/config_quick_test.json)
--gpu DEVICE          Force specific GPU (sequential execution)
--topn N              TopN value for template configs (replaces ${topn})
--corpus-size SIZE    MLM corpus size override
--corpus-step STEP    Batch corpus size experiments (requires --corpus-max)
--corpus-max MAX      Max corpus size for batch experiments
--dry-run             Preview commands only
```

## GPU Auto-Scheduling

GPU assignment is automatic. No configuration needed.

| Scenario | Behavior |
|----------|----------|
| Single GPU | Sequential execution, auto-detected |
| Multiple GPUs | Parallel queue: tasks dispatched to free GPUs |
| No GPU | Falls back to CPU |
| `--gpu N` | Force sequential execution on GPU N |

## Config JSON Structure

```json
{
  "experiment_name": "experiment_name",
  "description": "Description",

  "models": [
    {
      "name": "bert",
      "model_id": "bert_crf",
      "use_crf": true
    }
  ],

  "data": {
    "datasets": ["dataset/yakusha_annotated_data_few-shot"],
    "epochs": [20]
  },

  "training": {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "seed": 42
  },

  "mlm_pretrain": {
    "enabled": false,
    "corpus": "corpus/corpus_gennbun.txt",
    "corpus_size": 100000,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "mlm_probability": 0.15
  },

  "output": {
    "dir": "experiments/name_${timestamp}",
    "save_model": true
  },

  "evaluation": {
    "enabled": true,
    "output_dir": "experiments/name_${timestamp}/evaluation_report"
  }
}
```

### Model Short Names

| Short Name | Full Path | Type |
|------------|-----------|------|
| `bert` | cl-tohoku/bert-base-japanese-v3 | BERT |
| `bert-large` | cl-tohoku/bert-large-japanese | BERT |
| `roberta` | nlp-waseda/roberta-base-japanese | BERT |
| `luke` | studio-ousia/luke-japanese-base-lite | BERT |
| `albert` | ALINEAR/albert-japanese-v2 | BERT |
| `deberta` | ku-nlp/deberta-v2-base-japanese | BERT |
| `deberta-large` | ku-nlp/deberta-v2-large-japanese | BERT |
| `rinna-3.6b` | rinna/japanese-gpt-neox-3.6b | LLaMA |
| `rinna-1.3b` | rinna/japanese-gpt-1b | LLaMA |
| `bilstm` | BiLSTM | Sequential |

### Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | Model short name or full path | required |
| `model_id` | string | ID for output directory | auto |
| `use_crf` | boolean | CRF layer (BERT-family only) | false |
| `use_lora` | boolean | LoRA (LLaMA only) | false |
| `lora_r` | integer | LoRA rank | 16 |
| `lora_alpha` | integer | LoRA alpha | 32 |
| `use_4bit` | boolean | 4-bit quantization | false |
| `provider` | string | Remote LLM provider (`chatgpt` / `claude`) | - |
| `n_few_shot` | integer | Number of few-shot examples (remote LLM) | 5 |

### Data Configuration

`datasets` and `epochs` arrays must have the same length. For multi-stage training:

```json
{
  "datasets": ["dataset/synthetic_data", "dataset/few_shot_data"],
  "epochs": [10, 20]
}
```

Stage 1 trains on synthetic_data for 10 epochs, then Stage 2 fine-tunes on few_shot_data for 20 epochs.

### Template Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `${timestamp}` | Auto-generated `YYYYMMDD_HHMMSS` | run.sh |
| `${topn}` | TopN value for synthetic datasets | `--topn` argument |

## Usage Examples

### Quick Validation

```bash
# Fastest test: 3 models, 1 epoch, all code paths
./scripts/run.sh
```

### ChatGPT/Claude Synthetic Data Experiments

```bash
# Two-stage with ChatGPT data (all topN values)
for topn in 5 10 20 50 100; do
    ./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn $topn
done

# Three-stage with Claude data
./scripts/run.sh --config scripts/claude/config_three_stage_slm_claude.json --topn 50
```

### MLM Corpus Size Experiments

```bash
# Single corpus size
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --corpus-size 50000

# Batch: step through corpus sizes 0 to 100000
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json \
    --corpus-step 20000 --corpus-max 100000
```

### LLM Zero-Shot Inference

```bash
# Requires OPENAI_API_KEY and ANTHROPIC_API_KEY in .env
./scripts/run.sh --config scripts/llm_inference/config_llm_zero_shot.json
```

### Background Execution

```bash
nohup ./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 50 \
    > logs/two_stage_top50.log 2>&1 &

tail -f logs/two_stage_top50.log
```

## Troubleshooting

**JSON parse error**: Validate with `python3 -m json.tool <config_file>`

**Dataset not found**: Paths are relative to project root `/root/Yakushahyoubanki_NER/`

**CUDA out of memory**: Reduce `batch_size`, or enable `use_4bit` for LLaMA models

**API key missing**: Configure in `/root/Yakushahyoubanki_NER/.env` (see `.env.example`)
