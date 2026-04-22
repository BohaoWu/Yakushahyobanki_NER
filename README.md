# Yakushahyoubanki NER

Named Entity Recognition for Japanese historical texts (Yakushahyoubanki / actor critiques).

[![Python](https://img.shields.io/badge/python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()

## Overview

This system trains NER models on Yakushahyoubanki (Edo-period actor critique) literature with multiple training strategies:

- **Multi-model support**: BERT family (7 variants), BiLSTM, LLaMA (LoRA), remote LLMs (ChatGPT, Claude)
- **Multi-stage training**: One-stage, two-stage (synthetic data), three-stage (MLM + synthetic data)
- **Synthetic data augmentation**: LLM-generated training data via ChatGPT/Claude APIs
- **Domain-adaptive pretraining**: MLM pretraining on classical Japanese corpus
- **GPU auto-scheduling**: Automatic multi-GPU parallel execution

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys (required for LLM features)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and ANTHROPIC_API_KEY

# Quick validation test (default: BERT, BERT+CRF, BiLSTM, 1 epoch)
./scripts/run.sh

# Use a specific config
./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 20

# Preview commands without executing
./scripts/run.sh --dry-run
```

## API Configuration

API keys are managed via a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Keys are loaded automatically by `python-dotenv`. BERT/BiLSTM training works without API keys; only LLM data generation and inference require them.

## Training Pipelines

All training is driven by `scripts/run.sh` with JSON config files.

### One-Stage: Direct Fine-tuning

Train directly on expert-annotated few-shot data:

```bash
./scripts/run.sh --config scripts/config_one_stage_slm.json
```

### Two-Stage: Synthetic Data + Fine-tuning

Stage 1: Train on LLM-generated synthetic data. Stage 2: Fine-tune on few-shot data.

```bash
# ChatGPT synthetic data (topN = 5, 10, 20, 50, 100)
./scripts/run.sh --config scripts/chatgpt/config_two_stage_slm.json --topn 20

# Claude synthetic data
./scripts/run.sh --config scripts/claude/config_two_stage_slm_claude.json --topn 20
```

### Three-Stage: MLM + Synthetic Data + Fine-tuning

Stage 1: MLM domain-adaptive pretraining. Stage 2: Synthetic data. Stage 3: Few-shot fine-tuning.

```bash
./scripts/run.sh --config scripts/chatgpt/config_three_stage_slm.json --topn 50
./scripts/run.sh --config scripts/claude/config_three_stage_slm_claude.json --topn 50
```

### MLM Pretraining + Fine-tuning (No Synthetic Data)

```bash
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json
```

### LLM Zero-Shot / Few-Shot Inference

Evaluate ChatGPT and Claude directly on the test set:

```bash
./scripts/run.sh --config scripts/llm_inference/config_llm_zero_shot.json
```

## Supported Models

| Short Name | Full Path | Type |
|------------|-----------|------|
| `bert` | cl-tohoku/bert-base-japanese-v3 | Transformer |
| `bert-large` | cl-tohoku/bert-large-japanese | Transformer |
| `roberta` | nlp-waseda/roberta-base-japanese | Transformer |
| `luke` | studio-ousia/luke-japanese-base-lite | Transformer |
| `albert` | ALINEAR/albert-japanese-v2 | Transformer |
| `deberta` | ku-nlp/deberta-v2-base-japanese | Transformer |
| `deberta-large` | ku-nlp/deberta-v2-large-japanese | Transformer |
| `bilstm` | BiLSTM | Sequential |
| `rinna-3.6b` | rinna/japanese-gpt-neox-3.6b | LLaMA (LoRA) |
| `rinna-1.3b` | rinna/japanese-gpt-1b | LLaMA (LoRA) |

Remote LLMs (inference only): `gpt-4o-mini` (ChatGPT), `claude-3-haiku-20240307` (Claude)

Optional model parameters: `use_crf` (BERT family), `use_lora` / `use_4bit` (LLaMA).

## GPU Auto-Scheduling

GPU assignment is automatic. No configuration needed.

| Scenario | Behavior |
|----------|----------|
| Single GPU | Sequential execution, auto-detected |
| Multiple GPUs | Parallel queue: tasks dispatched to free GPUs |
| No GPU | Falls back to CPU |
| `--gpu N` | Force sequential execution on GPU N |

## run.sh Options

```
--config FILE         Configuration file (default: scripts/config_quick_test.json)
--gpu DEVICE          Force specific GPU (sequential execution)
--topn N              TopN value for template configs (replaces ${topn})
--corpus-size SIZE    MLM corpus size override
--corpus-step STEP    Batch corpus size experiments (requires --corpus-max)
--corpus-max MAX      Max corpus size for batch experiments
--dry-run             Preview commands only
```

## Dataset

The primary dataset is `dataset/yakusha_annotated_data_few-shot` containing expert-annotated Yakushahyoubanki NER data.

### Entity Types (11 types)

| Entity Type | Description |
|-------------|-------------|
| Actor (役者) | Actor name |
| Play Title (演目名) | Play name |
| Person Name (人名) | Person name |
| Book Title (書名) | Book title |
| Event (事項) | Event/matter |
| Stage Name (俳名) | Haiku stage name |
| House Name (屋号) | Theater house name |
| Character Name (役名) | Character name |
| Playwright (狂言作者) | Playwright |
| Theater Operator (興行関係者) | Theater operator |
| Musician (音曲) | Musician |

### Synthetic Datasets

Generated via ChatGPT/Claude APIs using the data generation tools in `src/data_generation/`. Naming convention:
- `yakusha_synthetic_topnN_uniform_chatgpt` - ChatGPT top-N uniform
- `yakusha_synthetic_topnN_uniform_claude` - Claude top-N uniform

## Project Structure

```
Yakushahyoubanki_NER/
├── src/
│   ├── config/                        # Configuration
│   │   └── config.py                  # API keys, model configs, paths
│   ├── core/                          # Training and evaluation
│   │   ├── train.py                   # Unified trainer (all model types)
│   │   └── evaluation.py              # Evaluation and reporting
│   └── data_generation/               # Synthetic data generation
│       ├── synthetic_data_classes.py   # Core classes (Corpus, Model, Dataset)
│       ├── synthetic_data_generator.py # Generator and convenience functions
│       └── generate_balanced.py        # CLI interface
│
├── scripts/                           # Training scripts and configs
│   ├── run.sh                         # Main entry point
│   ├── config_quick_test.json         # Quick validation (default)
│   ├── config_one_stage_slm.json      # One-stage training
│   ├── chatgpt/                       # ChatGPT two/three-stage configs
│   ├── claude/                        # Claude two/three-stage configs
│   ├── mlm_finetuning/                # MLM pretrain + fine-tune configs
│   └── llm_inference/                 # LLM zero/few-shot inference configs
│
├── dataset/                           # Datasets (gitignored)
├── corpus/                            # MLM pretraining corpus (gitignored)
├── synthetic_data/                    # Corpus data (gitignored)
├── experiments/                       # Experiment results (gitignored)
├── logs/                              # Training logs (gitignored)
│
├── .env                               # API keys (gitignored)
├── .env.example                       # API key template
└── requirements.txt                   # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- 16GB+ RAM (32GB+ recommended)
- GPU: 8GB+ VRAM for BERT, 16GB+ for LLaMA

### Install

```bash
pip install -r requirements.txt
```

### PyTorch with CUDA

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Synthetic Data Generation

Generate NER training data using ChatGPT or Claude:

```bash
cd src/data_generation

# ChatGPT
python3 generate_balanced.py \
    --source dataset/yakusha_annotated_data_few-shot \
    --output dataset/yakusha_synthetic_topn20_uniform_chatgpt \
    --top-n 20 --total-samples 10000

# Claude
python3 generate_balanced.py \
    --source dataset/yakusha_annotated_data_few-shot \
    --output dataset/yakusha_synthetic_topn20_uniform_claude \
    --top-n 20 --total-samples 10000 \
    --provider claude
```

See `src/data_generation/README.md` for details.

## Troubleshooting

**JSON parse error**: `python3 -m json.tool <config_file>`

**Dataset not found**: Paths in configs are relative to project root `/root/Yakushahyoubanki_NER/`

**CUDA out of memory**: Reduce `batch_size` in config, or enable `use_4bit` for LLaMA models

**API key missing**: Configure in `.env` at project root (see `.env.example`)
