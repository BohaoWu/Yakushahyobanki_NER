# Synthetic Data Generation

OOP system for generating NER-annotated synthetic data. Automatically analyzes real data distribution and uses stratified sampling to produce entity-balanced synthetic data.

## File Structure

```
src/data_generation/
├── synthetic_data_classes.py       # Core class definitions
│   ├── NERSyntheticDataCorpus      #   Corpus management
│   ├── NERSyntheticDataModel       #   LLM API wrapper
│   └── NERSyntheticDataDataset     #   Dataset management
├── synthetic_data_generator.py     # Generator and convenience functions
│   ├── NERSyntheticDataGenerator   #   Pipeline orchestration
│   └── create_topn_dataset()       #   One-click generation
├── generate_balanced.py            # CLI interface
└── README.md                       # This document
```

## Quick Start

### Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Keys are auto-loaded by `python-dotenv`. Alternatively, set environment variables directly:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### Method 1: Convenience Function (Recommended)

```python
from src.data_generation.synthetic_data_generator import create_topn_dataset

# Using OpenAI API
dataset = create_topn_dataset(
    real_dataset_path="dataset/yakusha_annotated_data_few-shot",
    output_dir="dataset/yakusha_synthetic_topn20_uniform_chatgpt",
    top_n=20,
    total_samples=10000,
    num_workers=10
)

# Using Claude API
dataset = create_topn_dataset(
    real_dataset_path="dataset/yakusha_annotated_data_few-shot",
    output_dir="dataset/yakusha_synthetic_ner_dataset_claude_N_5",
    top_n=5,
    total_samples=10000,
    num_workers=1,
    provider="claude",
    model="claude-haiku-4-5-20251001"
)
```

### Method 2: Command Line

```bash
cd src/data_generation

# Generate with OpenAI (ChatGPT)
python3 generate_balanced.py \
    --source dataset/yakusha_annotated_data_few-shot \
    --output dataset/yakusha_synthetic_topn20_uniform_chatgpt \
    --top-n 20 \
    --total-samples 2000

# Generate with Claude API
python3 generate_balanced.py \
    --source dataset/yakusha_annotated_data_few-shot \
    --output dataset/yakusha_synthetic_ner_dataset_claude_N_5 \
    --top-n 5 \
    --total-samples 2000 \
    --provider claude \
    --model claude-haiku-4-5-20251001
```

### Method 3: Step-by-Step (Full Control)

```python
from src.data_generation.synthetic_data_classes import (
    NERSyntheticDataCorpus,
    NERSyntheticDataModel,
    NERSyntheticDataDataset
)
from src.data_generation.synthetic_data_generator import NERSyntheticDataGenerator

# 1. Load corpus
corpus = NERSyntheticDataCorpus("dataset/yakusha_annotated_data_few-shot")
corpus.load()
corpus.print_statistics()

# 2. Initialize generation model
model = NERSyntheticDataModel(model="gpt-4o-mini")
# Or Claude:
# model = NERSyntheticDataModel(model="claude-haiku-4-5-20251001", provider="claude")

# 3. Create generator
generator = NERSyntheticDataGenerator(
    corpus=corpus,
    model=model,
    output_dir="dataset/my_output"
)

# 4. Generate data
dataset = generator.generate_topn_data(
    top_n=20,
    total_samples=10000,
    num_workers=10
)

# 5. Verify distribution
generator.verify_distribution(dataset)

# 6. Convert to HuggingFace format
hf_dataset = dataset.to_huggingface_dataset()
hf_dataset.save_to_disk("dataset/yakusha_synthetic_balanced")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  NERSyntheticDataGenerator                   │
│                  (Orchestrates entire pipeline)              │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
      ┌───────▼─────────┐           ┌────────▼───────────┐
      │ NERSynthetic     │           │  NERSynthetic      │
      │ DataCorpus       │           │  DataModel         │
      │ (Corpus)         │           │  (Generation Model)│
      └─────────────────┘           └────────────────────┘
              │                               │
              │          ┌────────────────────┘
              │          │
      ┌───────▼──────────▼────┐
      │  NERSyntheticData     │
      │  Dataset              │
      │  (Dataset)            │
      └───────────────────────┘
```

### NERSyntheticDataCorpus

Manages real data, provides distribution analysis and few-shot examples.

| Method | Description |
|--------|-------------|
| `load()` | Load real dataset |
| `get_few_shot_examples(n)` | Get few-shot examples |
| `get_top_entities(entity_type, n)` | Get Top-N entities |
| `print_statistics()` | Print statistics |

### NERSyntheticDataModel

Wraps LLM API calls and prompt generation.

| Method | Description |
|--------|-------------|
| `create_prompt(corpus, target_entities)` | Create generation prompt |
| `generate_single(corpus, target_entities)` | Generate a single sample |
| `_extract_json(text)` | Extract JSON from response |
| `_validate_and_fix_positions(sample)` | Validate and fix entity positions |

### NERSyntheticDataDataset

Stores, manages, and analyzes generated data.

| Method | Description |
|--------|-------------|
| `add_sample(sample)` | Add a sample |
| `save(output_path)` / `load(input_path)` | Save / Load |
| `analyze_distribution()` | Analyze entity distribution |
| `to_huggingface_dataset()` | Convert to HuggingFace format |

### NERSyntheticDataGenerator

Orchestrates the generation pipeline with stratified balanced generation.

| Method | Description |
|--------|-------------|
| `calculate_generation_config(target_samples)` | Calculate generation config |
| `generate_topn_data(top_n, total_samples, ...)` | Top-N generation |
| `generate_balanced_data(target_total_samples, ...)` | Proportional generation |
| `verify_distribution(dataset)` | Verify distribution consistency |

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | dataset/yakusha_annotated_data_few-shot | Real dataset path |
| `--output` | dataset/yakusha_synthetic_balanced | Output directory |
| `--top-n` | None | Select top N most frequent entities |
| `--total-samples` | None | Target total samples to generate |
| `--target-samples` | None | Proportional generation total (legacy mode) |
| `--workers` | 10 | Number of parallel threads |
| `--provider` | openai | API provider (openai or claude) |
| `--model` | auto | Model to use |
| `--api-key` | None | API key (optional, prefers env vars) |

Default models: OpenAI `gpt-4o-mini`, Claude `claude-3-haiku-20240307`

## Generation Modes

### Mode 1: Top-N Entity Generation (Recommended)

Selects the N most frequent entities and allocates samples proportionally:

```bash
python3 generate_balanced.py --top-n 20 --total-samples 10000
```

Top-N selection strategy:
1. Ensures at least one entity per entity type is selected
2. Remaining slots filled by global frequency ranking

### Mode 2: Proportional Generation

```bash
python3 generate_balanced.py --target-samples 2000
```

## Generation Pipeline

### Step 1: Analyze Real Data Distribution

```
Entity Type               Count        Ratio       Unique Entities
--------------------------------------------------------------------------------
Actor                 7819     61.4%         1860  <- Primary entity
Promoter              2267     17.8%          159
Stage Name             971      7.6%          161
...
Matter                   6      0.0%            4  <- Extremely low frequency
```

### Step 2: Calculate Balanced Generation Config

```
Entity Type          Top-N    Samples/Entity  Est. Samples   Target Ratio
--------------------------------------------------------------------------------
Actor                 30         40       1200     61.4%
Promoter              30         11        330     17.8%
Stage Name            30          5        150      7.6%
...
Matter                 3          1          3      0.0%
```

### Step 3: Stratified Parallel Generation

- Stratify by entity type
- Parallel generation within each stratum
- Ensure distribution matches real data

### Step 4: Verify Distribution

```
Entity Type          Real Ratio    Generated Ratio  Difference
--------------------------------------------------------------------------------
Actor                61.4%     61.2%      -0.2%
Promoter             17.8%     17.9%      +0.1%
Stage Name            7.6%      7.6%       0.0%
...
Mean Absolute Deviation: 0.1%
```

## Output Format

Generated NER data in JSON format:

```json
[
  {
    "text": "中村歌右衛門は江戸三座の一つで活躍した名優である。",
    "entities": [
      { "type": "役者", "name": "中村歌右衛門", "span": [0, 6] },
      { "type": "事項", "name": "江戸三座", "span": [7, 11] }
    ]
  }
]
```

## Request Strategy

Uses a queue + stack strategy for API requests:
- New tasks enter a queue (FIFO)
- Failed tasks enter a stack (LIFO, retried first)
- Max retries: 10

## Troubleshooting

**API key not found**: Configure keys in `.env` at project root.

**Library not installed**: `pip install openai anthropic`

**Import error** (`ModuleNotFoundError: No module named 'synthetic_data_classes'`): Run from `src/data_generation/` directory, or use absolute imports: `from src.data_generation.synthetic_data_classes import ...`

**API call failed**: Check that API keys are valid, network is connected, and quota is sufficient. For Claude, use `--workers 1` to avoid concurrency limits.
