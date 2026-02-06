# LLM Zero-Shot and Few-Shot NER Inference

This directory contains configuration files for evaluating ChatGPT and Claude on the test set using zero-shot and few-shot prompting.

## Overview

This experiment evaluates large language models on Named Entity Recognition for historical Japanese texts without any fine-tuning. The models are prompted with task instructions and optionally provided with a few examples (few-shot learning).

## Test Dataset

- **Dataset**: `dataset/yakusha_annotated_data_few-shot` (test split)
- **Samples**: 1,137 samples
- **Entities**: 3,005 total entities
- **Entity Types**: 11 types (役者, 興行関係者, 俳名, 演目名, etc.)

## Models and Configurations

### ChatGPT (gpt-4o-mini)
- 0-shot: No examples provided
- 3-shot: 3 stratified examples from training set
- 5-shot: 5 stratified examples from training set

### Claude (claude-3-haiku-20240307)
- 0-shot: No examples provided
- 3-shot: 3 stratified examples from training set
- 5-shot: 5 stratified examples from training set

**Total**: 6 configurations

## Few-Shot Selection Strategy

The `stratified` selection strategy ensures that few-shot examples:
- Cover diverse entity types proportionally to their distribution
- Represent different text lengths and complexities
- Are randomly sampled with a fixed seed for reproducibility

## Usage

### Dry Run (Preview Commands)

```bash
./scripts/run.sh --config scripts/llm_inference/config_llm_zero_shot.json --dry-run
```

### Foreground Execution

```bash
./scripts/run.sh --config scripts/llm_inference/config_llm_zero_shot.json
```

### Background Execution (Recommended)

```bash
nohup ./scripts/run.sh --config scripts/llm_inference/config_llm_zero_shot.json > logs/llm_zero_shot.log 2>&1 &
```

Monitor the progress:
```bash
tail -f logs/llm_zero_shot.log
```

## Expected Runtime and Cost

### Runtime (approximate)
- Batch size: 50 (concurrent requests)
- Each model: ~15-30 minutes for 1,137 samples
- Total for all 6 configurations: ~1.5-3 hours

### API Cost Estimates

**ChatGPT (gpt-4o-mini)**
- 0-shot: ~$0.50
- 3-shot: ~$1.50
- 5-shot: ~$2.50
- Subtotal: ~$4.50

**Claude (claude-3-haiku-20240307)**
- 0-shot: ~$0.30
- 3-shot: ~$1.00
- 5-shot: ~$1.50
- Subtotal: ~$2.80

**Total Estimated Cost**: ~$7.30

*Note: Actual costs may vary based on response length and API pricing changes.*

## Environment Variables

API keys are configured in `/root/Yakushahyoubanki_NER/.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

See `.env.example` for the template.

## Expected Output

Results will be saved to:
- Model predictions: `experiments/llm_zero_shot_{timestamp}/{model_id}/result.json`
- Evaluation report: `experiments/llm_zero_shot_{timestamp}/evaluation_report/`
- Logs: `logs/llm_zero_shot.log`

Each result file contains:
- Precision, Recall, F1 scores (overall and per-entity-type)
- Predicted entities for each test sample
- Confusion matrix
- Error analysis

## Comparison with Fine-Tuned Models

After running this experiment, you can compare the results with fine-tuned models using:

```bash
# The evaluation tool will automatically compare if you provide paths
python src/core/evaluation.py \
  --results experiments/llm_zero_shot_*/*/result.json \
            experiments/bert_baseline/*/result.json \
  --output comparison_report/
```

## Notes

- The models will only perform inference on the test set (no training)
- `epochs: [0]` indicates evaluation-only mode
- `batch_size: 50` controls concurrency - processes 50 samples before a 0.5s delay
- Results are reproducible with `seed: 42`
- Higher batch size = faster processing but may hit API rate limits
