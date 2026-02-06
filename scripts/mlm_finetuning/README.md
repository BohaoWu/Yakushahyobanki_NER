# MLM Pretraining + Fine-tuning Configurations

This directory contains configuration files for two-stage NER model training: MLM domain-adaptive pretraining followed by fine-tuning on expert-annotated data.

## Training Strategy

This approach consists of two stages:

1. **Stage 1: Domain-Adaptive Pretraining (MLM)**
   - Pretrain on `corpus/corpus_gennbun.txt` (118,279 lines of classical Japanese text)
   - Corpus size: 100,000 samples
   - Epochs: 3
   - Learning rate: 5e-5
   - Batch size: 16
   - MLM probability: 15%

2. **Stage 2: Fine-tune on Few-Shot Data**
   - Fine-tune on `dataset/yakusha_annotated_data_few-shot`
   - Epochs: 20
   - Learning rate: 2e-5
   - Batch size: 64

## Included Models

All configurations include the following 14 models:

**Transformer Models:**
- BERT (base)
- BERT (base) + CRF
- BERT-large
- BERT-large + CRF
- RoBERTa
- RoBERTa + CRF
- LUKE
- LUKE + CRF
- ALBERT
- ALBERT + CRF
- DeBERTa
- DeBERTa + CRF

**Sequential Models:**
- BiLSTM
- BiLSTM + CRF

## Configuration Files

- `config_mlm_finetuning_100000.json` - Two-stage training with MLM pretraining on 100,000 corpus samples

## Comparison with Other Approaches

- **vs. Two-Stage (Synthetic + Few-shot)**: This approach replaces synthetic data pretraining with MLM pretraining
- **vs. Three-Stage (MLM + Synthetic + Few-shot)**: This approach skips the synthetic data stage
- **vs. Direct Fine-tuning**: This approach adds domain-adaptive pretraining before fine-tuning

## Usage

### Dry Run (Preview Commands)

```bash
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --dry-run
```

### Foreground Execution

```bash
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json
```

### Background Execution (Recommended)

```bash
nohup ./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json > logs/mlm_finetuning_100000.log 2>&1 &
```

Monitor the progress:
```bash
tail -f logs/mlm_finetuning_100000.log
```

### Additional Options

```bash
# Specify different GPU device
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --gpu 0

# Use different corpus size
./scripts/run.sh --config scripts/mlm_finetuning/config_mlm_finetuning_100000.json --corpus-size 50000
```

## Expected Output

Results will be saved to:
- Model checkpoints: `experiments/mlm_finetuning_100000_{timestamp}/{model_name}/`
- Evaluation reports: `experiments/mlm_finetuning_100000_{timestamp}/evaluation_report/`
- Logs: `logs/mlm_finetuning_100000.log`
