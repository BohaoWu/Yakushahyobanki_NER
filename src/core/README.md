# Core Training and Evaluation

Unified NER training system for historical Japanese documents. Supports multiple model architectures, training strategies, and evaluation workflows.

## Files

| File | Description |
|------|-------------|
| `train.py` | Unified training system: models, datasets, trainers |
| `evaluation.py` | Result evaluation and comparative analysis |

## Architecture Overview

```
                    HistoryDocumentModel (Base)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   HistoryDocument    HistoryDocument   HistoryDocument
     ModelBERT          ModelBiLSTM        ModelLLM
         │                                   │
         │              ┌────────────────────┴────────────────────┐
         │              │                                         │
         │    HistoryDocumentModelLocalLLM          HistoryDocumentModelRemoteLLM
         │              │                                         │
         │    HistoryDocumentModelLLaMA           ┌───────────────┴───────────────┐
         │                                        │                               │
         │                         HistoryDocumentModel       HistoryDocumentModel
         │                          RemoteLLMChatGPT           RemoteLLMClaude
         │
    (with optional CRF layer)
```

## Model Classes

### Base Classes

| Class | Description |
|-------|-------------|
| `HistoryDocumentModel` | Abstract base for all models |
| `HistoryDocumentModelBERT` | BERT-family models (with optional CRF) |
| `HistoryDocumentModelBiLSTM` | BiLSTM model (with optional CRF) |
| `HistoryDocumentModelLLM` | Base for LLM models |

### LLM Models

| Class | Description |
|-------|-------------|
| `HistoryDocumentModelLocalLLM` | Base for local LLMs |
| `HistoryDocumentModelLLaMA` | LLaMA with LoRA fine-tuning |
| `HistoryDocumentModelRemoteLLM` | Base for remote API LLMs |
| `HistoryDocumentModelRemoteLLMChatGPT` | OpenAI ChatGPT |
| `HistoryDocumentModelRemoteLLMClaude` | Anthropic Claude |

### Neural Network Components

| Class | Description |
|-------|-------------|
| `BiLSTMForTokenClassification` | BiLSTM token classifier |
| `BiLSTMWithCRF` | BiLSTM with CRF layer |
| `CRFWrapper` | CRF layer wrapper |
| `BertWithCrfForTokenClassification` | BERT with CRF layer |

## Trainer Classes

```
                   HistoryDocumentTrainer (Base)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   HistoryDocument    HistoryDocument   HistoryDocument
     TrainerBERT        TrainerBiLSTM      TrainerLLM
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
              HistoryDocumentTrainerLocalLLM        HistoryDocumentTrainerRemoteLLM
                        │                                         │
              HistoryDocumentTrainerLLaMA          ┌──────────────┴──────────────┐
                                                   │                             │
                                        HistoryDocument           HistoryDocument
                                         TrainerRemoteLLM          TrainerRemoteLLM
                                           ChatGPT                   Claude
```

| Class | Description |
|-------|-------------|
| `HistoryDocumentTrainer` | Base trainer with common logic |
| `HistoryDocumentTrainerBERT` | BERT training (supports CRF, MLM pretrain) |
| `HistoryDocumentTrainerBiLSTM` | BiLSTM training |
| `HistoryDocumentTrainerLLaMA` | LLaMA fine-tuning with LoRA |
| `HistoryDocumentTrainerRemoteLLMChatGPT` | ChatGPT inference |
| `HistoryDocumentTrainerRemoteLLMClaude` | Claude inference |

## Data Classes

| Class | Description |
|-------|-------------|
| `HistoryDocumentDataset` | NER dataset wrapper (train/val/test splits) |
| `HistoryDocumentCorpus` | MLM pretraining corpus |
| `HistoryDocumentResult` | Training result container (metrics, paths, config) |

## Evaluation

| Class | Description |
|-------|-------------|
| `ResultEvaluator` | Multi-result comparative analysis |

### ResultEvaluator Methods

```python
from src.core.evaluation import ResultEvaluator

# Load from directory
evaluator = ResultEvaluator.from_directory("experiments/my_experiment/")

# Get best model by F1
best = evaluator.get_best_model(metric="f1")

# Get ranking
ranking = evaluator.get_ranking(metric="f1")
for model_key, score in ranking:
    print(f"{model_key}: {score:.4f}")
```

## Usage Examples

### BERT Training

```python
from src.core.train import (
    HistoryDocumentModelBERT,
    HistoryDocumentTrainerBERT,
    HistoryDocumentDataset
)

# Load dataset
dataset = HistoryDocumentDataset("dataset/yakusha_annotated_data_few-shot")

# Create model
model = HistoryDocumentModelBERT(
    model_name="bert",  # or full path: "cl-tohoku/bert-base-japanese-v3"
    use_crf=True
)

# Create trainer
trainer = HistoryDocumentTrainerBERT(
    model=model,
    dataset=dataset,
    output_dir="experiments/bert_crf"
)

# Train
result = trainer.train(num_epochs=20, batch_size=16)
print(f"F1: {result.metrics['f1']:.4f}")
```

### BiLSTM Training

```python
from src.core.train import (
    HistoryDocumentModelBiLSTM,
    HistoryDocumentTrainerBiLSTM,
    HistoryDocumentDataset
)

model = HistoryDocumentModelBiLSTM(use_crf=True)
trainer = HistoryDocumentTrainerBiLSTM(model=model, dataset=dataset, output_dir="experiments/bilstm")
result = trainer.train(num_epochs=20)
```

### LLaMA Fine-tuning

```python
from src.core.train import (
    HistoryDocumentModelLLaMA,
    HistoryDocumentTrainerLLaMA,
    HistoryDocumentDataset
)

model = HistoryDocumentModelLLaMA(
    model_name="rinna/japanese-gpt-neox-3.6b",
    use_lora=True,
    use_4bit=True
)
trainer = HistoryDocumentTrainerLLaMA(model=model, dataset=dataset, output_dir="experiments/llama")
result = trainer.train(num_epochs=5)
```

### Remote LLM Inference

```python
from src.core.train import (
    HistoryDocumentModelRemoteLLMChatGPT,
    HistoryDocumentTrainerRemoteLLMChatGPT,
    HistoryDocumentDataset
)

model = HistoryDocumentModelRemoteLLMChatGPT(
    model_name="gpt-4o-mini",
    n_few_shot=5
)
trainer = HistoryDocumentTrainerRemoteLLMChatGPT(model=model, dataset=dataset, output_dir="experiments/chatgpt")
result = trainer.evaluate()  # No training, just inference on test set
```

## Utility Functions

| Function | Description |
|----------|-------------|
| `resolve_model_name(name)` | Convert short name to full HuggingFace model ID |
| `get_model_type(name)` | Detect model type (bert, llama, bilstm, etc.) |

```python
from src.core.train import resolve_model_name, get_model_type

resolve_model_name("bert")  # "cl-tohoku/bert-base-japanese-v3"
get_model_type("bert")       # "bert"
get_model_type("bilstm")     # "bilstm"
```

## Supported Models

See `src/config/README.md` for the full model registry.

| Short Name | Type |
|------------|------|
| bert, bert-large | BERT |
| roberta | RoBERTa |
| luke | LUKE |
| albert | ALBERT |
| deberta, deberta-large | DeBERTa |
| bilstm | BiLSTM |
| rinna-3.6b, rinna-1.3b | LLaMA (local) |
| gpt-4o-mini | ChatGPT (remote) |
| claude-3-haiku-* | Claude (remote) |

## Integration with run.sh

This module is invoked by `scripts/run.sh` based on JSON config files. The config specifies models, datasets, epochs, and training parameters. See `scripts/README.md` for details.
