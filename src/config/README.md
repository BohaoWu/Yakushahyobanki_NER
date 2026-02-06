# Configuration

This directory contains the project configuration files.

## Files

### config.py

Main configuration file containing:
- API key settings (OpenAI, Anthropic)
- Data path settings
- **Unified model registry (MODEL_CONFIGS)** - all BERT and LLaMA model definitions
- Training parameter defaults

## Model Registry (MODEL_CONFIGS)

All supported models are centralized in the `MODEL_CONFIGS` dictionary, avoiding duplicate definitions across scripts.

```python
from config.config import MODEL_CONFIGS, get_model_name, get_model_config

# Get full model name
model_name = get_model_name("bert")
# Returns: "cl-tohoku/bert-base-japanese-v3"

# Get full config
config = get_model_config("bert")
print(config["description"])  # Japanese BERT (Tohoku University)
```

### Supported Models

| Short Name | Full Name | Organization |
|------------|-----------|--------------|
| bert | cl-tohoku/bert-base-japanese-v3 | Tohoku University |
| bert-large | cl-tohoku/bert-large-japanese-v2 | Tohoku University |
| roberta | nlp-waseda/roberta-base-japanese | Waseda University |
| luke | studio-ousia/luke-japanese-base-lite | Studio Ousia |
| albert | ALINEAR/albert-japanese-v2 | ALINEAR |
| deberta | ku-nlp/deberta-v2-base-japanese | Kyoto University |
| deberta-large | ku-nlp/deberta-v2-large-japanese | Kyoto University |

### Helper Functions

```python
# List all available models
from config.config import list_available_models
models = list_available_models()

# Get model name (with error checking)
from config.config import get_model_name
model = get_model_name("bert")      # Returns full name
model = get_model_name("invalid")   # Raises ValueError

# Get full config
from config.config import get_model_config
config = get_model_config("bert")
# Returns: {"name": "...", "description": "...", "size": "base", "organization": "..."}
```

## API Key Configuration

API keys are managed via a `.env` file in the project root (`/root/Yakushahyoubanki_NER/.env`).

### Setup

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Keys are automatically loaded by `python-dotenv`. `config.py` reads them via `os.getenv()`.

### Priority

API key resolution order (highest to lowest):
1. Environment variables (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`)
2. `.env` file (auto-loaded by python-dotenv)

## Security

- Never commit `.env` files containing real API keys to Git
- `.env` is already in `.gitignore`
- BERT/BiLSTM training works without API keys
