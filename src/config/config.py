"""
Project Configuration File

Stores environment variables such as API keys and path configurations
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# API Configuration
# ============================================================================

# OpenAI API configuration
# SECURITY: API keys should be set via environment variables, do not hardcode them
# Setup: export OPENAI_API_KEY=your_key_here or use a .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Read from environment variable
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default: GPT-4o-mini (more cost-effective)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # Optional: custom API endpoint

# Anthropic Claude API configuration
# SECURITY: API keys should be set via environment variables, do not hardcode them
# Setup: export ANTHROPIC_API_KEY=your_key_here or use a .env file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Read from environment variable
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")  # Default: Claude 3 Haiku

# ============================================================================
# Data Path Configuration
# ============================================================================

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "dataset"
STANDARD_DATASET = DATASET_DIR / "yakusya_annotated_data"
STANDARD_DATASET_JSON = STANDARD_DATASET / "yakusya_annotated_data.json"
ZEROSHOT_DATASET = DATASET_DIR / "yakusha_annotated_data_zero-shot"
FEWSHOT_DATASET = DATASET_DIR / "yakusha_annotated_data_few-shot"
TEST_DATASET = DATASET_DIR / "yakusha_annotated_data_test"

# Synthetic data path
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "synthetic_data"
CORPUS_FILE = SYNTHETIC_DATA_DIR / "corpus_augment.txt"

# Output data path
SYNTHETIC_NER_DATASET = DATASET_DIR / "yakusha_synthetic_ner_dataset_chatgpt"

# ============================================================================
# Experiment Configuration
# ============================================================================

# Experiment output directory
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Log directory
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# Model Configuration
# ============================================================================

# Unified model configuration (includes BERT family and LLaMA family)
MODEL_CONFIGS = {
    # Multilingual BERT models
    "bert-multilingual": {
        "name": "google-bert/bert-base-multilingual-cased",
        "description": "Multilingual BERT (104 languages)",
        "size": "base",
        "type": "bert",
        "organization": "Google",
    },
    # German BERT models
    "bert-german": {
        "name": "dbmdz/bert-base-german-cased",
        "description": "German BERT (DBMDZ/Bavarian State Library)",
        "size": "base",
        "type": "bert",
        "organization": "DBMDZ",
    },
    # English BERT models
    "bert-english": {
        "name": "google-bert/bert-base-cased",
        "description": "English BERT base cased (Google)",
        "size": "base",
        "type": "bert",
        "organization": "Google",
    },
    # French BERT models
    "bert-french": {
        "name": "dbmdz/bert-base-french-europeana-cased",
        "description": "French BERT trained on historical Europeana newspapers (DBMDZ)",
        "size": "base",
        "type": "bert",
        "organization": "DBMDZ",
    },
    # BERT family models (Japanese)
    "bert": {
        "name": "cl-tohoku/bert-base-japanese-v3",
        "description": "Japanese BERT (Tohoku University)",
        "size": "base",
        "type": "bert",
        "organization": "Tohoku University",
    },
    "bert-large": {
        "name": "cl-tohoku/bert-large-japanese-v2",
        "description": "Japanese large-BERT (Tohoku University)",
        "size": "large",
        "type": "bert",
        "organization": "Tohoku University",
    },
    "roberta": {
        "name": "nlp-waseda/roberta-base-japanese",
        "description": "Japanese RoBERTa (Waseda University)",
        "size": "base",
        "type": "bert",
        "organization": "Waseda University",
    },
    "xlnet": {
        "name": "hajime9652/xlnet-japanese",
        "description": "Japanese XLNet",
        "size": "base",
        "type": "bert",
        "organization": "hajime9652",
    },
    "luke": {
        "name": "studio-ousia/luke-japanese-base-lite",
        "description": "Japanese LUKE (Studio Ousia)",
        "size": "base",
        "type": "bert",
        "organization": "Studio Ousia",
    },
    "albert": {
        "name": "ALINEAR/albert-japanese-v2",
        "description": "Japanese ALBERT",
        "size": "base",
        "type": "bert",
        "organization": "ALINEAR",
    },
    "deberta": {
        "name": "ku-nlp/deberta-v2-base-japanese",
        "description": "Japanese DeBERTa (Kyoto University)",
        "size": "base",
        "type": "bert",
        "organization": "Kyoto University",
    },
    "deberta-large": {
        "name": "ku-nlp/deberta-v2-large-japanese",
        "description": "Japanese DeBERTa Large (Kyoto University)",
        "size": "large",
        "type": "bert",
        "organization": "Kyoto University",
    },
    # LLaMA family models (Official Meta LLaMA via NeMo)
    "llama-1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "description": "Official Meta Llama 3.2 1B",
        "size": "1B",
        "type": "llama",
        "organization": "Meta",
    },
    "llama-3b": {
        "name": "meta-llama/Llama-3.2-3B",
        "description": "Official Meta Llama 3.2 3B",
        "size": "3B",
        "type": "llama",
        "organization": "Meta",
    },
    # BiLSTM (Benchmark/Baseline)
    "bilstm": {
        "name": "bilstm",
        "description": "BiLSTM baseline model with BERT tokenizer",
        "size": "small",
        "type": "bilstm",
        "organization": "Custom",
    },
    # Remote LLM (API-based models)
    "chatgpt": {
        "name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o-mini (API)",
        "size": "unknown",
        "type": "remote_llm",
        "organization": "OpenAI",
        "provider": "chatgpt",
    },
    "claude": {
        "name": "claude-3-haiku-20240307",
        "description": "Anthropic Claude 3 Haiku (API)",
        "size": "unknown",
        "type": "remote_llm",
        "organization": "Anthropic",
        "provider": "claude",
    },
}

# Default training parameters
DEFAULT_TRAIN_PARAMS = {
    "mlm_epochs": 3,
    "ner_epochs": 20,
    "mlm_batch_size": 16,
    "ner_batch_size": 32,
    "learning_rate": 5e-5,
    "seed": 42,
}

# ============================================================================
# GPU Configuration
# ============================================================================

# Default GPU device
DEFAULT_GPU = 0

# ============================================================================
# Other Configuration
# ============================================================================

# Random seed
RANDOM_SEED = 42

# Log level
LOG_LEVEL = "INFO"


def get_openai_api_key() -> str:
    """
    Get the OpenAI API key

    Priority:
    1. Environment variable OPENAI_API_KEY
    2. OPENAI_API_KEY in the configuration file

    Returns:
        API key string
    """
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the environment variable OPENAI_API_KEY or configure it in config/config.py"
        )
    return api_key


def get_anthropic_api_key() -> str:
    """
    Get the Anthropic Claude API key

    Priority:
    1. Environment variable ANTHROPIC_API_KEY
    2. ANTHROPIC_API_KEY in the configuration file

    Returns:
        API key string
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Please set the environment variable ANTHROPIC_API_KEY or configure it in config/config.py"
        )
    return api_key


def get_dataset_path(dataset_type: str = "standard") -> Path:
    """
    Get the dataset path

    Args:
        dataset_type: Dataset type (standard/zeroshot/fewshot)

    Returns:
        Dataset path
    """
    dataset_map = {
        "standard": STANDARD_DATASET,
        "zeroshot": ZEROSHOT_DATASET,
        "fewshot": FEWSHOT_DATASET,
    }

    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset_map[dataset_type]


def get_model_name(model_key: str) -> str:
    """
    Get the full model name from the model shorthand key

    Args:
        model_key: Model shorthand key (e.g., "bert", "roberta")

    Returns:
        Full model name (e.g., "cl-tohoku/bert-base-japanese-v3")

    Examples:
        >>> get_model_name("bert")
        'cl-tohoku/bert-base-japanese-v3'
        >>> get_model_name("roberta")
        'nlp-waseda/roberta-base-japanese'
    """
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]["name"]
    else:
        raise ValueError(
            f"Unknown model: {model_key}. Supported models: {list(MODEL_CONFIGS.keys())}"
        )


def get_model_config(model_key: str) -> dict:
    """
    Get the full model configuration from the model shorthand key

    Args:
        model_key: Model shorthand key (e.g., "bert", "roberta")

    Returns:
        Model configuration dictionary containing name, description, size, organization

    Examples:
        >>> config = get_model_config("bert")
        >>> config["name"]
        'cl-tohoku/bert-base-japanese-v3'
        >>> config["description"]
        'Japanese BERT (Tohoku University)'
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_key}. Supported models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_key]


def list_available_models() -> list:
    """
    List all available models

    Returns:
        List of model shorthand keys

    Examples:
        >>> models = list_available_models()
        >>> "bert" in models
        True
    """
    return list(MODEL_CONFIGS.keys())


# Create necessary directories
def ensure_directories():
    """Ensure that necessary directories exist"""
    for directory in [EXPERIMENTS_DIR, LOGS_DIR, SYNTHETIC_DATA_DIR, SYNTHETIC_NER_DATASET]:
        directory.mkdir(parents=True, exist_ok=True)
