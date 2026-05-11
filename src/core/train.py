"""
Unified NER Training System

Provides a unified interface for training NER models on historical documents.
Supports BERT-based models with optional CRF layer, MLM pretraining, and multi-dataset training.
"""

__all__ = [
    # Utility functions
    'resolve_model_name',
    'get_model_type',
    # Base classes
    'HistoryDocumentModel',
    'HistoryDocumentModelBERT',
    'HistoryDocumentModelBiLSTM',
    # LLM Model hierarchy
    'HistoryDocumentModelLLM',           # Base for all LLM models
    'HistoryDocumentModelLocalLLM',      # Base for local LLMs
    'HistoryDocumentModelLLaMA',         # LLaMA implementation
    'HistoryDocumentModelRemoteLLM',     # Base for remote LLMs
    'HistoryDocumentModelRemoteLLMChatGPT',
    'HistoryDocumentModelRemoteLLMClaude',
    # Dataset classes
    'HistoryDocumentDataset',
    'HistoryDocumentCorpus',
    # Trainer base class
    'HistoryDocumentTrainer',
    'HistoryDocumentTrainerBERT',
    'HistoryDocumentTrainerBiLSTM',
    # LLM Trainer hierarchy
    'HistoryDocumentTrainerLLM',         # Base for all LLM trainers
    'HistoryDocumentTrainerLocalLLM',    # Base for local LLM trainers
    'HistoryDocumentTrainerLLaMA',       # LLaMA trainer
    'HistoryDocumentTrainerRemoteLLM',   # Base for remote LLM trainers
    'HistoryDocumentTrainerRemoteLLMChatGPT',
    'HistoryDocumentTrainerRemoteLLMClaude',
    # Result and utility classes
    'HistoryDocumentResult',
    'BiLSTMForTokenClassification',
]

import glob
import torch
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from spacy_alignments.tokenizations import get_alignments
from transformers.tokenization_utils_base import BatchEncoding

# Import MODEL_CONFIGS for short name resolution
from src.config.config import MODEL_CONFIGS, get_openai_api_key, get_anthropic_api_key

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    XLNetLMHeadModel,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    BertForTokenClassification,
    PretrainedConfig,
)
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import set_seed
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# Import CRF for optional use
try:
    from torchcrf import CRF
except ImportError:
    CRF = None

# Import for BiLSTM
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput


def resolve_model_name(model_name: str) -> str:
    """
    Resolve short model name to full HuggingFace model ID using MODEL_CONFIGS.

    Args:
        model_name: Short name (e.g., "bert", "bilstm") or full model ID

    Returns:
        Full model ID (e.g., "cl-tohoku/bert-base-japanese-v3")

    Examples:
        >>> resolve_model_name("bert")
        'cl-tohoku/bert-base-japanese-v3'
        >>> resolve_model_name("cl-tohoku/bert-base-japanese-v3")
        'cl-tohoku/bert-base-japanese-v3'
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]["name"]
    return model_name


def get_model_type(model_name: str) -> str:
    """
    Get the model type from MODEL_CONFIGS.

    Args:
        model_name: Short name or full model ID

    Returns:
        Model type: "bert", "llama", or "bilstm"
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].get("type", "bert")
    # Try to infer from full name
    for config in MODEL_CONFIGS.values():
        if config["name"] == model_name:
            return config.get("type", "bert")
    return "bert"  # Default to bert


class BiLSTMForTokenClassification(nn.Module):
    """
    BiLSTM model for token classification (NER benchmark)

    This model serves as a baseline/benchmark for comparing with transformer-based models.
    Supports optional CRF layer, character-level CNN features, and pretrained embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_token_id: int = 0,
        use_char_cnn: bool = False,
        char_vocab_size: int = 5000,
        char_embedding_dim: int = 50,
        char_cnn_filters: int = 50,
        char_cnn_kernel_size: int = 3,
    ):
        """
        Initialize BiLSTM model

        Args:
            vocab_size: Size of vocabulary
            num_labels: Number of NER labels
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pad_token_id: Padding token ID
            use_char_cnn: Whether to use character-level CNN features
            char_vocab_size: Size of character vocabulary
            char_embedding_dim: Dimension of character embeddings
            char_cnn_filters: Number of CNN filters for character features
            char_cnn_kernel_size: Kernel size for character CNN
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_char_cnn = use_char_cnn
        self.pad_token_id = pad_token_id

        # Word embedding layer
        self.word_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )

        # Character-level CNN (optional)
        self.char_cnn_dim = 0
        if use_char_cnn:
            self.char_embedding = nn.Embedding(
                char_vocab_size, char_embedding_dim, padding_idx=0
            )
            self.char_cnn = nn.Conv1d(
                char_embedding_dim, char_cnn_filters,
                kernel_size=char_cnn_kernel_size, padding=char_cnn_kernel_size // 2
            )
            self.char_cnn_dim = char_cnn_filters

        # BiLSTM layer
        lstm_input_dim = embedding_dim + self.char_cnn_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification layer (BiLSTM output is 2 * hidden_dim due to bidirectional)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        # Create a config-like object for compatibility with HuggingFace Trainer
        self.config = PretrainedConfig(
            num_labels=num_labels,
            hidden_size=hidden_dim * 2,
            vocab_size=vocab_size,
        )
        self.config.label2id = {}
        self.config.id2label = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        char_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ):
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            char_ids: Character IDs [batch_size, seq_len, max_char_len] (optional)
            labels: Label IDs [batch_size, seq_len] (optional)

        Returns:
            TokenClassifierOutput with loss and logits
        """
        batch_size, seq_len = input_ids.shape

        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch, seq, embed_dim]

        # Character CNN features (optional)
        if self.use_char_cnn and char_ids is not None:
            # char_ids: [batch, seq, max_char_len]
            batch_size, seq_len, max_char_len = char_ids.shape
            char_embeds = self.char_embedding(char_ids)  # [batch, seq, max_char, char_embed]

            # Reshape for CNN: [batch * seq, char_embed, max_char]
            char_embeds = char_embeds.view(-1, max_char_len, self.char_embedding.embedding_dim)
            char_embeds = char_embeds.permute(0, 2, 1)  # [batch * seq, char_embed, max_char]

            # Apply CNN and max pooling
            char_cnn_out = self.char_cnn(char_embeds)  # [batch * seq, filters, max_char]
            char_features = torch.max(char_cnn_out, dim=2)[0]  # [batch * seq, filters]
            char_features = char_features.view(batch_size, seq_len, -1)  # [batch, seq, filters]

            # Concatenate word and char features
            embeds = torch.cat([word_embeds, char_features], dim=-1)
        else:
            embeds = word_embeds

        # Apply dropout to embeddings
        embeds = self.dropout(embeds)

        # BiLSTM
        lstm_out, _ = self.lstm(embeds)  # [batch, seq, hidden * 2]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Classification
        logits = self.classifier(lstm_out)  # [batch, seq, num_labels]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights and config"""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = os.path.join(save_directory, "bilstm_config.json")
        config = {
            "vocab_size": self.vocab_size,
            "num_labels": self.num_labels,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.lstm.num_layers,
            "dropout": self.dropout.p,
            "pad_token_id": self.pad_token_id,
            "use_char_cnn": self.use_char_cnn,
            "label2id": self.config.label2id,
            "id2label": self.config.id2label,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ BiLSTM model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from pretrained checkpoint"""
        import os

        # Load config
        config_path = os.path.join(model_path, "bilstm_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model
        model = cls(
            vocab_size=config["vocab_size"],
            num_labels=config["num_labels"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pad_token_id=config["pad_token_id"],
            use_char_cnn=config.get("use_char_cnn", False),
        )

        # Load weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        # Restore label mappings
        model.config.label2id = config.get("label2id", {})
        model.config.id2label = config.get("id2label", {})

        return model


class BiLSTMWithCRF(nn.Module):
    """BiLSTM model with CRF layer for token classification"""

    def __init__(self, bilstm_model: BiLSTMForTokenClassification, label2id: dict):
        """
        Initialize BiLSTM with CRF wrapper

        Args:
            bilstm_model: BiLSTMForTokenClassification instance
            label2id: Label to ID mapping
        """
        super().__init__()
        if CRF is None:
            raise ImportError(
                "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
            )

        self.bilstm = bilstm_model
        self.label2id = label2id
        self.crf = CRF(len(label2id), batch_first=True)

        # Initialize CRF transitions
        st, t, et = create_crf_transitions(label2id)
        self.crf.start_transitions.data = st
        self.crf.transitions.data = t
        self.crf.end_transitions.data = et

        # Expose config from base model
        self.config = bilstm_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ):
        """Forward pass with CRF layer

        Note: BiLSTM uses BERT tokenizer which adds [CLS] at the start.
        We skip the first token (CLS) for CRF processing, similar to CRFWrapper.
        """
        # Get BiLSTM output
        output = self.bilstm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        if labels is not None:
            logits = output.logits

            # Skip first token (CLS) for CRF - same as CRFWrapper
            logits_for_crf = logits[:, 1:, :]
            labels_for_crf = labels[:, 1:]

            mask = labels_for_crf != -100
            labels_crf = labels_for_crf.clone()
            labels_crf[~mask] = 0

            # Ensure first timestep mask is True for CRF
            mask[:, 0] = True

            # CRF loss (negative log-likelihood)
            loss = -self.crf(
                logits_for_crf,
                labels_crf,
                mask=mask,
                reduction="mean"
            )
            output = TokenClassifierOutput(loss=loss, logits=logits)

        return output

    def decode(self, logits: torch.Tensor, attention_mask: torch.Tensor = None) -> list:
        """
        Use Viterbi decoding to get the best tag sequence

        Args:
            logits: Model output logits (batch_size, seq_len, num_labels)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of predicted tag sequences (same length as input logits)

        Note: BiLSTM uses BERT tokenizer which adds [CLS] at the start.
        We skip the first token (CLS) for CRF decoding, then pad the result.
        """
        # Skip first token (CLS) - same as CRFWrapper for BERT-like models
        logits_for_crf = logits[:, 1:, :]

        if attention_mask is None:
            mask = torch.ones(logits_for_crf.shape[:2], dtype=torch.bool, device=logits.device)
        else:
            mask = attention_mask[:, 1:].bool()

        # CRF requires first timestep mask to be True
        mask[:, 0] = True

        # Use CRF decode (Viterbi algorithm)
        decoded = self.crf.decode(logits_for_crf, mask=mask)

        # Pad the decoded sequences to match original sequence length
        # Prepend O tag (label 0) for [CLS] position
        result = []
        for seq in decoded:
            result.append([0] + seq)
        return result

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save BiLSTM and CRF weights"""
        self.bilstm.save_pretrained(save_directory, **kwargs)
        crf_path = Path(save_directory) / "crf_weights.pt"
        torch.save({
            'crf_state_dict': self.crf.state_dict(),
            'label2id': self.label2id,
        }, crf_path)

    @classmethod
    def from_pretrained(cls, model_path: str, label2id: dict = None, **kwargs):
        """Load BiLSTM+CRF from pretrained checkpoint"""
        bilstm_model = BiLSTMForTokenClassification.from_pretrained(model_path, **kwargs)

        crf_path = Path(model_path) / "crf_weights.pt"
        if crf_path.exists():
            crf_data = torch.load(crf_path, map_location="cpu")
            if label2id is None:
                label2id = crf_data['label2id']
        else:
            if label2id is None:
                label2id = bilstm_model.config.label2id

        wrapper = cls(bilstm_model, label2id)
        if crf_path.exists():
            wrapper.crf.load_state_dict(crf_data['crf_state_dict'])

        return wrapper


# Helper function to convert numpy types to native Python types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def create_crf_transitions(label2id):
    """Define transition scores for CRF layer

    Args:
        label2id: Label to ID mapping

    Returns:
        Tuple of (start_transitions, transitions, end_transitions)
    """
    # List of B- label IDs
    b_ids = [v for k, v in label2id.items() if k.startswith("B-")]
    # List of I- label IDs
    i_ids = [v for k, v in label2id.items() if k.startswith("I-")]
    o_id = label2id["O"]  # O label ID

    # Define start transition scores
    start_transitions = torch.full([len(label2id)], -100.0)
    start_transitions[b_ids] = 0
    start_transitions[o_id] = 0

    # Define label-to-label transition scores
    transitions = torch.full([len(label2id), len(label2id)], -100.0)
    transitions[:, b_ids] = 0
    transitions[:, o_id] = 0
    transitions[b_ids, i_ids] = 0
    transitions[i_ids, i_ids] = 0

    # Define end transition scores
    end_transitions = torch.zeros(len(label2id))
    return start_transitions, transitions, end_transitions


class CRFWrapper(torch.nn.Module):
    """
    Universal CRF wrapper for any transformer model (BERT, RoBERTa, DeBERTa, ALBERT, LUKE, XLNet, etc.)

    This wrapper adds a CRF layer on top of any model that outputs logits for token classification.
    Handles different special token positions:
    - BERT-like: [CLS] ... [SEP] (special tokens at start)
    - XLNet-like: ... <sep> <cls> (special tokens at end)
    """

    def __init__(self, base_model, label2id):
        """
        Initialize CRF wrapper

        Args:
            base_model: Any pretrained model for token classification
            label2id: Label to ID mapping
        """
        super().__init__()
        if CRF is None:
            raise ImportError(
                "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
            )
        self.base_model = base_model
        self.label2id = label2id
        self.crf = CRF(len(label2id), batch_first=True)

        # Initialize CRF transitions
        st, t, et = create_crf_transitions(label2id)
        self.crf.start_transitions.data = st
        self.crf.transitions.data = t
        self.crf.end_transitions.data = et

        # Expose config from base model
        self.config = base_model.config

        # Detect model type for special token handling
        # XLNet has special tokens at the end: ... <sep> <cls>
        # BERT-like models have special tokens at the start: [CLS] ... [SEP]
        model_type = getattr(self.config, 'model_type', '').lower()
        self.is_xlnet = model_type == 'xlnet'

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        token_type_ids=None,
        labels: torch.Tensor = None,
        **kwargs
    ):
        """Forward pass with CRF layer"""
        # Get predictions from base model
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            **kwargs
        )

        if labels is not None:
            logits = output.logits
            mask = labels != -100
            labels_crf = labels.clone()
            labels_crf[~mask] = 0

            if self.is_xlnet:
                # XLNet: skip last 2 tokens (<sep> <cls>)
                # Sequence: [token1, token2, ..., tokenN, <sep>, <cls>]
                logits_for_crf = logits[:, :-2, :]
                labels_for_crf = labels_crf[:, :-2]
                mask_for_crf = mask[:, :-2]
                # Ensure first timestep mask is True for CRF
                mask_for_crf[:, 0] = True
            else:
                # BERT-like: skip first token ([CLS])
                # Sequence: [CLS, token1, token2, ..., tokenN, SEP]
                logits_for_crf = logits[:, 1:, :]
                labels_for_crf = labels_crf[:, 1:]
                mask_for_crf = mask[:, 1:]
                # Ensure first timestep mask is True for CRF
                mask_for_crf[:, 0] = True

            loss = -self.crf(
                logits_for_crf,
                labels_for_crf,
                mask=mask_for_crf,
                reduction="mean",
            )
            output["loss"] = loss
        return output

    def decode(self, logits: torch.Tensor, attention_mask: torch.Tensor = None) -> list:
        """
        Use Viterbi decoding to get the best tag sequence

        Args:
            logits: Model output logits (batch_size, seq_len, num_labels)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of predicted tag sequences
        """
        if self.is_xlnet:
            # XLNet: skip last 2 tokens (<sep> <cls>)
            logits_for_crf = logits[:, :-2, :]
            if attention_mask is None:
                mask = torch.ones(logits_for_crf.shape[:2], dtype=torch.bool, device=logits.device)
            else:
                mask = attention_mask[:, :-2].bool()
        else:
            # BERT-like: skip first token ([CLS])
            logits_for_crf = logits[:, 1:, :]
            if attention_mask is None:
                mask = torch.ones(logits_for_crf.shape[:2], dtype=torch.bool, device=logits.device)
            else:
                mask = attention_mask[:, 1:].bool()

        # CRF requires first timestep mask to be True
        mask[:, 0] = True

        # Use CRF decode (Viterbi algorithm)
        decoded = self.crf.decode(logits_for_crf, mask=mask)

        # Pad the decoded sequences to match original sequence length
        result = []
        for seq in decoded:
            if self.is_xlnet:
                # XLNet: append O tags for <sep> <cls> at the end
                result.append(seq + [0, 0])
            else:
                # BERT-like: prepend O tag for [CLS] at the start
                result.append([0] + seq)
        return result

    def save_pretrained(self, save_directory, **kwargs):
        """Save the base model and CRF weights"""
        self.base_model.save_pretrained(save_directory, **kwargs)
        crf_path = Path(save_directory) / "crf_weights.pt"
        torch.save({
            'crf_state_dict': self.crf.state_dict(),
            'label2id': self.label2id,
        }, crf_path)

    @classmethod
    def from_pretrained(cls, model_path, label2id=None, **kwargs):
        """Load CRF wrapper from pretrained checkpoint"""
        base_model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        crf_path = Path(model_path) / "crf_weights.pt"
        if crf_path.exists():
            crf_data = torch.load(crf_path)
            if label2id is None:
                label2id = crf_data['label2id']
        else:
            if label2id is None:
                label2id = base_model.config.label2id

        wrapper = cls(base_model, label2id)
        if crf_path.exists():
            wrapper.crf.load_state_dict(crf_data['crf_state_dict'])
        return wrapper


class BertWithCrfForTokenClassification(BertForTokenClassification):
    """BERT model with CRF layer for token classification"""

    def __init__(self, config: PretrainedConfig):
        """Initialize BERT with CRF layer"""
        super().__init__(config)
        if CRF is None:
            raise ImportError(
                "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
            )
        # Define CRF layer
        self.crf = CRF(len(config.label2id), batch_first=True)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize weights with defined transition scores"""
        super()._init_weights(module)
        if isinstance(module, CRF):
            # Initialize CRF with valid transitions
            st, t, et = create_crf_transitions(self.config.label2id)
            module.start_transitions.data = st
            module.transitions.data = t
            module.end_transitions.data = et

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask,
        token_type_ids,
        labels: torch.Tensor,
    ):
        """Forward pass with CRF layer"""
        # Get predictions from BERT
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if labels is not None:
            logits = output.logits
            mask = labels != -100
            labels *= mask
            # Ensure first token is valid (skip CLS)
            mask[:, 1] = 1
            # Calculate CRF loss
            output["loss"] = -self.crf(
                logits[:, 1:, :],
                labels[:, 1:],
                mask=mask[:, 1:],
                reduction="mean",
            )
        return output

    def create_transitions(self, label2id):
        """Define transition scores for CRF layer (legacy, calls shared function)"""
        return create_crf_transitions(label2id)

    def decode(self, logits: torch.Tensor, attention_mask: torch.Tensor = None) -> list:
        """
        Use Viterbi decoding to get the best tag sequence

        Args:
            logits: Model output logits (batch_size, seq_len, num_labels)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of predicted tag sequences
        """
        # Skip CLS token (index 0) to match training behavior
        logits_no_cls = logits[:, 1:, :]

        if attention_mask is None:
            mask = torch.ones(logits_no_cls.shape[:2], dtype=torch.bool, device=logits.device)
        else:
            mask = attention_mask[:, 1:].bool()

        # CRF requires first timestep mask to be True
        mask[:, 0] = True

        # Use CRF decode (Viterbi algorithm)
        decoded = self.crf.decode(logits_no_cls, mask=mask)

        # Pad the decoded sequences to include CLS position (prepend 0/O tag)
        result = []
        for seq in decoded:
            result.append([0] + seq)  # Prepend O tag for CLS token
        return result


class HistoryDocumentResult:
    """Unified training result format class for saving and comparing experiment results"""

    def __init__(
        self,
        model_name: str,
        model_key: str = None,
        status: str = "success",
        output_dir: str = None,
    ):
        """
        Initialize result object

        Args:
            model_name: Model name or path
            model_key: Model short name (bert, roberta, etc.)
            status: Training status (success/failed)
            output_dir: Output directory
        """
        self.model_name = model_name
        self.model_key = model_key or self._extract_model_key(model_name)
        self.status = status
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Hyperparameters
        self.hyperparameters = {}

        # Training information
        self.training_time = 0.0
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0

        # Evaluation metrics (overall)
        self.metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

        # Metrics per entity type
        self.metrics_per_type = {}

        # Detailed classification report
        self.classification_report = ""

        # Error message (if failed)
        self.error = None

        # Best model path
        self.best_model_path = None

    def _extract_model_key(self, model_name: str) -> str:
        """Extract short name from model name"""
        model_name_lower = model_name.lower()
        if 'bert' in model_name_lower:
            if 'deberta' in model_name_lower:
                return 'deberta'
            elif 'roberta' in model_name_lower:
                return 'roberta'
            elif 'albert' in model_name_lower:
                return 'albert'
            else:
                return 'bert'
        elif 'xlnet' in model_name_lower:
            return 'xlnet'
        elif 'luke' in model_name_lower:
            return 'luke'
        elif 'bilstm' in model_name_lower:
            return 'bilstm'
        elif 'rinna' in model_name_lower or 'gpt-neox' in model_name_lower:
            return 'rinna'
        elif 'elyza' in model_name_lower or 'llama' in model_name_lower:
            return 'llama'
        elif 'gpt-4' in model_name_lower or 'chatgpt' in model_name_lower:
            return 'chatgpt'
        elif 'claude' in model_name_lower:
            return 'claude'
        elif 'gpt' in model_name_lower:
            return 'gpt'
        else:
            return model_name.split('/')[-1].replace('-', '_')[:20]  # Use the last part of the model name

    @staticmethod
    def _format_display_name(result) -> str:
        """
        Format model display name, adding CRF, large, and other flags

        Args:
            result: HistoryDocumentResult object

        Returns:
            Formatted display name
        """
        model_key = result.model_key or ""
        model_name = (result.model_name or "").lower()

        display_name = model_key

        # Check if CRF is used
        use_crf = result.hyperparameters.get("use_crf", False)
        has_crf_in_key = "_crf" in model_key.lower() or "-crf" in model_key.lower()

        # Check if it is a large model
        is_large = "large" in model_name or "large" in model_key.lower()
        has_large_in_key = "large" in model_key.lower()

        # Check model type
        is_bilstm = "bilstm" in model_key.lower() or "bilstm" in model_name
        is_llama = any(x in model_key.lower() or x in model_name for x in ["llama", "rinna", "gpt-neox", "elyza"])
        is_remote_llm = any(x in model_key.lower() for x in ["chatgpt", "claude", "remote"])

        # Add suffix (if not already in model_key)
        if is_large and not has_large_in_key:
            display_name = f"{display_name}-large"
        if use_crf and not has_crf_in_key:
            display_name = f"{display_name}-crf"

        # Add model type flags (in square brackets)
        flags = []
        if is_bilstm and "bilstm" not in model_key.lower():
            flags.append("BiLSTM")
        elif is_llama and not any(x in model_key.lower() for x in ["llama", "rinna"]):
            flags.append("LLM")
        elif is_remote_llm and "remote" not in model_key.lower():
            flags.append("API")

        if flags:
            display_name = f"{display_name} [{'/'.join(flags)}]"

        return display_name

    def set_hyperparameters(self, **kwargs):
        """Set hyperparameters"""
        self.hyperparameters.update(kwargs)

    def set_training_info(
        self,
        training_time: float,
        num_train_samples: int = 0,
        num_val_samples: int = 0,
        num_test_samples: int = 0,
    ):
        """Set training information"""
        self.training_time = training_time
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

    def set_metrics(
        self,
        precision: float,
        recall: float,
        f1: float,
        accuracy: float,
    ):
        """Set overall evaluation metrics"""
        self.metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }

    def set_per_type_metrics(self, metrics_dict: dict):
        """Set metrics for each entity type"""
        self.metrics_per_type = metrics_dict

    def set_classification_report(self, report: str):
        """Set detailed classification report"""
        self.classification_report = report

    def set_error(self, error: str):
        """Set error message"""
        self.status = "failed"
        self.error = error

    def set_best_model_path(self, path: str):
        """Set best model path"""
        self.best_model_path = path

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        result = {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "status": self.status,
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "hyperparameters": self.hyperparameters,
            "training_time": self.training_time,
            "num_samples": {
                "train": self.num_train_samples,
                "val": self.num_val_samples,
                "test": self.num_test_samples,
            },
            "metrics": self.metrics,
            "metrics_per_type": self.metrics_per_type,
            "classification_report": self.classification_report,
            "best_model_path": self.best_model_path,
        }

        if self.error:
            result["error"] = self.error

        return result

    def save(self, output_dir: str = None):
        """Save result to JSON file"""
        save_dir = output_dir or self.output_dir
        if not save_dir:
            raise ValueError("No output directory specified")

        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result_file = output_path / "result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"✓ Result saved to: {result_file}")
        return str(result_file)

    @classmethod
    def load(cls, result_file: str):
        """Load result from JSON file"""
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = cls(
            model_name=data["model_name"],
            model_key=data.get("model_key"),
            status=data.get("status", "success"),
            output_dir=data.get("output_dir"),
        )

        result.timestamp = data.get("timestamp", "")
        result.hyperparameters = data.get("hyperparameters", {})
        result.training_time = data.get("training_time", 0.0)

        num_samples = data.get("num_samples", {})
        result.num_train_samples = num_samples.get("train", 0)
        result.num_val_samples = num_samples.get("val", 0)
        result.num_test_samples = num_samples.get("test", 0)

        result.metrics = data.get("metrics", {})
        result.metrics_per_type = data.get("metrics_per_type", {})
        result.classification_report = data.get("classification_report", "")
        result.best_model_path = data.get("best_model_path")
        result.error = data.get("error")

        return result

    def print_summary(self):
        """Print result summary"""
        print("\n" + "=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Status: {self.status}")
        if self.status == "success":
            print(f"F1 Score: {self.metrics['f1']:.4f}")
            print(f"Precision: {self.metrics['precision']:.4f}")
            print(f"Recall: {self.metrics['recall']:.4f}")
            print(f"Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"Training time: {self.training_time:.1f}s")
        elif self.error:
            print(f"Error: {self.error}")
        print("=" * 80)

    def get_entity_type_metrics(self, entity_type: str) -> dict:
        """
        Get metrics for a specific entity type

        Args:
            entity_type: Entity type name

        Returns:
            Dictionary containing precision, recall, f1-score, support
        """
        return self.metrics_per_type.get(entity_type, {})

    def get_top_entity_types(self, metric: str = "f1-score", top_k: int = 5) -> list:
        """
        Get the best performing entity types

        Args:
            metric: Sorting metric (precision, recall, f1-score)
            top_k: Return top k

        Returns:
            [(entity_type, score), ...] list
        """
        if not self.metrics_per_type:
            return []

        entity_scores = [
            (entity_type, metrics.get(metric, 0))
            for entity_type, metrics in self.metrics_per_type.items()
        ]
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:top_k]

    def get_worst_entity_types(self, metric: str = "f1-score", bottom_k: int = 5) -> list:
        """
        Get the worst performing entity types

        Args:
            metric: Sorting metric (precision, recall, f1-score)
            bottom_k: Return bottom k

        Returns:
            [(entity_type, score), ...] list
        """
        if not self.metrics_per_type:
            return []

        entity_scores = [
            (entity_type, metrics.get(metric, 0))
            for entity_type, metrics in self.metrics_per_type.items()
        ]
        entity_scores.sort(key=lambda x: x[1])
        return entity_scores[:bottom_k]

    def compare_with(self, other: 'HistoryDocumentResult') -> dict:
        """
        Compare with another result

        Args:
            other: Another HistoryDocumentResult object

        Returns:
            Comparison result dictionary
        """
        comparison = {
            "models": {
                "this": self.model_key,
                "other": other.model_key,
            },
            "metrics_comparison": {},
            "training_time_comparison": {
                "this": self.training_time,
                "other": other.training_time,
                "diff": self.training_time - other.training_time,
            },
        }

        # Compare overall metrics
        for metric in ["precision", "recall", "f1", "accuracy"]:
            this_value = self.metrics.get(metric, 0)
            other_value = other.metrics.get(metric, 0)
            comparison["metrics_comparison"][metric] = {
                "this": this_value,
                "other": other_value,
                "diff": this_value - other_value,
                "improvement": ((this_value - other_value) / other_value * 100) if other_value > 0 else 0,
            }

        return comparison

    def plot_metrics_bar(self, save_path: str = None, show: bool = True):
        """
        Plot overall metrics bar chart

        Args:
            save_path: Save path (optional)
            show: Whether to display the chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except ImportError:
            print("Error: matplotlib is required. Please run: pip install matplotlib")
            return

        if self.status != "success":
            print("Warning: Result status is not success, cannot plot")
            return

        metrics = ["precision", "recall", "f1", "accuracy"]
        values = [self.metrics.get(m, 0) for m in metrics]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.ylim(0, 1.0)
        plt.ylabel("Score", fontsize=12)
        plt.title(f"Model Performance: {self.model_key}", fontsize=14, fontweight='bold')

        # Display values on bar chart
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def plot_entity_type_metrics(self, metric: str = "f1-score", save_path: str = None, show: bool = True):
        """
        Plot per-entity-type metrics bar chart

        Args:
            metric: Metric to plot (precision, recall, f1-score)
            save_path: Save path (optional)
            show: Whether to display the chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            # Support Unicode display
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("Error: matplotlib is required. Please run: pip install matplotlib")
            return

        if not self.metrics_per_type:
            print("Warning: No per-entity-type metrics data")
            return

        # Extract data
        entity_types = list(self.metrics_per_type.keys())
        values = [self.metrics_per_type[et].get(metric, 0) for et in entity_types]

        # Sort by value
        sorted_pairs = sorted(zip(entity_types, values), key=lambda x: x[1], reverse=True)
        entity_types, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

        plt.figure(figsize=(12, max(6, len(entity_types) * 0.4)))
        bars = plt.barh(entity_types, values, color='#3498db')
        plt.xlim(0, 1.0)
        plt.xlabel(metric.capitalize(), fontsize=12)
        plt.title(f"Per Entity Type {metric.capitalize()}: {self.model_key}",
                 fontsize=14, fontweight='bold')

        # Display values on bar chart
        for bar, value in zip(bars, values):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.4f}',
                    ha='left', va='center', fontsize=9, color='black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    @staticmethod
    def compare_multiple(results: list, metric: str = "f1", save_path: str = None, show: bool = True):
        """
        Compare performance of multiple models

        Args:
            results: List of HistoryDocumentResult objects
            metric: Metric to compare
            save_path: Save path (optional)
            show: Whether to display the chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("Error: matplotlib is required. Please run: pip install matplotlib")
            return

        if not results:
            print("Warning: No results to compare")
            return

        # Extract data
        successful_results = [r for r in results if r.status == "success"]
        if not successful_results:
            print("Warning: No successful results")
            return

        model_names = [HistoryDocumentResult._format_display_name(r) for r in successful_results]
        values = [r.metrics.get(metric, 0) for r in successful_results]

        # Sort by value
        sorted_pairs = sorted(zip(model_names, values), key=lambda x: x[1], reverse=True)
        model_names, values = zip(*sorted_pairs)

        plt.figure(figsize=(14, max(8, len(model_names) * 0.6)))
        bars = plt.barh(model_names, values, color='#2ecc71', height=0.6)
        plt.xlim(0, 1.0)
        plt.xlabel(metric.upper(), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f"Model Comparison: {metric.upper()}", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest score on top

        # Display values on bar chart
        for bar, value in zip(bars, values):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.4f}',
                    ha='left', va='center', fontsize=10, color='black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_training_time_comparison(results: list, save_path: str = None, show: bool = True):
        """
        Compare training time of multiple models

        Args:
            results: List of HistoryDocumentResult objects
            save_path: Save path (optional)
            show: Whether to display the chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("Error: matplotlib is required. Please run: pip install matplotlib")
            return

        if not results:
            print("Warning: No results to compare")
            return

        # Extract data
        successful_results = [r for r in results if r.status == "success"]
        if not successful_results:
            print("Warning: No successful results")
            return

        model_names = [HistoryDocumentResult._format_display_name(r) for r in successful_results]
        times = [r.training_time for r in successful_results]

        # Sort by time
        sorted_pairs = sorted(zip(model_names, times), key=lambda x: x[1])
        model_names, times = zip(*sorted_pairs)

        plt.figure(figsize=(14, max(8, len(model_names) * 0.6)))
        bars = plt.barh(model_names, times, color='#e74c3c', height=0.6)
        plt.xlabel("Training Time (seconds)", fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title("Training Time Comparison", fontsize=14, fontweight='bold')

        # Display values on bar chart
        for bar, time in zip(bars, times):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{time:.1f}s',
                    ha='left', va='center', fontsize=10, color='black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    @staticmethod
    def create_comparison_report(results: list, output_dir: str = None):
        """
        Create a complete comparison report (CSV + multiple charts)

        Args:
            results: List of HistoryDocumentResult objects
            output_dir: Output directory
        """
        if not results:
            print("Warning: No results to compare")
            return

        import pandas as pd
        from pathlib import Path

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(".")

        # Create DataFrame
        successful_results = [r for r in results if r.status == "success"]
        if not successful_results:
            print("Warning: No successful results")
            return

        report_data = []
        for r in successful_results:
            report_data.append({
                "Model": r.model_key,
                "Model Full Name": r.model_name,
                "F1": r.metrics["f1"],
                "Precision": r.metrics["precision"],
                "Recall": r.metrics["recall"],
                "Accuracy": r.metrics["accuracy"],
                "Training Time (s)": r.training_time,
                "Training Samples": r.num_train_samples,
                "Test Samples": r.num_test_samples,
            })

        df = pd.DataFrame(report_data)
        df = df.sort_values("F1", ascending=False)

        # Save CSV
        csv_file = output_path / "comparison_report.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"✓ CSV report saved: {csv_file}")

        # Generate charts
        print("\nGenerating comparison charts...")

        # 1. F1 comparison
        HistoryDocumentResult.compare_multiple(
            successful_results,
            metric="f1",
            save_path=str(output_path / "f1_comparison.png"),
            show=False
        )

        # 2. Precision comparison
        HistoryDocumentResult.compare_multiple(
            successful_results,
            metric="precision",
            save_path=str(output_path / "precision_comparison.png"),
            show=False
        )

        # 3. Recall comparison
        HistoryDocumentResult.compare_multiple(
            successful_results,
            metric="recall",
            save_path=str(output_path / "recall_comparison.png"),
            show=False
        )

        # 4. Training time comparison
        HistoryDocumentResult.plot_training_time_comparison(
            successful_results,
            save_path=str(output_path / "training_time_comparison.png"),
            show=False
        )

        print(f"\n✓ Full report generated at: {output_path}")
        print(f"  - comparison_report.csv")
        print(f"  - f1_comparison.png")
        print(f"  - precision_comparison.png")
        print(f"  - recall_comparison.png")
        print(f"  - training_time_comparison.png")


class HistoryDocumentModel:
    """Base wrapper class for NER models with tokenizer and data collator

    This is an abstract base class that provides common interface for different model types.
    Subclasses should implement model loading logic specific to their architecture.
    """

    def __init__(self, model_name: str, label2id: dict, id2label: dict, **kwargs):
        """
        Initialize model wrapper

        Args:
            model_name: Model name or path (supports short names from MODEL_CONFIGS)
            label2id: Label to ID mapping
            id2label: ID to label mapping
            **kwargs: Additional model-specific arguments
        """
        # Resolve short model name to full HuggingFace ID
        self.model_name = resolve_model_name(model_name)
        self.label2id = label2id
        # Ensure id2label keys are integers
        if id2label and isinstance(list(id2label.keys())[0], str):
            self.id2label = {int(k): v for k, v in id2label.items()}
        else:
            self.id2label = id2label

        # To be set by subclasses
        self.model = None
        self.tokenizer = None
        self.data_collator = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load model from pretrained checkpoint

        Args:
            model_path: Path to pretrained model
            **kwargs: Additional model-specific arguments
        """
        raise NotImplementedError("Subclasses must implement from_pretrained method")

    def _load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_model method")

    def _load_tokenizer(self):
        """Load the tokenizer - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_tokenizer method")

    def _create_data_collator(self):
        """Create data collator - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_data_collator method")


class HistoryDocumentModelBERT(HistoryDocumentModel):
    """BERT-based model wrapper for NER with optional CRF layer"""

    def __init__(self, model_name: str, label2id: dict, id2label: dict, use_crf: bool = False):
        """
        Initialize BERT model wrapper

        Args:
            model_name: BERT model name or path (e.g., 'cl-tohoku/bert-base-japanese-v3')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            use_crf: Whether to use CRF layer on top of BERT (default: False)
        """
        super().__init__(model_name, label2id, id2label)
        self.use_crf = use_crf

        # Load model, tokenizer, and data collator
        self._load_model()
        self._load_tokenizer()
        self._create_data_collator()

    def _load_model(self):
        """Load model with or without CRF layer (supports all BERT-like models)"""
        # First, load the base model using AutoModelForTokenClassification
        # This automatically handles BERT, RoBERTa, DeBERTa, ALBERT, LUKE, etc.
        print(f"Loading model: {self.model_name}")
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True
        )

        if self.use_crf:
            if CRF is None:
                raise ImportError(
                    "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
                )
            print(f"Adding CRF layer on top of {self.model_name}")
            self.model = CRFWrapper(base_model, self.label2id)
        else:
            self.model = base_model

    def _load_tokenizer(self):
        """Load BERT tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _create_data_collator(self):
        """Create data collator for token classification"""
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    @classmethod
    def from_pretrained(cls, model_path: str, use_crf: bool = False):
        """
        Load model from pretrained checkpoint (supports all BERT-like models with CRF)

        Args:
            model_path: Path to pretrained model
            use_crf: Whether the model uses CRF layer
        """
        if use_crf:
            if CRF is None:
                raise ImportError(
                    "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
                )
            model = CRFWrapper.from_pretrained(model_path)
            label2id = model.label2id
            id2label = {v: k for k, v in label2id.items()}
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            label2id = model.config.label2id
            id2label = model.config.id2label

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.model = model
        instance.tokenizer = tokenizer
        instance.label2id = label2id
        instance.id2label = id2label
        instance.use_crf = use_crf
        instance.data_collator = DataCollatorForTokenClassification(tokenizer)

        return instance


# =============================================================================
# LLM Model Base Classes (Generative NER)
# =============================================================================

class HistoryDocumentModelLLM(HistoryDocumentModel):
    """Base class for LLM-based NER models (both local and remote)

    This class provides shared functionality for generative NER:
    - Prompt construction
    - Entity response parsing
    - BIO tag conversion

    All LLM models use span-based JSON output format instead of BIO tagging.
    """

    # Entity type descriptions (shared by all LLM models)
    ENTITY_DESCRIPTIONS = {
        "役者": "歌舞伎役者の名前",
        "興行関係者": "劇場や興行に関わる人や組織",
        "俳名": "俳号",
        "演目名": "演目や作品のタイトル",
        "人名": "一般的な人名",
        "書名": "書籍名",
        "狂言作者": "狂言作者の名前",
        "役名": "役柄の名前",
        "屋号": "屋号",
        "音曲": "音曲関連",
        "事項": "その他の固有名詞",
    }

    def __init__(self, model_name: str, label2id: dict, id2label: dict, **kwargs):
        """Initialize LLM model wrapper

        Args:
            model_name: Model name or path
            label2id: Label to ID mapping
            id2label: ID to label mapping
        """
        super().__init__(model_name, label2id, id2label)
        self.label_list = [label for label in label2id.keys()]

    def _get_entity_types(self) -> List[str]:
        """Extract entity types from label list"""
        entity_types = set()
        for label in self.label_list:
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]
                entity_types.add(entity_type)
        return list(entity_types)

    def _build_ner_prompt(self, text: str, entities: list = None,
                          examples: List[dict] = None, is_training: bool = False) -> str:
        """Build NER prompt in unified format

        This method creates prompts compatible with all LLM models.

        Args:
            text: Input text to extract entities from
            entities: Ground truth entities (for training)
            examples: Few-shot examples (list of dicts with 'text' and 'entities')
            is_training: Whether this is for training (includes answer)

        Returns:
            Formatted prompt string
        """
        entity_types = self._get_entity_types()

        # Build prompt
        prompt = """以下の日本古典演劇関連のテキストから固有表現を抽出してJSON形式で出力してください。

固有表現の種類:
"""
        for etype in entity_types:
            desc = self.ENTITY_DESCRIPTIONS.get(etype, etype)
            prompt += f"- {etype}: {desc}\n"

        prompt += """
出力形式:
[{"type": "役者", "text": "三升", "span": [5, 7]}, {"type": "演目名", "text": "伊豆日記", "span": [15, 19]}]

固有表現が見つからない場合は空のリスト []を出力してください。
"""

        # Add few-shot examples if provided
        if examples:
            prompt += "\n以下に例を示します：\n"
            for i, ex in enumerate(examples, 1):
                ex_text = ex.get('text', '')
                ex_entities = ex.get('entities', [])
                entities_json = json.dumps(ex_entities, ensure_ascii=False)
                prompt += f"\n例{i}:\nテキスト: {ex_text}\n回答: {entities_json}\n"
            prompt += "\n---\n"

        # Add the input text
        prompt += f"\nテキスト: {text}\n回答:"

        # Add answer for training
        if is_training and entities is not None:
            entities_json = json.dumps(entities, ensure_ascii=False)
            prompt += f" {entities_json}"

        return prompt

    def _parse_entity_response(self, response: str, text: str) -> List[dict]:
        """Parse LLM response to extract entities in span format

        Args:
            response: Raw response from LLM
            text: Original input text (for validation)

        Returns:
            List of entity dicts with 'type', 'text', 'span' keys
        """
        import re

        response = response.strip()

        # Remove markdown code blocks if present
        if '```' in response:
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if code_match:
                response = code_match.group(1).strip()

        # Try to find JSON array
        output_match = re.search(r'回答:\s*(\[[\s\S]*?\])', response)
        if output_match:
            array_str = output_match.group(1)
        else:
            # Find JSON arrays
            arrays = re.findall(r'\[(?:[^\[\]]*|\[(?:[^\[\]]*|\[[^\[\]]*\])*\])*\]', response)
            if arrays:
                array_str = arrays[-1]
            else:
                array_str = '[]'

        try:
            entities = json.loads(array_str)
            if not isinstance(entities, list):
                return []

            # Validate and clean entities
            valid_entities = []
            entity_types = self._get_entity_types()

            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                etype = entity.get('type', '')
                etext = entity.get('text', entity.get('name', ''))
                span = entity.get('span', [])

                # Validate entity type
                if etype not in entity_types:
                    continue

                # Validate span
                if not isinstance(span, list) or len(span) != 2:
                    continue

                start, end = span
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue

                # Validate span within text bounds
                if start < 0 or end > len(text) or start >= end:
                    continue

                valid_entities.append({
                    'type': etype,
                    'text': etext,
                    'span': [start, end],
                    'name': etext
                })

            return valid_entities

        except json.JSONDecodeError:
            # Try line by line
            lines = response.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('['):
                    try:
                        entities = json.loads(line)
                        if isinstance(entities, list):
                            valid = []
                            for e in entities:
                                if isinstance(e, dict) and 'type' in e and 'span' in e:
                                    valid.append(e)
                            return valid
                    except json.JSONDecodeError:
                        continue

        return []

    def _entities_to_bio_tags(self, text: str, entities: List[dict]) -> List[str]:
        """Convert entities to BIO tags for evaluation

        Args:
            text: Original text
            entities: Entities with span info

        Returns:
            List of BIO tags (one per character)
        """
        tags = ['O'] * len(text)

        for entity in entities:
            etype = entity.get('type', '')
            span = entity.get('span', [])

            if len(span) != 2:
                continue

            start, end = span

            # Mark B- tag at start position
            if 0 <= start < len(tags):
                tag = f'B-{etype}'
                if tag in self.label2id:
                    tags[start] = tag

            # Mark I- tags for rest of entity
            for i in range(start + 1, min(end, len(tags))):
                tag = f'I-{etype}'
                if tag in self.label2id:
                    tags[i] = tag

        return tags

    def _convert_item_format(self, item: dict) -> dict:
        """Convert text+entities format to include tokens and ner_tags

        Args:
            item: Dict with 'text' and 'entities' keys

        Returns:
            Dict with 'tokens', 'ner_tags', 'text', 'entities' keys
        """
        if 'tokens' in item and 'ner_tags' in item:
            return item

        text = item.get('text', '')
        entities = item.get('entities', [])

        tokens = list(text)  # Character-level tokenization
        ner_tags = self._entities_to_bio_tags(text, entities)

        return {
            'tokens': tokens,
            'ner_tags': ner_tags,
            'text': text,
            'entities': entities,
        }

    def compute_seqeval_metrics(
        self,
        all_true_entities: List[List[dict]],
        all_pred_entities: List[List[dict]],
        all_texts: List[str],
    ) -> dict:
        """Compute seqeval metrics by converting entities to character-level BIO tags

        This is the unified evaluation method for all LLM models (LLaMA, Claude, ChatGPT).
        It converts JSON entity lists to character-level BIO tags and uses seqeval for evaluation.

        Args:
            all_true_entities: List of true entity lists for each sample
            all_pred_entities: List of predicted entity lists for each sample
            all_texts: List of input texts (needed for BIO tag length)

        Returns:
            Dict with precision, recall, f1, accuracy, and classification report
        """
        true_labels = []
        pred_labels = []

        for text, true_ents, pred_ents in zip(all_texts, all_true_entities, all_pred_entities):
            # Convert entities to BIO tags
            true_bio = self._entities_to_bio_tags(text, true_ents)
            pred_bio = self._entities_to_bio_tags(text, pred_ents)

            # Ensure same length (should already be same, but just in case)
            min_len = min(len(true_bio), len(pred_bio))
            true_labels.append(true_bio[:min_len])
            pred_labels.append(pred_bio[:min_len])

        # Use seqeval metrics
        report_str = classification_report(true_labels, pred_labels, digits=4, output_dict=False)
        report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)

        # Extract per-type metrics
        per_type_metrics = {}
        for entity_type, entity_metrics in report_dict.items():
            if entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                if isinstance(entity_metrics, dict) and 'precision' in entity_metrics:
                    per_type_metrics[entity_type] = {
                        "precision": float(entity_metrics['precision']),
                        "recall": float(entity_metrics['recall']),
                        "f1-score": float(entity_metrics['f1-score']),
                        "support": int(entity_metrics['support'])
                    }

        return {
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
            'accuracy': accuracy_score(true_labels, pred_labels),
            'classification_report': report_str,
            'per_type_metrics': per_type_metrics,
        }


class HistoryDocumentModelLocalLLM(HistoryDocumentModelLLM):
    """Base class for locally-run LLM models (LLaMA, etc.)

    Local LLMs require model loading, tokenization, and inference on local hardware.
    """
    pass


class HistoryDocumentModelLLaMA(HistoryDocumentModelLocalLLM):
    """LLaMA-based model wrapper for NER with LoRA and optional quantization"""

    def __init__(self, model_name: str, label2id: dict, id2label: dict,
                 use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, use_4bit: bool = False):
        """
        Initialize LLaMA model wrapper

        Args:
            model_name: LLaMA model name or path (e.g., 'rinna/japanese-gpt-neox-3.6b')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning (default: True)
            lora_r: LoRA rank (default: 16)
            lora_alpha: LoRA alpha scaling parameter (default: 32)
            lora_dropout: LoRA dropout rate (default: 0.05)
            use_4bit: Whether to use 4-bit quantization (default: False)
        """
        super().__init__(model_name, label2id, id2label)
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit

        # Load model, tokenizer, and data collator
        self._load_model()
        self._load_tokenizer()
        self._create_data_collator()

    def _load_model(self):
        """Load LLaMA model with optional LoRA and quantization"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "LLaMA support requires 'peft' and 'bitsandbytes'. "
                "Install with: pip install peft bitsandbytes"
            )

        import os

        # Check if this is a LoRA adapter path
        is_adapter_path = os.path.isdir(self.model_name) and os.path.exists(
            os.path.join(self.model_name, "adapter_config.json")
        )

        if is_adapter_path:
            print(f"✓ Detected LoRA adapter path: {self.model_name}")
            # Load adapter config to get base model
            import json
            with open(os.path.join(self.model_name, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            print(f"✓ Base model: {base_model_path}")

            # Configure quantization if needed
            bnb_config = None
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_4bit else torch.float32,
            )

            # Load LoRA adapter
            print(f"✓ Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.model_name)
            print(f"✅ Successfully loaded domain-adapted model!")

        else:
            # Load from scratch
            print(f"Loading LLaMA model: {self.model_name}")

            # Configure quantization
            bnb_config = None
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_4bit else torch.float32,
            )

            # Apply LoRA if requested
            if self.use_lora:
                if self.use_4bit:
                    self.model = prepare_model_for_kbit_training(self.model)

                # Auto-detect target modules
                target_modules = self._find_target_modules()
                print(f"Detected target modules for LoRA: {target_modules}")

                # Configure LoRA
                lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

    def _find_target_modules(self):
        """Auto-detect linear layers for LoRA"""
        model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else ""

        # Predefined target modules for different model types
        predefined_targets = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt_neox": ["query_key_value", "dense"],
            "gptneox": ["query_key_value", "dense"],
            "gpt2": ["c_attn", "c_proj"],
            "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        }

        if model_type in predefined_targets:
            return predefined_targets[model_type]

        # Auto-scan for linear layers
        print("Auto-scanning model for linear layers...")
        linear_modules = set()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]
                if module_name not in ['lm_head', 'embed_tokens', 'wte', 'wpe']:
                    linear_modules.add(module_name)

        if len(linear_modules) == 0:
            print("Warning: No linear layers found, using default config")
            return ["query_key_value", "dense"]

        return list(linear_modules)

    def _load_tokenizer(self):
        """Load LLaMA tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"  # GPT-style models need left padding
        )

        # Set pad_token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Update model config
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Resize embeddings if necessary
        if len(self.tokenizer) > self.model.config.vocab_size:
            print(f"Resizing model embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _create_data_collator(self):
        """Create data collator for causal language modeling"""
        from transformers import DataCollatorForLanguageModeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not MLM
        )

    @classmethod
    def from_pretrained(cls, model_path: str, use_lora: bool = True, use_4bit: bool = False):
        """
        Load LLaMA model from pretrained checkpoint

        Args:
            model_path: Path to pretrained LLaMA model or LoRA adapter
            use_lora: Whether to use LoRA
            use_4bit: Whether to use 4-bit quantization
        """
        import os
        import json

        try:
            from peft import PeftModel
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "LLaMA support requires 'peft' and 'bitsandbytes'. "
                "Install with: pip install peft bitsandbytes"
            )

        # Check if this is a LoRA adapter path
        is_adapter_path = os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "adapter_config.json")
        )

        if is_adapter_path:
            # Load adapter config to get base model
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")

            # Configure quantization if needed
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_4bit else torch.float32,
            )

            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load full model
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_4bit else torch.float32,
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.model = model
        instance.tokenizer = tokenizer
        instance.use_lora = use_lora
        instance.use_4bit = use_4bit

        # Set dummy label mappings (LLaMA uses generative approach, not token classification)
        instance.label2id = {}
        instance.id2label = {}

        # Create data collator
        from transformers import DataCollatorForLanguageModeling
        instance.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        return instance


class HistoryDocumentModelBiLSTM(HistoryDocumentModel):
    """BiLSTM-based model wrapper for NER (benchmark/baseline)

    Uses BERT tokenizer for consistency with transformer models, but with BiLSTM architecture.
    Supports optional CRF layer and character-level CNN features.
    """

    def __init__(
        self,
        model_name: str,
        label2id: dict,
        id2label: dict,
        use_crf: bool = False,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_char_cnn: bool = False,
        tokenizer_name: str = "cl-tohoku/bert-base-japanese-v3",
    ):
        """
        Initialize BiLSTM model wrapper

        Args:
            model_name: Model identifier (used for logging, e.g., 'bilstm')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            use_crf: Whether to use CRF layer on top of BiLSTM
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_char_cnn: Whether to use character-level CNN features
            tokenizer_name: Name of tokenizer to use (default: Japanese BERT tokenizer)
        """
        super().__init__(model_name, label2id, id2label)
        self.use_crf = use_crf
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_char_cnn = use_char_cnn
        self.tokenizer_name = tokenizer_name

        # Load tokenizer first (needed for vocab_size)
        self._load_tokenizer()

        # Load model
        self._load_model()

        # Create data collator
        self._create_data_collator()

    def _load_tokenizer(self):
        """Load tokenizer (uses BERT tokenizer for consistency)"""
        print(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _load_model(self):
        """Load BiLSTM model with optional CRF layer"""
        vocab_size = len(self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id or 0

        print(f"Creating BiLSTM model with vocab_size={vocab_size}, num_labels={len(self.label2id)}")
        print(f"  embedding_dim={self.embedding_dim}, hidden_dim={self.hidden_dim}")
        print(f"  num_layers={self.num_layers}, dropout={self.dropout}")
        print(f"  use_char_cnn={self.use_char_cnn}, use_crf={self.use_crf}")

        # Create BiLSTM model
        bilstm_model = BiLSTMForTokenClassification(
            vocab_size=vocab_size,
            num_labels=len(self.label2id),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_token_id=pad_token_id,
            use_char_cnn=self.use_char_cnn,
        )

        # Store label mappings in model config
        bilstm_model.config.label2id = self.label2id
        bilstm_model.config.id2label = self.id2label

        if self.use_crf:
            if CRF is None:
                raise ImportError(
                    "torchcrf is required for CRF layer. Install it with: pip install pytorch-crf"
                )
            print("Adding CRF layer on top of BiLSTM")
            self.model = BiLSTMWithCRF(bilstm_model, self.label2id)
        else:
            self.model = bilstm_model

    def _create_data_collator(self):
        """Create data collator for token classification"""
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    @classmethod
    def from_pretrained(cls, model_path: str, use_crf: bool = False):
        """
        Load BiLSTM model from pretrained checkpoint

        Args:
            model_path: Path to pretrained BiLSTM model
            use_crf: Whether the model uses CRF layer
        """
        # Load config to get parameters
        config_path = Path(model_path) / "bilstm_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        label2id = config.get("label2id", {})
        id2label = {int(v): k for k, v in label2id.items()} if label2id else {}

        # Load model
        if use_crf:
            model = BiLSTMWithCRF.from_pretrained(model_path, label2id=label2id)
        else:
            model = BiLSTMForTokenClassification.from_pretrained(model_path)

        # Load tokenizer (try to find saved tokenizer, fallback to default)
        tokenizer_path = Path(model_path)
        if (tokenizer_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.model = model
        instance.tokenizer = tokenizer
        instance.label2id = label2id
        instance.id2label = id2label
        instance.use_crf = use_crf
        instance.embedding_dim = config.get("embedding_dim", 256)
        instance.hidden_dim = config.get("hidden_dim", 256)
        instance.num_layers = config.get("num_layers", 2)
        instance.dropout = config.get("dropout", 0.3)
        instance.use_char_cnn = config.get("use_char_cnn", False)
        instance.data_collator = DataCollatorForTokenClassification(tokenizer)

        return instance


class HistoryDocumentDataset:
    """Wrapper for dataset with loading and tokenization"""

    def __init__(self, dataset_path: str):
        """
        Initialize dataset wrapper

        Args:
            dataset_path: Path to dataset
        """
        self.dataset_path = dataset_path
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from path (supports both HuggingFace dataset scripts and saved datasets)"""
        import os

        # Check if this is a saved dataset (has dataset_dict.json or state.json)
        is_saved_dataset = (
            os.path.exists(os.path.join(self.dataset_path, 'dataset_dict.json')) or
            os.path.exists(os.path.join(self.dataset_path, 'state.json'))
        )

        if is_saved_dataset:
            # Load using load_from_disk for saved datasets
            dataset = load_from_disk(self.dataset_path)
        else:
            try:
                # Try loading as HuggingFace dataset script
                dataset = load_dataset(self.dataset_path, trust_remote_code=True)
            except:
                # Fallback to load_from_disk
                dataset = load_from_disk(self.dataset_path)

        # Handle both DatasetDict and dict-like structures
        if isinstance(dataset, DatasetDict):
            self.train_data = dataset['train']
            self.val_data = dataset.get('validation', dataset.get('val', None))
            self.test_data = dataset.get('test', None)
        else:
            self.train_data = dataset.get('train', None)
            self.val_data = dataset.get('validation', dataset.get('val', None))
            self.test_data = dataset.get('test', None)

        if self.train_data is None:
            raise ValueError(f"No training data found in {self.dataset_path}")


class HistoryDocumentCorpus:
    """Wrapper for corpus data for MLM pretraining"""

    def __init__(self, corpus_file: str, max_length: int = 512, corpus_size: int = None, seed: int = 42, enable_split: bool = True):
        """
        Initialize corpus wrapper

        Args:
            corpus_file: Path to corpus file (JSON Lines format)
            max_length: Maximum sequence length (characters)
            corpus_size: Number of samples to randomly sample (None = use all)
            seed: Random seed for sampling
            enable_split: Whether to split long texts into chunks (no overlap)
        """
        self.corpus_file = corpus_file
        self.max_length = max_length
        self.corpus_size = corpus_size
        self.seed = seed
        self.enable_split = enable_split
        self.texts = []

        # Load corpus
        self._load_corpus()

    def _split_long_text(self, text: str) -> list:
        """
        Split long text into multiple non-overlapping segments at sentence boundaries

        Args:
            text: Original text

        Returns:
            List of text segments after splitting
        """
        import re

        if len(text) <= self.max_length:
            return [text]

        # Find all possible split points (sentence boundaries)
        # Common sentence-ending characters in Japanese/Chinese
        sentence_endings = re.finditer(r'[。．.！!？?」』\n]', text)
        split_points = [0] + [m.end() for m in sentence_endings] + [len(text)]

        # Greedy splitting: take the longest segment not exceeding max_length each time
        segments = []
        current_start = 0

        while current_start < len(text):
            best_end = current_start
            for point in split_points:
                if point <= current_start:
                    continue
                if point - current_start <= self.max_length:
                    best_end = point
                else:
                    break

            # If no suitable split point found (single sentence too long), force split
            if best_end == current_start:
                best_end = min(current_start + self.max_length, len(text))

            segment = text[current_start:best_end].strip()
            if segment:
                segments.append(segment)
            current_start = best_end

        return segments

    def _load_corpus(self):
        """Load corpus from file (JSON Lines format with 'aug' field, or plain text)"""
        import random

        # First pass: count total lines and detect format
        total_lines = 0
        is_json_format = None
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                # Detect format from first non-empty line
                if is_json_format is None and line.strip():
                    try:
                        data = json.loads(line.strip())
                        is_json_format = isinstance(data, dict) and 'aug' in data
                    except json.JSONDecodeError:
                        is_json_format = False

        if is_json_format:
            print(f"Detected JSON Lines format with 'aug' field")
        else:
            print(f"Detected plain text format")

        # If corpus_size specified and less than total, randomly sample
        if self.corpus_size is not None and self.corpus_size < total_lines:
            random.seed(self.seed)
            selected_indices = set(random.sample(range(total_lines), self.corpus_size))
            print(f"Randomly sampling {self.corpus_size} from {total_lines} lines (seed={self.seed})")
        else:
            selected_indices = None
            if self.corpus_size is not None:
                print(f"Requested {self.corpus_size} lines but corpus only has {total_lines}, using all")

        # Second pass: load selected lines
        raw_texts = []
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if selected_indices is None or idx in selected_indices:
                    line = line.strip()
                    if not line:
                        continue
                    if is_json_format:
                        try:
                            data = json.loads(line)
                            if 'aug' in data:
                                raw_texts.append(data['aug'])
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Plain text format: each line is a text sample
                        raw_texts.append(line)

        # Split long texts if enabled
        if self.enable_split:
            split_count = 0
            for text in raw_texts:
                if len(text) > self.max_length:
                    segments = self._split_long_text(text)
                    self.texts.extend(segments)
                    split_count += len(segments) - 1
                else:
                    self.texts.append(text)
            if split_count > 0:
                print(f"Split long texts: {len(raw_texts)} -> {len(self.texts)} (+{split_count})")
        else:
            self.texts = raw_texts

        print(f"Loaded {len(self.texts)} texts from corpus")

    def prepare_dataset(self, tokenizer) -> Dataset:
        """
        Prepare tokenized dataset for MLM training

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            Tokenized HuggingFace Dataset
        """
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        # Create dataset
        dataset = Dataset.from_dict({'text': self.texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing corpus",
        )

        return tokenized_dataset


class HistoryDocumentTrainer():
    """Base trainer class for NER tasks on historical documents

    This is an abstract base class that defines common interface for different model types.
    Subclasses should implement model-specific tokenization and label alignment logic.
    """

    def __init__(self, model: HistoryDocumentModel):
        """
        Initialize trainer

        Args:
            model: HistoryDocumentModel instance
        """
        self.model_wrapper = model
        self.compute_metrics_fn = None
        self.trainer = None

    def convert_entities_to_bio_tags(self, text, entities):
        """Convert entity annotations to BIO tags aligned with tokenizer

        This method should be overridden by subclasses to handle model-specific tokenization.
        """
        raise NotImplementedError("Subclasses must implement convert_entities_to_bio_tags method")

    def tokenize_and_align_labels(self, example):
        """Tokenize text and align BIO labels with tokens

        This method should be overridden by subclasses to handle model-specific tokenization.
        """
        raise NotImplementedError("Subclasses must implement tokenize_and_align_labels method")

    def create_compute_metrics(self, id2label):
        """Create compute_metrics function for seqeval evaluation"""
        # Ensure id2label keys are integers
        if id2label and isinstance(list(id2label.keys())[0], str):
            id2label = {int(k): v for k, v in id2label.items()}

        # Get model reference for CRF decoding
        model = self.model_wrapper.model
        has_crf_decode = hasattr(model, 'decode') and hasattr(model, 'crf')

        def compute_metrics(p):
            logits, labels = p

            # Check if model has CRF decode method
            if has_crf_decode:
                # Use CRF Viterbi decoding
                device = next(model.parameters()).device
                batch_size = 32

                all_predictions = []
                for i in range(0, len(logits), batch_size):
                    batch_logits = torch.tensor(logits[i:i+batch_size], device=device)
                    batch_labels = labels[i:i+batch_size]
                    batch_mask = torch.tensor(batch_labels != -100, dtype=torch.bool, device=device)

                    with torch.no_grad():
                        decoded = model.decode(batch_logits, batch_mask)
                    all_predictions.extend(decoded)

                # Convert decoded sequences to numpy array for consistent processing
                max_len = logits.shape[1]
                predictions = np.zeros((len(all_predictions), max_len), dtype=np.int64)
                for i, seq in enumerate(all_predictions):
                    seq_len = min(len(seq), max_len)
                    predictions[i, :seq_len] = seq[:seq_len]
            else:
                # Standard argmax decoding for non-CRF models
                predictions = np.argmax(logits, axis=2)

            # Convert ids to labels
            true_labels = []
            pred_labels = []

            for pred_seq, label_seq in zip(predictions, labels):
                true_seq = []
                pred_seq_labels = []

                for pred_id, label_id in zip(pred_seq, label_seq):
                    if label_id != -100:  # Ignore special tokens
                        true_seq.append(id2label.get(label_id, f"UNK-{label_id}"))
                        pred_seq_labels.append(id2label.get(pred_id, f"UNK-{pred_id}"))

                if true_seq:  # Only add non-empty sequences
                    true_labels.append(true_seq)
                    pred_labels.append(pred_seq_labels)

            # Calculate metrics using seqeval
            return {
                "precision": precision_score(true_labels, pred_labels),
                "recall": recall_score(true_labels, pred_labels),
                "f1": f1_score(true_labels, pred_labels),
                "accuracy": accuracy_score(true_labels, pred_labels),
            }

        return compute_metrics

    def evaluate_and_create_result(
        self,
        dataset: HistoryDocumentDataset,
        output_dir: str,
        model_name: str,
        hyperparameters: dict,
        training_time: float,
    ) -> HistoryDocumentResult:
        """
        Evaluate the test set and create a HistoryDocumentResult object

        Args:
            dataset: Dataset object
            output_dir: Output directory
            model_name: Model name
            hyperparameters: Hyperparameters dictionary
            training_time: Training duration (seconds)

        Returns:
            HistoryDocumentResult object
        """
        result = HistoryDocumentResult(
            model_name=model_name,
            output_dir=output_dir,
        )

        # Set hyperparameters
        result.set_hyperparameters(**hyperparameters)

        # Set sample counts
        result.set_training_info(
            training_time=training_time,
            num_train_samples=len(dataset.train_data) if dataset.train_data else 0,
            num_val_samples=len(dataset.val_data) if dataset.val_data else 0,
            num_test_samples=len(dataset.test_data) if dataset.test_data else 0,
        )

        if dataset.test_data is None or len(dataset.test_data) == 0:
            print("Warning: No test set, skipping evaluation")
            return result

        try:
            # Tokenize test data
            print("\nTokenizing the test set...")
            test_tokenized = dataset.test_data.map(
                lambda x: self.tokenize_and_align_labels(x),
                batched=False,
                desc="Tokenizing test data"
            )

            # Evaluate
            print("Evaluating test set...")
            test_results = self.trainer.evaluate(test_tokenized)

            # Get predictions for detailed report
            predictions_output = self.trainer.predict(test_tokenized)
            pred_logits = predictions_output.predictions
            label_ids = predictions_output.label_ids

            # Convert to labels
            id2label = self.model_wrapper.id2label
            # Ensure id2label keys are integers
            if id2label and isinstance(list(id2label.keys())[0], str):
                id2label = {int(k): v for k, v in id2label.items()}

            pred_labels = []
            true_labels = []

            # Check if model has CRF decode method
            model = self.model_wrapper.model
            has_crf_decode = hasattr(model, 'decode') and hasattr(model, 'crf')

            if has_crf_decode:
                print("CRF model detected, using Viterbi decoding...")
                # Use CRF Viterbi decoding
                # Convert logits to tensor and decode in batches
                device = next(model.parameters()).device
                batch_size = 32

                all_pred_ids = []
                for i in range(0, len(pred_logits), batch_size):
                    batch_logits = torch.tensor(pred_logits[i:i+batch_size], device=device)
                    # Create attention mask from label_ids (non -100 positions)
                    batch_labels = label_ids[i:i+batch_size]
                    batch_mask = torch.tensor(batch_labels != -100, dtype=torch.bool, device=device)

                    with torch.no_grad():
                        decoded = model.decode(batch_logits, batch_mask)
                    all_pred_ids.extend(decoded)

                for pred_ids, label_id in zip(all_pred_ids, label_ids):
                    pred_label = []
                    true_label = []
                    for idx, l_id in enumerate(label_id):
                        if l_id != -100:
                            p_id = pred_ids[idx] if idx < len(pred_ids) else 0
                            pred_label.append(id2label.get(p_id, f"UNK-{p_id}"))
                            true_label.append(id2label.get(l_id, f"UNK-{l_id}"))
                    if true_label:
                        pred_labels.append(pred_label)
                        true_labels.append(true_label)
            else:
                # Standard argmax decoding for non-CRF models
                for pred_logit, label_id in zip(pred_logits, label_ids):
                    pred_ids = np.argmax(pred_logit, axis=-1)
                    pred_label = []
                    true_label = []
                    for p_id, l_id in zip(pred_ids, label_id):
                        if l_id != -100:
                            pred_label.append(id2label.get(p_id, f"UNK-{p_id}"))
                            true_label.append(id2label.get(l_id, f"UNK-{l_id}"))
                    if true_label:  # Only add non-empty sequences
                        pred_labels.append(pred_label)
                        true_labels.append(true_label)

            # Get detailed classification report
            report_str = classification_report(true_labels, pred_labels, digits=4, output_dict=False)
            report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)

            # Set overall metrics
            result.set_metrics(
                precision=test_results.get("eval_precision", 0.0),
                recall=test_results.get("eval_recall", 0.0),
                f1=test_results.get("eval_f1", 0.0),
                accuracy=test_results.get("eval_accuracy", 0.0),
            )

            # Extract per-type metrics
            per_type_metrics = {}
            for entity_type, entity_metrics in report_dict.items():
                if entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                    if isinstance(entity_metrics, dict) and 'precision' in entity_metrics:
                        per_type_metrics[entity_type] = {
                            "precision": float(entity_metrics['precision']),
                            "recall": float(entity_metrics['recall']),
                            "f1-score": float(entity_metrics['f1-score']),
                            "support": int(entity_metrics['support'])
                        }

            result.set_per_type_metrics(per_type_metrics)
            result.set_classification_report(report_str)

            # Set best model path
            best_model_path = Path(output_dir) / "best_model"
            if best_model_path.exists():
                result.set_best_model_path(str(best_model_path))

            print("\n" + "=" * 80)
            print("Detailed Evaluation Results:")
            print("=" * 80)
            print(report_str)
            print("=" * 80)

        except Exception as e:
            print(f"\nError: Exception during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            result.set_error(str(e))

        return result

    def train(self, dataset: HistoryDocumentDataset, output_dir, seed=42, num_epochs=20,
              batch_size=32, learning_rate=1e-5, resume_from_checkpoint=None, save_model=True,
              early_stopping_patience=0):
        """
        Train model on dataset

        Args:
            dataset: HistoryDocumentDataset instance
            early_stopping_patience: Stop training if eval F1 doesn't improve for N epochs (0 = disabled)
            output_dir: Output directory
            seed: Random seed
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            resume_from_checkpoint: Path to checkpoint to resume from
            save_model: Whether to save model checkpoints (default: True)

        Returns:
            Trainer instance
        """
        set_seed(seed)

        # Tokenize datasets
        train_tokenized = dataset.train_data.map(
            lambda x: self.tokenize_and_align_labels(x),
            batched=False,
            desc="Tokenizing training data"
        )
        val_tokenized = dataset.val_data.map(
            lambda x: self.tokenize_and_align_labels(x),
            batched=False,
            desc="Tokenizing validation data"
        )

        # Initialize compute_metrics if id2label is available
        if self.model_wrapper.id2label:
            self.compute_metrics_fn = self.create_compute_metrics(self.model_wrapper.id2label)

        # Enable early stopping requires load_best_model_at_end=True
        use_early_stopping = early_stopping_patience > 0 and self.compute_metrics_fn is not None
        load_best = save_model or use_early_stopping

        # Initialize training arguments to pass to Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            warmup_steps=5,
            weight_decay=0.01,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch" if save_model else "no",
            logging_strategy="epoch",
            fp16=True,
            save_total_limit=1 if (save_model or use_early_stopping) else None,
            load_best_model_at_end=load_best,
            metric_for_best_model="f1" if (self.compute_metrics_fn and load_best) else None,
            greater_is_better=True if (self.compute_metrics_fn and load_best) else None,
            # Memory optimization for evaluation
            eval_accumulation_steps=4,
        )

        # Build callbacks list
        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            print(f"  Early stopping enabled (patience={early_stopping_patience} epochs)")

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model_wrapper.model,
            tokenizer=self.model_wrapper.tokenizer,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=self.model_wrapper.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics_fn,
            callbacks=callbacks if callbacks else None,
        )

        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return self.trainer

    def train_sequential(self, dataset_paths, epochs_per_dataset, output_base_dir, seed=42,
                        batch_size=32, learning_rate=1e-5):
        """
        Train model sequentially on multiple datasets

        Args:
            dataset_paths: List of dataset paths
            epochs_per_dataset: List of epoch counts per dataset
            output_base_dir: Base output directory
            seed: Random seed
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            List of stage results
        """
        set_seed(seed)

        if len(dataset_paths) != len(epochs_per_dataset):
            raise ValueError("Number of datasets must match number of epoch specifications")

        results = []

        for stage_idx, (dataset_path, num_epochs) in enumerate(zip(dataset_paths, epochs_per_dataset)):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}/{len(dataset_paths)}: Training on {dataset_path}")
            print(f"Epochs: {num_epochs}")
            print(f"{'='*60}\n")

            # Load dataset
            dataset = HistoryDocumentDataset(dataset_path)

            # Create stage output directory
            stage_dir = Path(output_base_dir) / f"stage{stage_idx + 1}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            # Train on this dataset and record time
            import time
            train_start_time = time.time()

            self.train(
                dataset=dataset,
                output_dir=str(stage_dir),
                seed=seed,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                resume_from_checkpoint=None,
                early_stopping_patience=early_stopping_patience,
            )

            stage_training_time = time.time() - train_start_time

            # Save stage model
            stage_model_dir = stage_dir / "final_model"
            self.trainer.save_model(str(stage_model_dir))
            # Also save tokenizer and config for loading in next stage
            if hasattr(self.model_wrapper, 'tokenizer') and self.model_wrapper.tokenizer is not None:
                self.model_wrapper.tokenizer.save_pretrained(str(stage_model_dir))
            if hasattr(self.model_wrapper, 'model') and hasattr(self.model_wrapper.model, 'config'):
                self.model_wrapper.model.config.save_pretrained(str(stage_model_dir))
            # For BiLSTM models, also call save_pretrained directly to ensure bilstm_config.json is saved
            if hasattr(self.model_wrapper.model, 'save_pretrained'):
                self.model_wrapper.model.save_pretrained(str(stage_model_dir))

            # Evaluate on test set if available
            if dataset.test_data is not None:
                test_tokenized = dataset.test_data.map(
                    lambda x: self.tokenize_and_align_labels(x),
                    batched=False,
                    desc=f"Tokenizing stage {stage_idx + 1} test data"
                )
                test_results = self.trainer.evaluate(test_tokenized)

                # For the last stage, also get detailed classification report
                if stage_idx == len(dataset_paths) - 1:
                    predictions = self.trainer.predict(test_tokenized)
                    label_ids = predictions.label_ids

                    # Check if model has CRF layer and use appropriate decoding
                    model = self.model_wrapper.model
                    has_crf = hasattr(model, 'crf') or hasattr(model, 'decode')

                    if has_crf and hasattr(model, 'decode'):
                        # Use CRF Viterbi decoding
                        import torch
                        device = next(model.parameters()).device
                        logits = torch.tensor(predictions.predictions).to(device)

                        # Use labels != -100 as mask (same as compute_metrics)
                        # This correctly masks out special tokens (CLS, SEP, PAD) and subword continuations
                        attention_mask = torch.tensor(label_ids != -100, dtype=torch.bool, device=device)

                        # Decode using CRF
                        pred_ids = model.decode(logits, attention_mask)
                        # Convert to numpy array with padding
                        max_len = max(len(seq) for seq in pred_ids)
                        pred_ids_padded = np.zeros((len(pred_ids), max_len), dtype=np.int64)
                        for i, seq in enumerate(pred_ids):
                            pred_ids_padded[i, :len(seq)] = seq
                        pred_ids = pred_ids_padded
                    else:
                        # Use argmax for non-CRF models
                        pred_ids = np.argmax(predictions.predictions, axis=2)

                    id2label = self.model_wrapper.model.config.id2label
                    # Ensure id2label keys are integers (keys may be strings in HuggingFace config)
                    if id2label and isinstance(list(id2label.keys())[0], str):
                        id2label = {int(k): v for k, v in id2label.items()}

                    true_labels = []
                    pred_labels = []

                    for pred, label in zip(pred_ids, label_ids):
                        true_label_seq = []
                        pred_label_seq = []
                        for p, l in zip(pred, label):
                            if l != -100:
                                true_label_seq.append(id2label.get(l, f"UNK-{l}"))
                                pred_label_seq.append(id2label.get(p, f"UNK-{p}"))
                        true_labels.append(true_label_seq)
                        pred_labels.append(pred_label_seq)

                    # Generate classification report
                    report_str = classification_report(true_labels, pred_labels, digits=4, output_dict=False)
                    report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)

                    # Store for later use
                    test_results["classification_report"] = report_str
                    test_results["classification_report_dict"] = report_dict
            else:
                test_results = {}

            stage_results = {
                "stage": stage_idx + 1,
                "dataset": dataset_path,
                "epochs": num_epochs,
                "training_time": stage_training_time,
                "test_results": test_results,
                "model_path": str(stage_model_dir)
            }
            results.append(stage_results)

            print(f"\nStage {stage_idx + 1} Results:")
            print(json.dumps(convert_numpy_types(test_results), indent=2))

            # Update model for next stage (if not last stage)
            if stage_idx < len(dataset_paths) - 1:
                # Use the same model class as current model
                model_class = type(self.model_wrapper)
                if model_class == HistoryDocumentModelBERT:
                    self.model_wrapper = HistoryDocumentModelBERT.from_pretrained(
                        str(stage_model_dir),
                        use_crf=self.model_wrapper.use_crf
                    )
                elif model_class == HistoryDocumentModelLLaMA:
                    self.model_wrapper = HistoryDocumentModelLLaMA.from_pretrained(
                        str(stage_model_dir),
                        use_lora=self.model_wrapper.use_lora
                    )
                elif model_class == HistoryDocumentModelBiLSTM:
                    self.model_wrapper = HistoryDocumentModelBiLSTM.from_pretrained(
                        str(stage_model_dir),
                        use_crf=self.model_wrapper.use_crf
                    )
                else:
                    raise ValueError(f"Unknown model class: {model_class}")

                # Update compute_metrics_fn to use the new model (important for CRF models)
                if self.model_wrapper.id2label:
                    self.compute_metrics_fn = self.create_compute_metrics(self.model_wrapper.id2label)

        # Save sequential training results (for debugging/analysis)
        final_results_path = Path(output_base_dir) / "sequential_training_results.json"
        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types({
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "stages": results,
                "total_stages": len(dataset_paths)
            }), f, ensure_ascii=False, indent=2)

        # Also save a standard result.json for the final stage (for evaluation compatibility)
        if results:
            last_stage = results[-1]
            last_test_results = last_stage["test_results"]

            # Create a HistoryDocumentResult for the final stage
            final_result = HistoryDocumentResult(
                model_name=self.model_wrapper.model_name,
                model_key=f"{Path(output_base_dir).name}",  # Use directory name as model key
                output_dir=output_base_dir
            )

            # Set training info
            final_result.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_result.hyperparameters = {
                "num_epochs": last_stage["epochs"],
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "two_stage_training": True,
                "stage1_dataset": results[0]["dataset"] if len(results) > 1 else None,
                "stage1_epochs": results[0]["epochs"] if len(results) > 1 else None,
                "stage2_dataset": last_stage["dataset"],
                "stage2_epochs": last_stage["epochs"],
            }

            # Set sample counts (from last stage dataset)
            final_dataset = HistoryDocumentDataset(last_stage["dataset"])
            final_result.num_train_samples = len(final_dataset.train_data) if final_dataset.train_data else 0
            final_result.num_val_samples = len(final_dataset.val_data) if final_dataset.val_data else 0
            final_result.num_test_samples = len(final_dataset.test_data) if final_dataset.test_data else 0

            # Set metrics from last stage
            # Prefer metrics from classification_report_dict (uses correct CRF decoding)
            # over eval_* metrics (which may use incorrect decoding for CRF models)
            if "classification_report_dict" in last_test_results:
                report_dict = last_test_results["classification_report_dict"]
                # Use micro avg for overall metrics (consistent with seqeval)
                if "micro avg" in report_dict:
                    micro_avg = report_dict["micro avg"]
                    final_result.set_metrics(
                        f1=micro_avg.get("f1-score", 0),
                        precision=micro_avg.get("precision", 0),
                        recall=micro_avg.get("recall", 0),
                        accuracy=last_test_results.get("eval_accuracy", 0)
                    )
                elif "eval_f1" in last_test_results:
                    # Fallback to eval metrics if micro avg not available
                    final_result.set_metrics(
                        f1=last_test_results["eval_f1"],
                        precision=last_test_results["eval_precision"],
                        recall=last_test_results["eval_recall"],
                        accuracy=last_test_results["eval_accuracy"]
                    )
            elif "eval_f1" in last_test_results:
                final_result.set_metrics(
                    f1=last_test_results["eval_f1"],
                    precision=last_test_results["eval_precision"],
                    recall=last_test_results["eval_recall"],
                    accuracy=last_test_results["eval_accuracy"]
                )

            # Set per-entity metrics if available
            if "classification_report_dict" in last_test_results:
                report_dict = last_test_results["classification_report_dict"]
                per_type_metrics = {}
                for entity_type, entity_metrics in report_dict.items():
                    if entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                        if isinstance(entity_metrics, dict) and 'precision' in entity_metrics:
                            per_type_metrics[entity_type] = {
                                "precision": float(entity_metrics['precision']),
                                "recall": float(entity_metrics['recall']),
                                "f1-score": float(entity_metrics['f1-score']),
                                "support": int(entity_metrics['support'])
                            }
                final_result.set_per_type_metrics(per_type_metrics)

            # Set classification report if available
            if "classification_report" in last_test_results:
                final_result.set_classification_report(last_test_results["classification_report"])

            # Calculate total training time (sum of all stages)
            # Note: This is an estimate since individual stage times aren't tracked
            # You may want to track this more precisely in future implementations
            total_training_time = sum([
                stage.get("training_time", 0) for stage in results
            ]) if any("training_time" in stage for stage in results) else 0
            final_result.training_time = total_training_time

            # Save the standard result.json
            final_result.save()
            print(f"Standard result.json saved to: {output_base_dir}/result.json")

        print(f"\n{'='*60}")
        print(f"Sequential training completed!")
        print(f"Results saved to: {final_results_path}")
        print(f"{'='*60}\n")

        return results

    def mlm_pretrain(
        self,
        corpus: HistoryDocumentCorpus,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        mlm_probability: float = 0.15,
        warmup_ratio: float = 0.1,
        seed: int = 42,
    ) -> str:
        """
        Perform MLM (Masked Language Model) pretraining on domain corpus

        Args:
            corpus: HistoryDocumentCorpus instance
            output_dir: Output directory for pretrained model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            mlm_probability: Probability of masking tokens
            warmup_ratio: Warmup ratio
            seed: Random seed

        Returns:
            Path to final pretrained model
        """
        set_seed(seed)

        print(f"\n{'='*80}")
        print(f"MLM Domain Adaptation Pretraining")
        print(f"{'='*80}")
        print(f"Base model: {self.model_wrapper.model_name}")
        print(f"Corpus: {corpus.corpus_file}")
        print(f"Output directory: {output_dir}")
        print(f"Training epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"MLM probability: {mlm_probability}")
        print(f"{'='*80}\n")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load MLM model (instead of token classification model)
        tokenizer = self.model_wrapper.tokenizer

        # Check model type
        config = AutoConfig.from_pretrained(self.model_wrapper.model_name)
        is_xlnet = config.model_type == 'xlnet'
        is_luke = config.model_type == 'luke'

        print(f"Loading MLM model...")
        if is_xlnet:
            print(f"  XLNet detected - using Causal LM")
            mlm_model = XLNetLMHeadModel.from_pretrained(self.model_wrapper.model_name)
        else:
            mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_wrapper.model_name)

        # Prepare dataset
        print(f"Preparing corpus dataset...")
        tokenized_corpus = corpus.prepare_dataset(tokenizer)

        # Split train/val (90/10)
        split_dataset = tokenized_corpus.train_test_split(test_size=0.1, seed=seed)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")

        # Data collator
        if is_xlnet:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM
            )
            print(f"Using Causal LM (next token prediction)")
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
            )
            print(f"Using Masked LM (mask probability: {mlm_probability})")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="linear",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            save_total_limit=1,
            load_best_model_at_end=False if is_luke else True,
            metric_for_best_model=None if is_luke else "eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=mlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        print(f"\nStarting MLM pretraining...")
        if is_luke:
            print(f"  Note: LUKE doesn't support load_best_model_at_end")

        train_result = trainer.train()

        # Save final model
        print(f"\nSaving pretrained model...")
        final_model_dir = output_path / "final_model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        # Save training info
        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)
        metrics['eval_samples'] = len(eval_dataset)

        info = {
            'base_model': self.model_wrapper.model_name,
            'model_type': 'xlnet' if is_xlnet else ('luke' if is_luke else 'mlm'),
            'corpus_file': corpus.corpus_file,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'mlm_probability': mlm_probability if not is_xlnet else 'N/A',
            'metrics': metrics,
        }

        info_file = output_path / "pretrain_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*80}")
        print(f"MLM Pretraining completed!")
        print(f"Final model: {final_model_dir}")
        print(f"Training info: {info_file}")
        print(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")

        if not is_luke:
            eval_metrics = trainer.evaluate()
            if 'eval_loss' in eval_metrics:
                print(f"Eval loss: {eval_metrics['eval_loss']:.4f}")

        print(f"Training time: {metrics.get('train_runtime', 0):.2f}s")
        print(f"{'='*80}\n")

        return str(final_model_dir)


class HistoryDocumentTrainerBERT(HistoryDocumentTrainer):
    """BERT-specific trainer implementation for NER tasks

    Implements BERT-specific tokenization, label alignment, and training logic.
    Uses spacy_alignments for precise character-to-token alignment.
    """

    def tokenize_and_align_labels(self, example):
        """Tokenize text and align BIO labels with BERT tokens using spacy_alignments

        This method uses spacy_alignments.get_alignments() for precise character-to-token
        mapping, which correctly handles cases where tokenizer normalizes characters
        (e.g., fullwidth ＝ to halfwidth =).
        """
        text = example['text']
        entities = example['entities']

        # Tokenize the text
        tokenized_inputs = self.model_wrapper.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_special_tokens_mask=True,
        )

        # Get tokens for alignment
        tokens = self.model_wrapper.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])

        # Use spacy_alignments for precise character-to-token mapping
        characters = list(text)
        char_to_token_indices, _ = get_alignments(characters, tokens)

        # Initialize all labels to O
        label2id = self.model_wrapper.label2id
        labels = [label2id['O']] * len(tokens)

        # Process each entity and assign BIO tags
        for entity in sorted(entities, key=lambda x: x['span'][0]):
            entity_start, entity_end = entity['span']
            entity_type = entity['type']

            # Skip invalid spans
            if entity_start >= len(char_to_token_indices) or entity_end > len(text):
                continue

            try:
                # Get token indices for entity start and end characters
                start_token_indices = char_to_token_indices[entity_start]
                end_token_indices = char_to_token_indices[entity_end - 1] if entity_end > 0 else []
            except IndexError:
                continue

            # Skip if character doesn't map to any token
            if len(start_token_indices) == 0 or len(end_token_indices) == 0:
                continue

            start_token = start_token_indices[0]
            end_token = end_token_indices[-1]

            # Assign B- tag to the first token
            b_label = f'B-{entity_type}'
            if b_label in label2id:
                labels[start_token] = label2id[b_label]

            # Assign I- tag to remaining tokens
            i_label = f'I-{entity_type}'
            if i_label in label2id and start_token != end_token:
                for token_idx in range(start_token + 1, end_token + 1):
                    if token_idx < len(labels):
                        labels[token_idx] = label2id[i_label]

        # Set special token positions to -100 (ignored in loss calculation)
        special_tokens_mask = tokenized_inputs.get('special_tokens_mask', [])
        for idx, is_special in enumerate(special_tokens_mask):
            if is_special:
                labels[idx] = -100

        # Remove special_tokens_mask from output (not needed for training)
        if 'special_tokens_mask' in tokenized_inputs:
            del tokenized_inputs['special_tokens_mask']

        tokenized_inputs['labels'] = labels
        return tokenized_inputs


# =============================================================================
# LLM Trainer Base Classes
# =============================================================================

class HistoryDocumentTrainerLLM(HistoryDocumentTrainer):
    """Base trainer class for LLM-based NER (both local and remote)

    This class provides common functionality for LLM-based NER training:
    - Few-shot example selection (random, diverse, stratified)
    - Format conversion utilities

    Subclasses:
    - HistoryDocumentTrainerLocalLLM: For locally-run LLMs (LLaMA)
    - HistoryDocumentTrainerRemoteLLM: For remote API-based LLMs (ChatGPT, Claude)
    """

    def __init__(self, model: 'HistoryDocumentModelLLM',
                 dataset: 'HistoryDocumentDataset' = None,
                 n_few_shot: int = 5,
                 few_shot_selection: str = "random",
                 **kwargs):
        """
        Initialize LLM trainer

        Args:
            model: HistoryDocumentModelLLM instance
            dataset: HistoryDocumentDataset instance
            n_few_shot: Number of few-shot examples to use
            few_shot_selection: Selection strategy ('random', 'diverse', 'stratified')
        """
        # Don't call super().__init__() as it expects specific model structure
        self.model_wrapper = model
        self.dataset = dataset
        self.n_few_shot = n_few_shot
        self.few_shot_selection = few_shot_selection
        self.few_shot_examples = []
        self.trainer = None  # May be used by subclasses
        self.compute_metrics_fn = None

    def _select_few_shot_examples(self, n_examples: int = None) -> List[dict]:
        """Select few-shot examples from training and validation data

        Args:
            n_examples: Number of examples to select (defaults to self.n_few_shot)

        Returns:
            List of few-shot examples with 'tokens' and 'labels' keys
        """
        import random

        n_examples = n_examples or self.n_few_shot

        if self.dataset is None or self.dataset.train_data is None:
            print("Warning: No training data available for few-shot selection")
            return []

        # Combine training and validation data for few-shot selection
        available_data = list(self.dataset.train_data)
        if self.dataset.val_data is not None:
            available_data.extend(list(self.dataset.val_data))
            print(f"Few-shot selection pool: {len(self.dataset.train_data)} train + {len(self.dataset.val_data)} val = {len(available_data)} samples")
        else:
            print(f"Few-shot selection pool: {len(available_data)} train samples")

        if len(available_data) == 0:
            return []

        if self.few_shot_selection == "random":
            selected = random.sample(available_data, min(n_examples, len(available_data)))
        elif self.few_shot_selection == "diverse":
            # Select examples with diverse entity types
            selected = self._select_diverse_examples(available_data, n_examples)
        elif self.few_shot_selection == "stratified":
            # Select examples to cover all entity types
            selected = self._select_stratified_examples(available_data, n_examples)
        else:
            selected = random.sample(available_data, min(n_examples, len(available_data)))

        # Convert to few-shot format
        examples = []
        for item in selected:
            # Convert format if needed (supports text+entities format)
            converted_item = self.model_wrapper._convert_item_format(item)
            tokens = converted_item['tokens']
            ner_tags = converted_item['ner_tags']

            # Convert IDs to labels if necessary
            if ner_tags and isinstance(ner_tags[0], int):
                labels = [self.model_wrapper.id2label[t] for t in ner_tags]
            else:
                labels = list(ner_tags)

            examples.append({
                'tokens': list(tokens),
                'labels': labels
            })

        return examples

    def _select_diverse_examples(self, train_data: List[dict], n_examples: int) -> List[dict]:
        """Select diverse examples covering different entity types"""
        import random

        # Group by entity types present
        entity_examples = {}
        for item in train_data:
            # Convert format if needed (supports text+entities format)
            converted_item = self.model_wrapper._convert_item_format(item)
            ner_tags = converted_item['ner_tags']
            if ner_tags and isinstance(ner_tags[0], int):
                tags = [self.model_wrapper.id2label[t] for t in ner_tags]
            else:
                tags = list(ner_tags) if ner_tags else []

            entity_types = set()
            for tag in tags:
                if tag.startswith('B-'):
                    entity_types.add(tag[2:])

            entity_key = tuple(sorted(entity_types))
            if entity_key not in entity_examples:
                entity_examples[entity_key] = []
            entity_examples[entity_key].append(item)

        # Select from different entity type combinations
        selected = []
        keys = list(entity_examples.keys())
        random.shuffle(keys)

        while len(selected) < n_examples and keys:
            for key in keys:
                if len(selected) >= n_examples:
                    break
                if entity_examples[key]:
                    selected.append(random.choice(entity_examples[key]))
                    entity_examples[key].remove(selected[-1])

        return selected

    def _select_stratified_examples(self, train_data: List[dict], n_examples: int) -> List[dict]:
        """Select examples prioritizing high-frequency entity types"""
        import random
        from collections import Counter

        # Count entity types across all data
        entity_counts = Counter()
        item_entity_info = []  # Store (item, entity_types, total_frequency)

        for item in train_data:
            # Convert format if needed (supports text+entities format)
            converted_item = self.model_wrapper._convert_item_format(item)
            ner_tags = converted_item['ner_tags']
            if ner_tags and isinstance(ner_tags[0], int):
                tags = [self.model_wrapper.id2label[t] for t in ner_tags]
            else:
                tags = list(ner_tags) if ner_tags else []

            entity_types = set()
            for tag in tags:
                if tag.startswith('B-'):
                    entity_type = tag[2:]
                    entity_types.add(entity_type)
                    entity_counts[entity_type] += 1

            # Calculate total frequency score for this item
            frequency_score = sum(entity_counts.get(et, 0) for et in entity_types)
            item_entity_info.append((item, entity_types, frequency_score))

        # Sort entity types by frequency (high to low)
        sorted_entity_types = [et for et, _ in entity_counts.most_common()]

        print(f"Entity type frequencies: {dict(entity_counts.most_common())}")

        # Phase 1: Ensure coverage of all entity types, prioritizing high-frequency ones
        selected = []
        covered_types = set()

        # Sort items by frequency score (descending)
        item_entity_info.sort(key=lambda x: x[2], reverse=True)

        # First, select items that cover uncovered high-frequency entity types
        for entity_type in sorted_entity_types:
            if len(selected) >= n_examples:
                break
            if entity_type in covered_types:
                continue

            # Find best item containing this entity type
            for item, entity_types, freq_score in item_entity_info:
                if item in selected:
                    continue
                if entity_type in entity_types:
                    selected.append(item)
                    covered_types.update(entity_types)
                    print(f"Selected example with {entity_types} (priority: {entity_type}, freq: {entity_counts[entity_type]})")
                    break

        # Phase 2: Fill remaining slots with high-frequency entity samples
        if len(selected) < n_examples:
            for item, entity_types, freq_score in item_entity_info:
                if len(selected) >= n_examples:
                    break
                if item not in selected and entity_types:
                    selected.append(item)
                    print(f"Selected additional example with {entity_types} (freq_score: {freq_score})")

        return selected

    def _print_few_shot_summary(self):
        """Print summary of selected few-shot examples"""
        if self.few_shot_examples:
            # Show selected entity types
            entity_types = set()
            for ex in self.few_shot_examples:
                # Support both formats: text+entities and tokens+labels
                if 'entities' in ex:
                    for entity in ex['entities']:
                        entity_types.add(entity.get('type', ''))
                elif 'labels' in ex:
                    for label in ex['labels']:
                        if label.startswith('B-'):
                            entity_types.add(label[2:])
                elif 'ner_tags' in ex:
                    # Handle converted format
                    converted = self.model_wrapper._convert_item_format(ex)
                    for tag in converted.get('ner_tags', []):
                        if isinstance(tag, str) and tag.startswith('B-'):
                            entity_types.add(tag[2:])
            print(f"✓ Covered entity types: {', '.join(sorted(entity_types))}")

    def convert_entities_to_bio_tags(self, text, entities):
        """Not used for LLM trainers - use model's method instead"""
        return self.model_wrapper._entities_to_bio_tags(text, entities)

    def tokenize_and_align_labels(self, example):
        """Must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement tokenize_and_align_labels")


class HistoryDocumentTrainerLocalLLM(HistoryDocumentTrainerLLM):
    """Base trainer class for locally-run LLM models (LLaMA, etc.)

    This class provides common training logic for local LLM models.
    """
    pass


class HistoryDocumentTrainerLLaMA(HistoryDocumentTrainerLocalLLM):
    """LLaMA-specific trainer implementation for NER tasks

    Implements generative NER using prompts and JSON output format.
    """

    def train(self, dataset: HistoryDocumentDataset, output_dir, seed=42, num_epochs=20,
              batch_size=32, learning_rate=1e-5, resume_from_checkpoint=None, save_model=True):
        """
        Train LLaMA model on dataset (overrides base class to disable compute_metrics)

        LLaMA uses generative NER, so standard token-classification metrics don't apply.
        Evaluation should be done separately using generate() and entity parsing.

        Args:
            dataset: HistoryDocumentDataset instance
            output_dir: Output directory
            seed: Random seed
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            resume_from_checkpoint: Path to checkpoint to resume from
            save_model: Whether to save model checkpoints (default: True)

        Returns:
            Trainer instance
        """
        set_seed(seed)

        # Tokenize datasets
        train_tokenized = dataset.train_data.map(
            lambda x: self.tokenize_and_align_labels(x),
            batched=False,
            desc="Tokenizing training data"
        )
        val_tokenized = dataset.val_data.map(
            lambda x: self.tokenize_and_align_labels(x),
            batched=False,
            desc="Tokenizing validation data"
        )

        # LLaMA uses generative NER - disable compute_metrics during training
        # (standard token classification metrics don't work for generative models)
        self.compute_metrics_fn = None

        # Training arguments (no compute_metrics for generative models)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            warmup_steps=5,
            weight_decay=0.01,
            num_train_epochs=num_epochs,
            evaluation_strategy="no",  # Disable eval during training for generative models
            save_strategy="epoch" if save_model else "no",
            logging_strategy="epoch",
            fp16=True,
            save_total_limit=1 if save_model else None,
            load_best_model_at_end=False,  # No eval metrics to compare
            # Memory optimization
            eval_accumulation_steps=4,
        )

        # Initialize Trainer without compute_metrics
        self.trainer = Trainer(
            model=self.model_wrapper.model,
            tokenizer=self.model_wrapper.tokenizer,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=self.model_wrapper.data_collator,
            args=training_args,
            compute_metrics=None,  # Disabled for generative models
        )

        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return self.trainer

    def format_ner_prompt(self, text: str, entities: list = None, is_training: bool = True):
        """
        Format NER task prompt for LLaMA

        Args:
            text: Input text
            entities: Entity list (used during training)
            is_training: Whether in training mode

        Returns:
            Formatted prompt string
        """
        # Build prompt with entity types
        prompt = f"""以下の日本古典演劇関連のテキストから固有表現を抽出してJSON形式で出力してください。

テキスト: {text}

固有表現の種類:
- 役者: 歌舞伎役者の名前
- 興行関係者: 劇場や興行に関わる人や組織
- 俳名: 俳号
- 演目名: 演目や作品のタイトル
- 人名: 一般的な人名
- 書名: 書籍名
- 狂言作者: 狂言作者の名前
- 役名: 役柄の名前
- 屋号: 屋号
- 音曲: 音曲関連
- 事項: その他の固有名詞

出力形式:
[{{"type": "役者", "text": "三升", "span": [5, 7]}}, {{"type": "演目名", "text": "伊豆日記", "span": [15, 19]}}]

固有表現が見つからない場合は空のリスト []を出力してください。

回答:"""

        if is_training and entities:
            # Add correct answer during training
            import json
            entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
            prompt += f" {entities_json}"

        return prompt

    def convert_entities_to_bio_tags(self, text, entities):
        """
        Convert entities to prompt format (not BIO tags for LLaMA)

        This method is required by base class but LLaMA doesn't use BIO tags.
        Instead, we format entities as JSON in the prompt.
        """
        # For LLaMA, we don't need BIO tags - we use the entities directly in prompts
        return entities

    def tokenize_and_align_labels(self, example):
        """
        Tokenize text and prepare for causal language modeling

        LLaMA uses generative NER, so we:
        1. Format prompt with text
        2. Add answer (entities JSON) for training
        3. Mask prompt part in labels (only compute loss on answer)
        """
        text = example['text']
        entities = example['entities']

        # Format prompt with answer
        prompt = self.format_ner_prompt(text, entities, is_training=True)

        # Get max length from model config (respecting position embedding limits)
        model_config = self.model_wrapper.model.config
        if hasattr(model_config, 'n_positions'):
            max_length = model_config.n_positions  # GPT-2 style
        elif hasattr(model_config, 'max_position_embeddings'):
            max_length = model_config.max_position_embeddings  # LLaMA style
        else:
            max_length = 512  # Safe default

        # Tokenize
        tokenized_inputs = self.model_wrapper.tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal LM, labels = input_ids
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Mask prompt part (only compute loss on answer)
        prompt_without_answer = self.format_ner_prompt(text, None, is_training=False)
        prompt_tokens = self.model_wrapper.tokenizer(
            prompt_without_answer,
            truncation=True,
            max_length=max_length
        )
        prompt_length = len(prompt_tokens["input_ids"])

        # Mask prompt part with -100
        tokenized_inputs["labels"][0, :prompt_length] = -100

        # Remove batch dimension for dataset.map()
        tokenized_inputs = {k: v.squeeze(0) for k, v in tokenized_inputs.items()}

        return tokenized_inputs

    def evaluate_and_create_result(
        self,
        dataset: HistoryDocumentDataset,
        output_dir: str,
        model_name: str,
        hyperparameters: dict,
        training_time: float,
    ) -> HistoryDocumentResult:
        """
        Evaluate LLaMA model using generative approach

        For generative NER, we:
        1. Generate responses for each test sample
        2. Parse JSON entities from generated text
        3. Compare with ground truth entities
        4. Calculate F1/precision/recall

        Args:
            dataset: HistoryDocumentDataset instance
            output_dir: Output directory for results
            model_name: Model name for reporting
            hyperparameters: Training hyperparameters
            training_time: Training time in seconds

        Returns:
            HistoryDocumentResult with evaluation metrics
        """
        from tqdm import tqdm

        # Create result object (same as base class)
        result = HistoryDocumentResult(
            model_name=model_name,
            output_dir=output_dir,
        )

        # Set hyperparameters
        result.set_hyperparameters(**hyperparameters)

        # Set training info
        result.set_training_info(
            training_time=training_time,
            num_train_samples=len(dataset.train_data) if dataset.train_data else 0,
            num_val_samples=len(dataset.val_data) if dataset.val_data else 0,
            num_test_samples=len(dataset.test_data) if dataset.test_data else 0,
        )

        if dataset.test_data is None or len(dataset.test_data) == 0:
            print("Warning: No test set, skipping evaluation")
            return result

        try:
            test_data = list(dataset.test_data)

            print(f"\nEvaluating test set (using unified seqeval evaluation method)...")

            all_true_entities = []
            all_pred_entities = []
            all_texts = []

            # Set model to eval mode
            self.model_wrapper.model.eval()

            for item in tqdm(test_data, desc="Evaluating"):
                text = item.get('text', '')
                true_entities = item.get('entities', [])

                # Generate prediction
                prompt = self.format_ner_prompt(text, entities=None, is_training=False)

                # Tokenize prompt
                inputs = self.model_wrapper.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.model_wrapper.model.device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=1,
                        do_sample=False,
                        pad_token_id=self.model_wrapper.tokenizer.pad_token_id,
                    )

                # Decode generated text
                generated_text = self.model_wrapper.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Parse entities from generated JSON
                pred_entities = self._parse_generated_entities(generated_text)

                all_true_entities.append(true_entities)
                all_pred_entities.append(pred_entities)
                all_texts.append(text)

            # Use unified seqeval evaluation (same as BERT models)
            metrics = self.model_wrapper.compute_seqeval_metrics(
                all_true_entities, all_pred_entities, all_texts
            )

            # Set overall metrics (compatible with HistoryDocumentResult)
            result.set_metrics(
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1=metrics.get('f1', 0.0),
                accuracy=metrics.get('accuracy', 0.0),
            )

            # Set per-type metrics if available
            if 'per_type_metrics' in metrics:
                result.set_per_type_metrics(metrics['per_type_metrics'])

            # Set classification report
            report_str = metrics.get('classification_report', '')
            result.set_classification_report(report_str)

            print("\n" + "=" * 80)
            print("LLaMA NER Evaluation Results (seqeval):")
            print("=" * 80)
            print(report_str)
            print("=" * 80)
            print(f"Samples evaluated: {len(test_data)}")

        except Exception as e:
            print(f"\nError: Exception during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            result.set_error(str(e))

        return result

    def _parse_generated_entities(self, generated_text: str) -> list:
        """Parse entities from generated JSON text"""
        import json
        import re

        try:
            # Try to find JSON array in generated text
            # Look for patterns like [{...}, {...}] or []
            json_match = re.search(r'\[.*?\]', generated_text, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                # Validate entity format
                valid_entities = []
                for e in entities:
                    if isinstance(e, dict) and 'type' in e and 'text' in e:
                        valid_entities.append({
                            'type': e['type'],
                            'text': e['text'],
                            'span': e.get('span', [0, len(e['text'])])
                        })
                return valid_entities
        except (json.JSONDecodeError, AttributeError):
            pass

        return []

    def _calculate_entity_metrics(self, all_true: list, all_pred: list) -> dict:
        """Calculate entity-level precision, recall, F1"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for true_entities, pred_entities in zip(all_true, all_pred):
            # Convert to comparable tuples (type, text)
            true_set = {(e.get('type', ''), e.get('text', '')) for e in true_entities}
            pred_set = {(e.get('type', ''), e.get('text', '')) for e in pred_entities}

            true_positives += len(true_set & pred_set)
            false_positives += len(pred_set - true_set)
            false_negatives += len(true_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
        }


class HistoryDocumentTrainerBiLSTM(HistoryDocumentTrainerBERT):
    """BiLSTM-specific trainer implementation for NER tasks

    Inherits from HistoryDocumentTrainerBERT since both use the same tokenization
    and label alignment logic. The only difference is the model architecture.
    """

    def __init__(self, model: HistoryDocumentModelBiLSTM):
        """
        Initialize BiLSTM trainer

        Args:
            model: HistoryDocumentModelBiLSTM instance
        """
        super().__init__(model)

    # All methods are inherited from HistoryDocumentTrainerBERT:
    # - convert_entities_to_bio_tags()
    # - tokenize_and_align_labels()
    # - create_compute_metrics()
    # - evaluate_and_create_result()
    # - train()
    # - mlm_pretrain() (not applicable for BiLSTM, but harmless)
    # - multi_dataset_train()


def main():
    parser = argparse.ArgumentParser(description="Train BERT model for NER with single or sequential multi-dataset support")

    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--datasets", type=str, required=True,
                       help="Comma-separated list of dataset paths (e.g., 'dataset1,dataset2')")
    parser.add_argument("--epochs", type=str, required=True,
                       help="Comma-separated list of epochs per dataset (e.g., '10,20')")

    # Training arguments
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Label arguments
    parser.add_argument("--entity-types", type=str,
                       default="役者,演目名,屋号,俳名,役名,興行関係者,狂言作者,書名,人名,音曲,事項",
                       help="Comma-separated entity types")

    # Model architecture arguments
    parser.add_argument("--use-crf", action="store_true",
                       help="Use CRF layer on top of BERT for better sequence labeling")

    # LoRA arguments (for LLaMA models)
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA for parameter-efficient fine-tuning (LLaMA models)")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha scaling parameter (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout rate (default: 0.05)")
    parser.add_argument("--use-4bit", action="store_true",
                       help="Use 4-bit quantization for LLaMA models")

    # MLM pretraining arguments
    parser.add_argument("--mlm-pretrain", action="store_true",
                       help="Perform MLM pretraining before NER training")
    parser.add_argument("--mlm-corpus", type=str,
                       help="Path to corpus file for MLM pretraining")
    parser.add_argument("--mlm-epochs", type=int, default=3,
                       help="Number of epochs for MLM pretraining (default: 3)")
    parser.add_argument("--mlm-corpus-size", type=int, default=None,
                       help="Number of corpus samples to randomly sample for MLM pretraining (default: use all)")
    parser.add_argument("--mlm-batch-size", type=int, default=None,
                       help="Batch size for MLM pretraining (default: use same as NER training)")
    parser.add_argument("--mlm-learning-rate", type=float, default=5e-5,
                       help="Learning rate for MLM pretraining (default: 5e-5)")
    parser.add_argument("--mlm-probability", type=float, default=0.15,
                       help="Probability of masking tokens in MLM (default: 0.15)")

    # Output options
    parser.add_argument("--no-save-model", action="store_true",
                       help="Do not save the trained model (only save results)")

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                       help="Stop training if eval F1 doesn't improve for N consecutive epochs (0 = disabled)")

    # Remote LLM arguments (for ChatGPT, Claude)
    parser.add_argument("--n-few-shot", type=int, default=5,
                       help="Number of few-shot examples for remote LLM (default: 5)")
    parser.add_argument("--few-shot-selection", type=str, default="stratified",
                       choices=["random", "diverse", "stratified"],
                       help="Few-shot selection strategy for remote LLM (default: stratified)")
    parser.add_argument("--remote-llm-provider", type=str, default=None,
                       choices=["chatgpt", "claude"],
                       help="Remote LLM provider (auto-detected from model name if not specified)")

    args = parser.parse_args()

    # Parse datasets and epochs
    dataset_paths = [d.strip() for d in args.datasets.split(',')]
    epochs_per_dataset = [int(e.strip()) for e in args.epochs.split(',')]

    if len(dataset_paths) != len(epochs_per_dataset):
        raise ValueError(f"Number of datasets ({len(dataset_paths)}) must match number of epoch specifications ({len(epochs_per_dataset)})")

    # Create label mappings
    entity_types = [e.strip() for e in args.entity_types.split(',')]
    labels = ["O"]
    for entity_type in entity_types:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # Determine model type
    model_name_lower = args.model.lower()
    is_bilstm = 'bilstm' in model_name_lower or args.model == 'bilstm'

    # Check for Remote LLM (ChatGPT/Claude API models)
    is_remote_llm = False
    remote_llm_provider = args.remote_llm_provider
    if remote_llm_provider is not None:
        is_remote_llm = True
    elif model_name_lower in ['chatgpt', 'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']:
        is_remote_llm = True
        remote_llm_provider = 'chatgpt'
    elif model_name_lower in ['claude', 'claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'] or 'claude' in model_name_lower:
        is_remote_llm = True
        remote_llm_provider = 'claude'

    # Check for local LLaMA models (but not ChatGPT which contains 'gpt')
    is_llama = not is_remote_llm and (
        'llama' in model_name_lower or
        'rinna' in model_name_lower or
        'elyza' in model_name_lower or
        ('gpt' in model_name_lower and 'neox' in model_name_lower)
    )

    # Determine model type string for display
    if is_bilstm:
        model_type_str = 'BiLSTM'
    elif is_remote_llm:
        model_type_str = f'Remote LLM ({remote_llm_provider})'
    elif is_llama:
        model_type_str = 'LLaMA/GPT'
    else:
        model_type_str = 'BERT-based'

    print(f"{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    # Adjust learning rate for BiLSTM (needs higher LR since training from scratch)
    learning_rate = args.learning_rate
    if is_bilstm and args.learning_rate == 2e-5:
        learning_rate = 1e-3  # BiLSTM needs higher learning rate (no pretrained weights)
        print(f"Note: Adjusted learning rate to {learning_rate} for BiLSTM (training from scratch)")

    print(f"Model Type: {model_type_str}")
    print(f"Use CRF: {args.use_crf}")
    print(f"Datasets: {dataset_paths}")
    print(f"Epochs per dataset: {epochs_per_dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Number of labels: {len(labels)}")
    print(f"{'='*60}\n")

    # Initialize model wrapper based on model type
    if is_bilstm:
        # BiLSTM model (benchmark)
        model_wrapper = HistoryDocumentModelBiLSTM(
            model_name=args.model,
            label2id=label2id,
            id2label=id2label,
            use_crf=args.use_crf,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
        )
        trainer = HistoryDocumentTrainerBiLSTM(model=model_wrapper)
    elif is_remote_llm:
        # Remote LLM models (ChatGPT, Claude API)
        # Load dataset first for Remote LLM (needed for trainer initialization)
        dataset = HistoryDocumentDataset(dataset_paths[0])

        if remote_llm_provider == 'chatgpt':
            model_wrapper = HistoryDocumentModelRemoteLLMChatGPT(
                model_name=args.model if args.model.lower() != 'chatgpt' else 'gpt-4o-mini',
                label2id=label2id,
                id2label=id2label,
                temperature=0.0,
            )
            trainer = HistoryDocumentTrainerRemoteLLMChatGPT(
                model=model_wrapper,
                dataset=dataset,
                n_few_shot=args.n_few_shot,
                few_shot_selection=args.few_shot_selection,
            )
        else:  # claude
            model_wrapper = HistoryDocumentModelRemoteLLMClaude(
                model_name=args.model if args.model.lower() != 'claude' else 'claude-3-haiku-20240307',
                label2id=label2id,
                id2label=id2label,
                temperature=0.0,
            )
            trainer = HistoryDocumentTrainerRemoteLLMClaude(
                model=model_wrapper,
                dataset=dataset,
                n_few_shot=args.n_few_shot,
                few_shot_selection=args.few_shot_selection,
            )
    elif is_llama:
        # LLaMA/GPT models (generative NER)
        # Default to LoRA if not explicitly specified
        use_lora = getattr(args, 'use_lora', True) or True
        model_wrapper = HistoryDocumentModelLLaMA(
            model_name=args.model,
            label2id=label2id,
            id2label=id2label,
            use_lora=use_lora,
            lora_r=getattr(args, 'lora_r', 16),
            lora_alpha=getattr(args, 'lora_alpha', 32),
            lora_dropout=getattr(args, 'lora_dropout', 0.05),
            use_4bit=getattr(args, 'use_4bit', False),
        )
        trainer = HistoryDocumentTrainerLLaMA(model=model_wrapper)
    else:
        # BERT-based models (default)
        model_wrapper = HistoryDocumentModelBERT(
            model_name=args.model,
            label2id=label2id,
            id2label=id2label,
            use_crf=args.use_crf
        )
        trainer = HistoryDocumentTrainerBERT(model=model_wrapper)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Special handling for Remote LLM (ChatGPT, Claude)
    if is_remote_llm:
        # Remote LLM uses different workflow: select few-shot examples then evaluate
        print("Running Remote LLM inference\n")

        # Record start time
        start_time = datetime.now()

        # "Train" = select few-shot examples (dataset already loaded during model init)
        trainer.train()

        # Evaluate
        result = trainer.evaluate(output_dir=str(output_dir))

        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds()

        # Update result with additional info
        result.hyperparameters = {
            "n_few_shot": args.n_few_shot,
            "few_shot_selection": args.few_shot_selection,
            "provider": remote_llm_provider,
        }
        result.training_time = inference_time

        # Save result
        result.save()

        # Print summary
        result.print_summary()

        print(f"\n✓ API calls: {model_wrapper.total_api_calls}")
        print(f"✓ Total tokens: {model_wrapper.total_tokens_used}")

        print(f"\nRemote LLM inference completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        return

    # MLM Pretraining (if enabled) - not applicable for Remote LLM or BiLSTM
    if args.mlm_pretrain:
        # Skip MLM pretraining for BiLSTM (not a pretrained language model)
        if is_bilstm:
            print(f"\n{'='*60}")
            print(f"Skipping MLM pretraining for BiLSTM model")
            print(f"(BiLSTM is not a pretrained language model)")
            print(f"{'='*60}\n")
        else:
            if not args.mlm_corpus:
                raise ValueError("--mlm-corpus is required when --mlm-pretrain is enabled")

            print(f"\n{'='*60}")
            print(f"MLM Pretraining Configuration")
            print(f"{'='*60}")
            print(f"Corpus file: {args.mlm_corpus}")
            print(f"Corpus size: {args.mlm_corpus_size if args.mlm_corpus_size else 'All'}")
            print(f"Epochs: {args.mlm_epochs}")
            print(f"Batch size: {args.mlm_batch_size if args.mlm_batch_size else args.batch_size}")
            print(f"Learning rate: {args.mlm_learning_rate}")
            print(f"MLM probability: {args.mlm_probability}")
            print(f"Seed: {args.seed}")
            print(f"{'='*60}\n")

            # Load corpus
            corpus = HistoryDocumentCorpus(
                corpus_file=args.mlm_corpus,
                max_length=512,
                corpus_size=args.mlm_corpus_size,
                seed=args.seed
            )

            # Create MLM output directory
            mlm_output_dir = output_dir / "mlm_pretrain"
            mlm_output_dir.mkdir(parents=True, exist_ok=True)

            # Perform MLM pretraining
            trainer.mlm_pretrain(
                corpus=corpus,
                output_dir=str(mlm_output_dir),
                num_epochs=args.mlm_epochs,
                batch_size=args.mlm_batch_size if args.mlm_batch_size else args.batch_size,
                learning_rate=args.mlm_learning_rate,
                mlm_probability=args.mlm_probability,
                seed=args.seed
            )

            # Reload MLM-pretrained encoder weights into the NER model.
            # mlm_pretrain() trains a separate AutoModelForMaskedLM instance and
            # writes it to <mlm_output_dir>/final_model, but does not modify
            # trainer.model_wrapper.model. Without this reload the subsequent NER
            # training would use the original pretrained weights and the MLM
            # adaptation would have no effect on downstream performance.
            mlm_final_model = mlm_output_dir / "final_model"
            if mlm_final_model.exists():
                print(f"\nReloading MLM-pretrained weights from {mlm_final_model}")
                trainer.model_wrapper.model_name = str(mlm_final_model)
                trainer.model_wrapper._load_model()
                trainer.model_wrapper._load_tokenizer()
                trainer.model_wrapper._create_data_collator()
            else:
                print(f"\nWarning: MLM final_model not found at {mlm_final_model}; "
                      f"NER training will use the original pretrained weights.")

            print(f"\n{'='*60}")
            print(f"MLM Pretraining completed!")
            print(f"Proceeding to NER training...")
            print(f"{'='*60}\n")

    if len(dataset_paths) == 1:
        # Single dataset training
        print("Running single-dataset training\n")

        # Load dataset
        dataset = HistoryDocumentDataset(dataset_paths[0])

        # Record start time
        start_time = datetime.now()

        # Train
        trainer.train(
            dataset=dataset,
            output_dir=str(output_dir),
            seed=args.seed,
            num_epochs=epochs_per_dataset[0],
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            save_model=not args.no_save_model,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()

        # Save model (unless --no-save-model is specified)
        if not args.no_save_model:
            final_model_dir = output_dir / "final_model"
            trainer.trainer.save_model(str(final_model_dir))
            # For CRF models, also save config and CRF weights
            if hasattr(trainer.model_wrapper.model, 'save_pretrained'):
                trainer.model_wrapper.model.save_pretrained(str(final_model_dir))
            print(f"\n✓ Model saved to: {final_model_dir}")
        else:
            print(f"\n✓ Skipping model save (--no-save-model)")

        # Create and save result using HistoryDocumentResult
        hyperparameters = {
            "num_epochs": epochs_per_dataset[0],
            "batch_size": args.batch_size,
            "learning_rate": learning_rate,
            "seed": args.seed,
            "use_crf": args.use_crf,
        }

        result = trainer.evaluate_and_create_result(
            dataset=dataset,
            output_dir=str(output_dir),
            model_name=args.model,
            hyperparameters=hyperparameters,
            training_time=training_time,
        )

        # Set best model path
        best_model_dir = output_dir / "best_model"
        if best_model_dir.exists():
            result.set_best_model_path(str(best_model_dir))

        # Save result
        result.save()

        # Print summary
        result.print_summary()
    else:
        # Sequential multi-dataset training
        print(f"Running sequential multi-dataset training ({len(dataset_paths)} stages)\n")
        trainer.train_sequential(
            dataset_paths=dataset_paths,
            epochs_per_dataset=epochs_per_dataset,
            output_base_dir=str(output_dir),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=learning_rate
        )

    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")


# =============================================================================
# Remote LLM Classes (ChatGPT, Claude API)
# =============================================================================

class HistoryDocumentModelRemoteLLM(HistoryDocumentModelLLM):
    """Base class for remote LLM API-based NER models

    This class extends HistoryDocumentModelLLM with remote API functionality.
    Subclasses should implement the _call_api method for specific providers.

    Inherits from HistoryDocumentModelLLM:
    - _build_ner_prompt(): Unified prompt construction
    - _parse_entity_response(): Entity parsing from JSON
    - _entities_to_bio_tags(): BIO tag conversion
    - _convert_item_format(): Format conversion
    """

    def __init__(self, model_name: str, label2id: dict, id2label: dict,
                 api_key: str = None, max_retries: int = 3, retry_delay: float = 1.0,
                 temperature: float = 0.0, **kwargs):
        """
        Initialize remote LLM model wrapper

        Args:
            model_name: Model name (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            api_key: API key (if None, will try to get from environment variable)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            temperature: Temperature for generation (0.0 for deterministic)
        """
        super().__init__(model_name, label2id, id2label, **kwargs)
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

        # Statistics
        self.total_tokens_used = 0
        self.total_api_calls = 0

    def _call_api(self, prompt: str) -> str:
        """Call the remote LLM API

        Args:
            prompt: Full prompt (system + user combined for simpler API)

        Returns:
            Response text from LLM

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement _call_api method")

    def predict_text(self, text: str, examples: List[dict] = None) -> Tuple[List[dict], List[str]]:
        """Predict entities for a text using span-based extraction

        Args:
            text: Input text
            examples: Optional few-shot examples

        Returns:
            Tuple of (entities list, BIO tags list)
        """
        import time

        prompt = self._build_ner_prompt(text, examples=examples, is_training=False)

        for attempt in range(self.max_retries):
            try:
                response = self._call_api(prompt)
                self.total_api_calls += 1
                # Parse response to get entities
                entities = self._parse_entity_response(response, text)
                # Convert to BIO tags for evaluation
                bio_tags = self._entities_to_bio_tags(text, entities)
                return entities, bio_tags
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print(f"All retries failed, returning empty results")
                    return [], ['O'] * len(text)

    def predict(self, data, examples: List[dict] = None) -> List[str]:
        """Predict NER tags - wrapper for compatibility

        Args:
            data: Either a list of tokens or a dict with 'text' key
            examples: Optional few-shot examples

        Returns:
            List of predicted BIO tags
        """
        if isinstance(data, dict):
            text = data.get('text', '')
        elif isinstance(data, list) and all(isinstance(t, str) for t in data):
            text = ''.join(data)
        else:
            text = str(data)

        _, bio_tags = self.predict_text(text, examples)
        return bio_tags

    def predict_batch(self, test_data, examples: List[dict] = None,
                      batch_size: int = 10, show_progress: bool = True) -> List[List[str]]:
        """Predict NER tags for a batch of sequences

        Args:
            test_data: Test dataset with 'text' or 'tokens' fields
            examples: Optional few-shot examples
            batch_size: Number of concurrent requests (for rate limiting)
            show_progress: Whether to show progress bar

        Returns:
            List of predicted tag sequences (BIO format)
        """
        import time
        from tqdm import tqdm

        predictions = []
        iterator = tqdm(test_data, desc="Predicting") if show_progress else test_data

        for i, item in enumerate(iterator):
            # Get text from item
            if isinstance(item, dict):
                if 'text' in item:
                    text = item['text']
                elif 'tokens' in item:
                    text = ''.join(item['tokens'])
                else:
                    text = str(item)
            else:
                text = str(item)

            _, bio_tags = self.predict_text(text, examples)
            predictions.append(bio_tags)

            # Rate limiting: small delay between requests
            if (i + 1) % batch_size == 0:
                time.sleep(0.5)

        return predictions

    def evaluate(self, dataset: 'HistoryDocumentDataset',
                 examples: List[dict] = None,
                 output_dir: str = None) -> 'HistoryDocumentResult':
        """Evaluate model on test dataset

        Args:
            dataset: Dataset with test split
            examples: Few-shot examples
            output_dir: Output directory for results

        Returns:
            HistoryDocumentResult object
        """
        import time

        start_time = time.time()

        result = HistoryDocumentResult(
            model_name=self.model_name,
            model_key=f"remote-llm-{self.model_name.split('/')[-1]}",
            output_dir=output_dir,
        )

        # Set training info
        result.set_training_info(
            training_time=0.0,  # No training time for remote LLM
            num_train_samples=len(dataset.train_data) if dataset.train_data else 0,
            num_val_samples=len(dataset.val_data) if dataset.val_data else 0,
            num_test_samples=len(dataset.test_data) if dataset.test_data else 0,
        )

        if dataset.test_data is None or len(dataset.test_data) == 0:
            print("Warning: No test data, skipping evaluation")
            return result

        try:
            # Get predictions - collect entities for unified evaluation
            print(f"\nEvaluating {self.model_name} on test set...")

            all_true_entities = []
            all_pred_entities = []
            all_texts = []

            from tqdm import tqdm
            for item in tqdm(list(dataset.test_data), desc="Predicting"):
                # Get text from item
                if isinstance(item, dict):
                    if 'text' in item:
                        text = item['text']
                    elif 'tokens' in item:
                        text = ''.join(item['tokens'])
                    else:
                        text = str(item)
                else:
                    text = str(item)

                # Get true entities
                true_entities = item.get('entities', []) if isinstance(item, dict) else []

                # Get predicted entities (via API call)
                pred_entities, _ = self.predict_text(text, examples)

                all_texts.append(text)
                all_true_entities.append(true_entities)
                all_pred_entities.append(pred_entities)

            # Use unified seqeval evaluation (same as BERT and LLaMA)
            metrics = self.compute_seqeval_metrics(
                all_true_entities, all_pred_entities, all_texts
            )

            # Set overall metrics
            result.set_metrics(
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1=metrics.get('f1', 0.0),
                accuracy=metrics.get('accuracy', 0.0),
            )

            # Set per-type metrics
            if 'per_type_metrics' in metrics:
                result.set_per_type_metrics(metrics['per_type_metrics'])

            # Set classification report
            report_str = metrics.get('classification_report', '')
            result.set_classification_report(report_str)

            # Set hyperparameters
            result.set_hyperparameters(
                model_name=self.model_name,
                temperature=self.temperature,
                n_few_shot=len(examples) if examples else 0,
                total_api_calls=self.total_api_calls,
            )

            inference_time = time.time() - start_time
            result.training_time = inference_time  # Use training_time field for inference time

            print("\n" + "=" * 80)
            print("Evaluation Results:")
            print("=" * 80)
            print(report_str)
            print("=" * 80)
            print(f"Total API calls: {self.total_api_calls}")
            print(f"Inference time: {inference_time:.2f}s")

        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            result.set_error(str(e))

        return result


class HistoryDocumentModelRemoteLLMChatGPT(HistoryDocumentModelRemoteLLM):
    """ChatGPT (OpenAI) API-based NER model"""

    def __init__(self, model_name: str = "gpt-4o", label2id: dict = None,
                 id2label: dict = None, api_key: str = None, **kwargs):
        """
        Initialize ChatGPT model wrapper

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        import os

        # Try to get API key from: 1) parameter, 2) environment variable, 3) config file
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = get_openai_api_key()
            except ValueError:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable, pass api_key parameter, or configure in src/config/config.py")

        super().__init__(model_name, label2id, id2label, api_key, **kwargs)

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

    def _call_api(self, prompt: str) -> str:
        """Call OpenAI API

        Args:
            prompt: Full prompt for NER task

        Returns:
            Response text from ChatGPT
        """
        # Use system message for task context and user message for prompt
        system_msg = "あなたは日本語の歴史文書（浮世絵・歌舞伎関連）の固有表現認識の専門家です。指示に従ってJSON形式で回答してください。"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=4096,
        )

        # Update token statistics
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens_used += response.usage.total_tokens

        return response.choices[0].message.content


class HistoryDocumentModelRemoteLLMClaude(HistoryDocumentModelRemoteLLM):
    """Claude (Anthropic) API-based NER model"""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514", label2id: dict = None,
                 id2label: dict = None, api_key: str = None, **kwargs):
        """
        Initialize Claude model wrapper

        Args:
            model_name: Anthropic model name (e.g., 'claude-sonnet-4-20250514', 'claude-3-haiku-20240307')
            label2id: Label to ID mapping
            id2label: ID to label mapping
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        """
        import os

        # Try to get API key from: 1) parameter, 2) environment variable, 3) config file
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                api_key = get_anthropic_api_key()
            except ValueError:
                raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable, pass api_key parameter, or configure in src/config/config.py")

        super().__init__(model_name, label2id, id2label, api_key, **kwargs)

        # Initialize Anthropic client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

    def _call_api(self, prompt: str) -> str:
        """Call Anthropic API

        Args:
            prompt: Full prompt for NER task

        Returns:
            Response text from Claude
        """
        # Use system message for task context
        system_msg = "あなたは日本語の歴史文書（浮世絵・歌舞伎関連）の固有表現認識の専門家です。指示に従ってJSON形式で回答してください。"

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_msg,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )

        # Update token statistics
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text


class HistoryDocumentTrainerRemoteLLM(HistoryDocumentTrainerLLM):
    """Trainer class for remote LLM-based NER (ChatGPT, Claude)

    Inherits from HistoryDocumentTrainerLLM:
    - _select_few_shot_examples(): Few-shot example selection
    - _select_diverse_examples(): Diverse selection strategy
    - _select_stratified_examples(): Stratified selection strategy
    - _print_few_shot_summary(): Summary printing

    The train() method prepares few-shot examples instead of actual training.

    Future extensions:
    - OpenAI fine-tuning API integration
    - Automatic prompt optimization
    - RAG-based example retrieval
    """

    def train(self, dataset: 'HistoryDocumentDataset' = None, output_dir: str = None,
              n_few_shot: int = None, **kwargs):
        """Prepare for inference by selecting few-shot examples

        For remote LLM, "training" means selecting representative few-shot examples.

        Args:
            dataset: Dataset to use (overrides constructor dataset)
            output_dir: Output directory (not used for remote LLM)
            n_few_shot: Number of few-shot examples (overrides constructor value)
            **kwargs: Additional arguments (ignored for remote LLM)
        """
        if dataset is not None:
            self.dataset = dataset

        if n_few_shot is not None:
            self.n_few_shot = n_few_shot

        print(f"\n{'='*60}")
        print(f"Preparing Remote LLM for NER: {self.model_wrapper.model_name}")
        print(f"{'='*60}")
        print(f"Few-shot examples: {self.n_few_shot}")
        print(f"Selection strategy: {self.few_shot_selection}")

        # Select few-shot examples (inherited from base class)
        self.few_shot_examples = self._select_few_shot_examples()

        print(f"✓ Selected {len(self.few_shot_examples)} few-shot examples")
        self._print_few_shot_summary()

        print(f"{'='*60}\n")

        return self

    def evaluate(self, dataset: 'HistoryDocumentDataset' = None,
                 output_dir: str = None) -> 'HistoryDocumentResult':
        """Evaluate model on test dataset

        Args:
            dataset: Dataset to evaluate (overrides constructor dataset)
            output_dir: Output directory for results

        Returns:
            HistoryDocumentResult object
        """
        if dataset is not None:
            self.dataset = dataset

        if self.dataset is None:
            raise ValueError("No dataset provided for evaluation")

        return self.model_wrapper.evaluate(
            self.dataset,
            examples=self.few_shot_examples,
            output_dir=output_dir
        )

    def tokenize_and_align_labels(self, example):
        """Not used for remote LLM - raises NotImplementedError"""
        raise NotImplementedError("Remote LLM does not use tokenization")


class HistoryDocumentTrainerRemoteLLMChatGPT(HistoryDocumentTrainerRemoteLLM):
    """ChatGPT-specific trainer

    Currently identical to base class, but provides extension point for:
    - OpenAI fine-tuning API integration
    - ChatGPT-specific prompt optimization
    """

    def __init__(self, model: HistoryDocumentModelRemoteLLMChatGPT,
                 dataset: 'HistoryDocumentDataset' = None,
                 n_few_shot: int = 5,
                 **kwargs):
        if not isinstance(model, HistoryDocumentModelRemoteLLMChatGPT):
            raise TypeError("model must be HistoryDocumentModelRemoteLLMChatGPT instance")
        super().__init__(model, dataset, n_few_shot, **kwargs)

    # Future: Add OpenAI fine-tuning support
    # def finetune(self, dataset, ...):
    #     """Fine-tune using OpenAI's fine-tuning API"""
    #     pass


class HistoryDocumentTrainerRemoteLLMClaude(HistoryDocumentTrainerRemoteLLM):
    """Claude-specific trainer

    Currently identical to base class, but provides extension point for:
    - Future Claude fine-tuning API (if released)
    - Claude-specific prompt optimization
    """

    def __init__(self, model: HistoryDocumentModelRemoteLLMClaude,
                 dataset: 'HistoryDocumentDataset' = None,
                 n_few_shot: int = 5,
                 **kwargs):
        if not isinstance(model, HistoryDocumentModelRemoteLLMClaude):
            raise TypeError("model must be HistoryDocumentModelRemoteLLMClaude instance")
        super().__init__(model, dataset, n_few_shot, **kwargs)


if __name__ == "__main__":
    main()