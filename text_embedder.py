import threading
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class TextEmbeddingModel:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Some HuggingFace models (e.g., T5 variants) require SentencePiece and slow tokenizer
        # Force use_fast=False to avoid conversion errors
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.is_encoder_decoder = bool(getattr(self.model.config, "is_encoder_decoder", False))

        # Cache embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            if self.is_encoder_decoder:
                encoder = self.model.get_encoder()
                out = encoder(**dummy)
                last_hidden = out.last_hidden_state
            else:
                out = self.model(**dummy)
                last_hidden = out.last_hidden_state
            self.embedding_dim = last_hidden.size(-1)

    @classmethod
    def get(cls, model_name: str, device: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = TextEmbeddingModel(model_name, device=device)
        return cls._instance

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        if self.is_encoder_decoder:
            encoder = self.model.get_encoder()
            outputs = encoder(**encoded)
            hidden = outputs.last_hidden_state
        else:
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state  # (B, L, H)
        # Mean-pool over tokens, masking padding
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom
        else:
            pooled = hidden.mean(dim=1)
        return pooled  # (B, H)


