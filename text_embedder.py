import threading
from typing import List, Optional
import re

import torch
from transformers import AutoModel, AutoTokenizer


class TextEmbeddingModel:
    """Singleton wrapper around a HuggingFace text model used for producing pooled text embeddings.

    New optional args:
    - finetune_top_k: int = 0
        If >0, unfreeze the top K transformer layers (see _unfreeze_top_k_layers) so they can be fine-tuned.

    Note: the singleton is created on first call to get(...). Subsequent calls return the same instance and
    will ignore different finetune options. If you need different configurations, construct instances directly.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: Optional[str] = None, finetune_top_k: int = 0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Some HuggingFace models (e.g., T5 variants) require SentencePiece and slow tokenizer
        # Force use_fast=False to avoid conversion errors
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        # By default, set eval. Training mode can be enabled by caller if they plan to fine-tune.
        self.model.eval()
        self.is_encoder_decoder = bool(getattr(self.model.config, "is_encoder_decoder", False))

        # Optionally configure finetuning of the top K transformer layers.
        self.finetune_top_k = int(finetune_top_k or 0)
        if self.finetune_top_k > 0:
            # Freeze all params first
            for p in self.model.parameters():
                p.requires_grad = False
            # Unfreeze top K layers on encoder (or model) as appropriate
            try:
                target = self.model.get_encoder() if self.is_encoder_decoder else self.model
            except Exception:
                target = self.model
            self._unfreeze_top_k_layers(target, self.finetune_top_k)

        # Cache embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            if self.is_encoder_decoder:
                encoder = self.model.get_encoder()
                out = encoder(**dummy)
                last_hidden = getattr(out, "last_hidden_state", None)
                if last_hidden is None:
                    last_hidden = out[0]
            else:
                out = self.model(**dummy)
                last_hidden = getattr(out, "last_hidden_state", None)
                if last_hidden is None:
                    last_hidden = out[0]
            self.embedding_dim = last_hidden.size(-1)

    @classmethod
    def get(cls, model_name: str, device: Optional[str] = None, finetune_top_k: int = 0):
        with cls._lock:
            if cls._instance is None:
                cls._instance = TextEmbeddingModel(model_name, device=device, finetune_top_k=finetune_top_k)
        return cls._instance

    def _unfreeze_top_k_layers(self, model: torch.nn.Module, top_k: int):
        """Unfreeze the top-k transformer layers in `model`.

        This function searches parameter names for common transformer layer patterns and
        enables requires_grad for parameters that belong to the top-k layers (highest indices).

        It also attempts to unfreeze final layer norms and poolers that commonly appear at the end.
        """
        # Collect candidate parameter names with layer indices
        param_index_map = {}  # idx -> [param_name, ...]
        pattern = re.compile(r"(?:\.layer\.|\.h\.|\.blocks\.)?(\d+)(?:\.|$)")
        for name, _ in model.named_parameters():
            m = pattern.search(name)
            if m:
                idx = int(m.group(1))
                param_index_map.setdefault(idx, []).append(name)

        if not param_index_map:
            # Fallback: try to match names like 'encoder.layer_._N' or 'transformer.layers.N'
            alt_pattern = re.compile(r"(\d+)")
            for name, _ in model.named_parameters():
                m = alt_pattern.search(name)
                if m:
                    idx = int(m.group(1))
                    param_index_map.setdefault(idx, []).append(name)

        if not param_index_map:
            # Nothing matched; as a last resort, just unfreeze last `top_k` parameters
            all_params = list(model.named_parameters())
            for name, p in all_params[-top_k:]:
                p.requires_grad = True
            return

        # Choose top_k highest indices
        indices = sorted(param_index_map.keys())
        selected = indices[-top_k:]

        # Unfreeze params for the selected layer indices
        names_to_unfreeze = set()
        for idx in selected:
            names_to_unfreeze.update(param_index_map.get(idx, []))

        # Also unfreeze final layernorms / poolers if present
        extra_patterns = [re.compile(r"layernorm", re.IGNORECASE), re.compile(r"layer_norm", re.IGNORECASE), re.compile(r"pooler", re.IGNORECASE), re.compile(r"ln_", re.IGNORECASE), re.compile(r"final", re.IGNORECASE)]
        for name, p in model.named_parameters():
            for pat in extra_patterns:
                if pat.search(name):
                    names_to_unfreeze.add(name)
                    break

        # Apply requires_grad True to selected params
        for name, p in model.named_parameters():
            if name in names_to_unfreeze:
                p.requires_grad = True

    def encode(self, texts: List[str], with_grad: bool = False) -> torch.Tensor:
        """Encode a list of texts into pooled embeddings.

        Args:
            texts: list of str
            with_grad: if True, compute with gradients enabled (no torch.no_grad()).
                Defaults to False to preserve previous inference-only behavior.
        Returns:
            Tensor of shape (B, H)
        """
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            # Ensure tokenized inputs are moved to the same device as the model.
            # The TextEmbeddingModel instance may have been constructed on CPU but
            # the underlying HF model can be moved later (e.g. by Lightning). Use
            # the actual device of the model parameters when available.
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                # Model has no parameters (unlikely) - fall back to configured device
                model_device = torch.device(self.device)
            encoded = {k: v.to(model_device) for k, v in encoded.items()}
            if self.is_encoder_decoder:
                encoder = self.model.get_encoder()
                outputs = encoder(**encoded)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs[0]
            else:
                outputs = self.model(**encoded)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs[0]
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

    def list_trainable_parameters(self) -> List[str]:
        """Return list of parameter names that currently have requires_grad=True."""
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def save_pretrained(self, save_dir: str, save_meta: bool = True):
        """Save model and tokenizer to `save_dir` using HuggingFace save_pretrained.

        Also writes a small metadata JSON containing finetune_top_k if save_meta is True.
        """
        import json
        import os

        os.makedirs(save_dir, exist_ok=True)
        # HuggingFace model & tokenizer saving
        try:
            self.model.save_pretrained(save_dir)
        except Exception:
            # Some models may not expose save_pretrained; fall back to torch.save
            torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        try:
            self.tokenizer.save_pretrained(save_dir)
        except Exception:
            pass

        if save_meta:
            meta = {"finetune_top_k": int(self.finetune_top_k)}
            with open(os.path.join(save_dir, "finetune_meta.json"), "w") as f:
                json.dump(meta, f)

    @classmethod
    def load_pretrained(cls, save_dir: str, device: Optional[str] = None, finetune_top_k: int = 0, set_singleton: bool = False):
        """Load a pretrained model/tokenizer from `save_dir` and return a TextEmbeddingModel instance.

        If `set_singleton` is True, the created instance will be stored as the module-level singleton.
        """
        # The constructor accepts a local path for AutoTokenizer/AutoModel.from_pretrained
        inst = TextEmbeddingModel(save_dir, device=device, finetune_top_k=finetune_top_k)
        if set_singleton:
            with cls._lock:
                cls._instance = inst
        return inst


if __name__ == "__main__":
    # Quick local check: instantiate with finetune_top_k to see which params are unfrozen.
    mod = TextEmbeddingModel.get("GT4SD/multitask-text-and-chemistry-t5-base-standard", finetune_top_k=3)
    print("Embedding dim:", mod.embedding_dim)
    trainable, all_params = mod.list_trainable_parameters()
    print(f"Trainable parameters ({len(trainable)} / {len(all_params)}):")
    for n in trainable[:20]:
        print(" ", n)


