import types
import torch
import transformers

from text_embedder import TextEmbeddingModel


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = texts
        maxlen = 4
        input_ids = torch.randint(0, 100, (len(batch), maxlen))
        attention_mask = torch.ones(len(batch), maxlen, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyModel:
    def __init__(self):
        # Simulate a transformer with 6 layers (0..5) plus pooler and final ln
        self.config = types.SimpleNamespace(is_encoder_decoder=False)
        self._named = []
        for i in range(6):
            name = f"encoder.layer.{i}.attn.weight"
            p = torch.nn.Parameter(torch.randn(2, 2))
            self._named.append((name, p))
        self._named.append(("pooler.dense.weight", torch.nn.Parameter(torch.randn(2, 2))))
        self._named.append(("ln_f.weight", torch.nn.Parameter(torch.randn(2, 2))))

    def named_parameters(self):
        return iter(self._named)

    def parameters(self):
        return (p for _, p in self._named)

    def to(self, device):
        return self

    def eval(self):
        return self
    
    def __call__(self, **kwargs):
        # Return an object with last_hidden_state: (batch, seq_len, hidden_dim)
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            batch = 1
            seq = 4
        else:
            batch, seq = input_ids.shape
        # Hidden dim = 2 to match parameter shapes above
        last_hidden = torch.randn(batch, seq, 2)
        return types.SimpleNamespace(last_hidden_state=last_hidden)


def test_unfreeze_top_k_layers(monkeypatch):
    # Monkeypatch HF loader functions to return our dummies
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: DummyModel())

    # Ensure fresh singleton
    TextEmbeddingModel._instance = None

    # Create model with finetune_top_k=2 (non-zero enables unfreezing)
    mod = TextEmbeddingModel.get("dummy-model", finetune_top_k=2)

    trainable = set(mod.list_trainable_parameters())

    # Expect top 2 layers (indices 4 and 5) to be unfrozen, plus pooler and ln_f
    expected = {"encoder.layer.4.attn.weight", "encoder.layer.5.attn.weight", "pooler.dense.weight", "ln_f.weight"}

    assert expected.issubset(trainable), f"Expected {expected} subset of trainable params, got {trainable}"
