from src.pruning.vision_token_pruning import select_topk_vision_tokens
from src.quantization.sensitivity import fake_quantize_tensor, rank_layers
import torch

def test_fake_quant_roundtrip_shape():
    w = torch.randn(16, 16)
    q = fake_quantize_tensor(w, bits=8)
    assert q.shape == w.shape

def test_rank_layers():
    assert rank_layers({"a": 0.1, "b": 0.5, "c": 0.2}) == ["b", "c", "a"]

def test_pruning_keep_all():
    attn = torch.rand(1, 4, 20, 20)
    idx = select_topk_vision_tokens(attn, num_vision_tokens=10, keep_ratio=1.0)
    assert len(idx) == 10
