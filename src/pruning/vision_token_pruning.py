"""Day 5-6: Vision token pruning, FastV style.

LLaVA-1.5 creates 576 vision tokens per image. After an early layer,
most get very little attention from text tokens.

Implementation plan:
  1. Run with output_attentions=True on 20 images. Confirm the pattern.
  2. Register a hook / patch the forward at prune_after_layer (default 2).
  3. At that layer, score vision tokens by attention received, keep top-k
     (keep_ratio in configs/pruning.yaml), drop the rest from hidden states
     and from the KV cache path.
  4. Ablate keep_ratio: 1.0, 0.5, 0.35, 0.25. Plot accuracy vs latency.

This scaffold gives you the scoring function, which is the core logic.
"""
import torch

def select_topk_vision_tokens(attn_weights: torch.Tensor, num_vision_tokens: int, keep_ratio: float):
    """
    attn_weights: [batch, heads, seq_len, seq_len] from the pruning layer.
    Returns indices of vision tokens to keep, sorted by position.
    """
    if keep_ratio >= 1.0:
        return torch.arange(num_vision_tokens)
    # Average attention that vision tokens receive from all later tokens, over heads.
    # Vision tokens are assumed to be the first num_vision_tokens positions after BOS.
    # Verify that offset for your exact processor before trusting the numbers.
    scores = attn_weights.mean(dim=1).mean(dim=1)  # [batch, seq_len] approx, refine in implementation
    vision_scores = scores[:, :num_vision_tokens]
    k = max(1, int(num_vision_tokens * keep_ratio))
    topk = torch.topk(vision_scores, k=k, dim=-1).indices[0]
    return topk.sort().values.cpu()

if __name__ == "__main__":
    print("TODO Day 5: verify vision token offset, then patch the model forward.")
