"""Day 7-9: Speculative decoding loop.

Correctness rule: speculative decoding must produce the SAME distribution
as target-only decoding when you use rejection sampling. Greedy version
is simpler to start with: accept draft tokens that match the target
argmax, resample the first mismatch from the target.

Pseudocode:
  while tokens left:
    draft proposes gamma tokens autoregressively (cheap)
    target scores all gamma positions in ONE forward pass
    accept the longest matching prefix, add one bonus token from target
    append, update KV caches for both models

Start with transformers assisted generation to get a working number:
  model.generate(..., assistant_model=draft_model)
Then implement the manual loop only if you want the deeper learning.
"""
import argparse
from src.common.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/speculative.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("Target:", cfg["target_model"]["name"])
    print("Draft:", cfg["draft_model"]["name"])
    print("TODO: load both models, run assisted generation, log acceptance rate.")

if __name__ == "__main__":
    main()
