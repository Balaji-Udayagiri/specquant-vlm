"""Day 3 correction: finer layer sensitivity via logit margin.

The accuracy sweep (outputs/sensitivity.json) was a near tie: on a
50-example POPE slice, layers 0-30 all had drop 0.0 and layer 31 alone
had drop 0.04. Yes/no accuracy is too coarse to rank layers.

This script measures something continuous instead, on the same 50
examples and the same one-layer-at-a-time per-tensor INT8 fake
quantization:
  - For each example, run a single forward pass (no generation).
  - Take the logits at the last prompt position.
  - Margin = logit(gold yes/no token) - logit(other yes/no token).
  - For a quantized layer, record the mean absolute change in that
    margin across the 50 examples, plus the first-token accuracy
    implied by the margin sign.
Layers with a larger mean margin change are more sensitive to
quantization, even when no full-generation answer flips.

Token check, verified empirically on the pod: for a POPE example the
first generated token is 3869 ("Yes") or 1939 ("No") directly, with
no space token first, even though decoded text shows "ASSISTANT: Yes".
Do NOT use " Yes" / " No": those encode to two tokens
([29871, 3869] / [29871, 1939]) in this tokenizer. The script verifies
"Yes" and "No" each encode to a single token and prints their ids;
if they do not, it fails loudly rather than silently measuring the
wrong token.

Usage:
  python -m src.quantization.sensitivity_logits --config configs/baseline.yaml --limit 50
Writes outputs/sensitivity_logits.json (updated after every layer).
"""
import argparse, json, pathlib, torch
from src.common.config import load_config
from src.common.load_model import load_llava, make_prompt
from src.quantization.sensitivity import fake_quantize_tensor, layer_linears, language_layers

def token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f"{text!r} encodes to {ids}, not a single token"
    return ids[0]

def margins(model, processor, examples, yes_id, no_id):
    out = []
    with torch.no_grad():
        for ex in examples:
            image = ex["image"].convert("RGB")
            prompt = make_prompt(processor, ex["question"])
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            logits = model(**inputs).logits[0, -1].float()
            gold = ex["answer"].strip().lower()
            gold_id = yes_id if gold == "yes" else no_id
            other_id = no_id if gold == "yes" else yes_id
            margin = float(logits[gold_id] - logits[other_id])
            out.append(margin)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_adv = args.limit // 3 + (1 if args.limit % 3 else 0)
    n_pop = args.limit // 3 + (1 if args.limit % 3 == 2 else 0)
    n_rand = args.limit - n_adv - n_pop
    indices = (list(range(0, n_adv)) + list(range(3000, 3000 + n_pop))
               + list(range(6000, 6000 + n_rand)))

    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    examples = [ds[i] for i in indices]

    model, processor = load_llava(cfg["model"]["name"], cfg["model"].get("dtype", "float16"))
    tokenizer = processor.tokenizer
    yes_id = token_id(tokenizer, "Yes")
    no_id = token_id(tokenizer, "No")
    print(f"yes_id={yes_id} no_id={no_id}", flush=True)

    base = margins(model, processor, examples, yes_id, no_id)
    base_acc = sum(1 for m in base if m > 0) / len(base)
    print(f"baseline first-token accuracy={base_acc:.4f}", flush=True)

    out_path = pathlib.Path("outputs/sensitivity_logits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"baseline_first_token_accuracy": base_acc, "limit": len(examples),
               "indices": indices, "bits": args.bits, "yes_id": yes_id, "no_id": no_id,
               "layers": {}}
    layers = language_layers(model)
    for i, layer in enumerate(layers):
        linears = layer_linears(layer)
        backup = [m.weight.data.clone() for m in linears]
        with torch.no_grad():
            for m in linears:
                m.weight.data = fake_quantize_tensor(m.weight.data, bits=args.bits)
        ms = margins(model, processor, examples, yes_id, no_id)
        with torch.no_grad():
            for m, w in zip(linears, backup):
                m.weight.data = w
        changes = [abs(q - b) for q, b in zip(ms, base)]
        acc = sum(1 for m in ms if m > 0) / len(ms)
        results["layers"][f"language_model.layer.{i}"] = {
            "mean_abs_margin_change": round(sum(changes) / len(changes), 4),
            "max_abs_margin_change": round(max(changes), 4),
            "first_token_accuracy": round(acc, 4),
            "first_token_accuracy_drop": round(base_acc - acc, 4),
        }
        results["ranking"] = sorted(results["layers"],
                                    key=lambda k: results["layers"][k]["mean_abs_margin_change"],
                                    reverse=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"layer {i}: mean_abs_margin_change={results['layers'][f'language_model.layer.{i}']['mean_abs_margin_change']:.4f} first_token_acc={acc:.4f}", flush=True)
    print(f"Ranking (most sensitive first): {results['ranking']}", flush=True)
    print(f"Top 5: {results['ranking'][:5]}", flush=True)

if __name__ == "__main__":
    main()
