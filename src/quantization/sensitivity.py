"""Day 3: Layer sensitivity analysis.

Idea, from adaptive quantization work: quantize one block at a time,
measure the accuracy hit, rank layers from most to least sensitive.
Keep the top 10 to 15 percent in FP16, quantize the rest.

Method used here:
  - Calibration / eval set: a stratified 50-example slice of POPE
    (17 adversarial, 17 popular, 16 random), the same indices every run.
  - Target: the 32 decoder layers of LLaVA's language model
    (model.model.language_model.layers), 7 linear weights each
    (q/k/v/o_proj, gate/up/down_proj). LayerNorms are not quantized.
  - Fake quantization: per-tensor symmetric INT8, computed in float32
    and cast back to the weight dtype. This is a ranking proxy, not
    bitsandbytes: bnb uses finer-grained scales, so absolute drops here
    will differ from Day 4, but the relative ranking is the signal.
  - For each layer: quantize only that layer in place, run the 50
    examples, record accuracy, restore the original weights, move on.
    The model is loaded once, so a full sweep is 33 eval runs, not
    33 model loads.

Usage:
  python -m src.quantization.sensitivity --config configs/baseline.yaml --limit 50
Writes outputs/sensitivity.json (updated after every layer, so an
interrupted sweep still keeps the layers it finished).
"""
import argparse, json, pathlib, torch
from src.common.config import load_config
from src.common.load_model import load_llava, make_prompt
from src.common.metrics import timed
from src.evaluation.evaluate import parse_yes_no

def fake_quantize_tensor(w, bits: int = 8):
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    scale = wf.abs().max() / qmax if wf.numel() else 1.0
    if scale == 0:
        return w
    q = (wf / scale).round().clamp(-qmax - 1, qmax) * scale
    return q.to(w.dtype)

def rank_layers(deltas: dict) -> list:
    """deltas: {layer_name: accuracy_drop}. Returns most sensitive first."""
    return sorted(deltas, key=lambda k: deltas[k], reverse=True)

def language_layers(model):
    return model.model.language_model.layers

def layer_linears(layer):
    mods = []
    for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        mod = layer
        for part in name.split("."):
            mod = getattr(mod, part)
        mods.append(mod)
    return mods

def eval_accuracy(model, processor, examples, max_new_tokens=16):
    correct = 0
    total_latency = 0.0
    for ex in examples:
        image = ex["image"].convert("RGB")
        prompt = make_prompt(processor, ex["question"])
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        out, secs = timed(lambda: model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False))
        total_latency += secs
        pred = parse_yes_no(processor.decode(out[0], skip_special_tokens=True))
        correct += int(pred == ex["answer"].strip().lower())
    return correct / len(examples) if examples else 0.0, total_latency / len(examples) if examples else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Stratified 50: 17 adversarial, 17 popular, 16 random.
    n_adv = args.limit // 3 + (1 if args.limit % 3 else 0)
    n_pop = args.limit // 3 + (1 if args.limit % 3 == 2 else 0)
    n_rand = args.limit - n_adv - n_pop
    indices = (list(range(0, n_adv)) + list(range(3000, 3000 + n_pop))
               + list(range(6000, 6000 + n_rand)))

    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    examples = [ds[i] for i in indices]

    model, processor = load_llava(cfg["model"]["name"], cfg["model"].get("dtype", "float16"))
    layers = language_layers(model)
    print(f"{len(layers)} language layers, {len(examples)} POPE examples", flush=True)

    base_acc, base_lat = eval_accuracy(model, processor, examples)
    print(f"baseline (no quant): accuracy={base_acc:.4f} latency={base_lat:.3f}s/image", flush=True)

    out_path = pathlib.Path("outputs/sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"baseline_accuracy": base_acc, "baseline_latency_s_per_image": base_lat,
               "limit": len(examples), "indices": indices, "bits": args.bits, "layers": {}}

    for i, layer in enumerate(layers):
        linears = layer_linears(layer)
        backup = [m.weight.data.clone() for m in linears]
        with torch.no_grad():
            for m in linears:
                m.weight.data = fake_quantize_tensor(m.weight.data, bits=args.bits)
        acc, lat = eval_accuracy(model, processor, examples)
        with torch.no_grad():
            for m, w in zip(linears, backup):
                m.weight.data = w
        drop = round(base_acc - acc, 4)
        results["layers"][f"language_model.layer.{i}"] = {
            "accuracy": round(acc, 4), "drop": drop, "latency_s_per_image": round(lat, 3)}
        results["ranking"] = rank_layers({k: v["drop"] for k, v in results["layers"].items()})
        out_path.write_text(json.dumps(results, indent=2))
        print(f"layer {i}: accuracy={acc:.4f} drop={drop:.4f}", flush=True)

    print(f"Ranking (most sensitive first): {results['ranking']}", flush=True)
    print(f"Top 5: {results['ranking'][:5]}", flush=True)

if __name__ == "__main__":
    main()
