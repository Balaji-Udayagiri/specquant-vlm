"""Day 4: Evaluate a bitsandbytes-quantized LLaVA on the same POPE 200.

Same indices, same prompts, same greedy decoding as Day 2, so the
accuracy drop is comparable:
  - stratified 200: 66 adversarial (0-65), 66 popular (3000-3065),
    68 random (6000-6067)
  - max_new_tokens 16, do_sample False, parse first yes/no word

Variants:
  --bits 8   : BitsAndBytes load_in_8bit
  --bits 4   : BitsAndBytes load_in_4bit, NF4, float16 compute
  --keep_fp16_layers "31,1,16,10,7" : after loading the quantized
    model, copy those language layers back from a fresh FP16 load.
    This is the mixed-precision step from Day 3: the most sensitive
    layers stay FP16, the rest stay quantized. It is NOT AWQ: AWQ
    (configs/quant_int4.yaml) is a separate quantizer and is not
    claimed here. If --keep_fp16_layers is set, the variant name
    records it.

Usage:
  python -m src.quantization.eval_quantized --bits 8 --variant quant_int8
  python -m src.quantization.eval_quantized --bits 4 --keep_fp16_layers 31,1,16,10,7 --variant quant_int4_sensitive_fp16
Writes outputs/<variant>/eval.json and appends to benchmarks/eval_results.csv
"""
import argparse, csv, json, pathlib, time
import torch
from transformers import BitsAndBytesConfig
from src.common.config import load_config
from src.common.load_model import load_llava, make_prompt
from src.common.metrics import timed, peak_vram_gb, reset_peak
from src.evaluation.evaluate import parse_yes_no, stratified_indices

def load_quantized(model_name, bits):
    if bits == 8:
        qc = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype="float16")
    else:
        raise ValueError("bits must be 8 or 4")
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, quantization_config=qc, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

def restore_fp16_layers(quant_model, model_name, layer_ids):
    """Copy FP16 weights for the given language layers into the quantized model."""
    fp_model, _ = load_llava(model_name, "float16")
    q_layers = quant_model.model.language_model.layers
    f_layers = fp_model.model.language_model.layers
    for lid in layer_ids:
        q_layer = q_layers[lid]
        f_layer = f_layers[lid]
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            q_mod, f_mod = q_layer, f_layer
            for part in name.split("."):
                q_mod, f_mod = getattr(q_mod, part), getattr(f_mod, part)
            # bnb linear modules store weight in a quantized form; replacing
            # the module with the FP16 module is the cleanest restore.
            parent_q, parent_f = q_layer, f_layer
            parts = name.split(".")
            for part in parts[:-1]:
                parent_q, parent_f = getattr(parent_q, part), getattr(parent_f, part)
            setattr(parent_q, parts[-1], f_mod)
    del fp_model
    torch.cuda.empty_cache()
    return quant_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--keep_fp16_layers", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    indices = stratified_indices(args.limit)
    examples = [ds[i] for i in indices]

    model, processor = load_quantized(model_name, args.bits)
    keep = [int(x) for x in args.keep_fp16_layers.split(",") if x.strip()] if args.keep_fp16_layers else []
    if keep:
        print(f"Restoring FP16 layers: {keep}", flush=True)
        model = restore_fp16_layers(model, model_name, keep)
    reset_peak()

    correct = 0
    total_latency = 0.0
    total_tokens = 0
    per_category = {}
    out_examples = []
    t_start = time.time()
    for idx, ex in zip(indices, examples):
        image = ex["image"].convert("RGB")
        prompt = make_prompt(processor, ex["question"])
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        out, secs = timed(lambda: model.generate(
            **inputs, max_new_tokens=16, do_sample=False))
        n_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
        text = processor.decode(out[0], skip_special_tokens=True)
        pred = parse_yes_no(text)
        gold = ex["answer"].strip().lower()
        ok = pred == gold
        correct += int(ok)
        total_latency += secs
        total_tokens += n_tokens
        cat = ex["category"]
        c = per_category.setdefault(cat, {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(ok)
        out_examples.append({"index": int(idx), "category": cat, "question": ex["question"],
                             "gold": gold, "pred": pred, "correct": ok,
                             "latency_s": round(secs, 3), "tokens_generated": n_tokens})
        if len(out_examples) % 50 == 0:
            print(f"{len(out_examples)}/{len(examples)} correct: {correct}/{len(out_examples)}", flush=True)

    n = len(out_examples)
    try:
        footprint_gb = round(model.get_memory_footprint() / (1024 ** 3), 3)
    except Exception:
        footprint_gb = None
    summary = {
        "variant": args.variant, "config": args.config, "model": model_name,
        "bits": args.bits, "keep_fp16_layers": keep,
        "dataset": "lmms-lab-encoder/POPE test, stratified", "limit": n,
        "accuracy": round(correct / n, 4), "correct": correct,
        "latency_s_per_image": round(total_latency / n, 3),
        "total_latency_s": round(total_latency, 3),
        "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency else None,
        "avg_tokens_generated": round(total_tokens / n, 2),
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "memory_footprint_gb": footprint_gb,
        "wall_time_s": round(time.time() - t_start, 1),
        "per_category": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                         for k, v in per_category.items()},
        "examples": out_examples,
    }
    out_dir = pathlib.Path("outputs") / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    csv_path = pathlib.Path("benchmarks/eval_results.csv")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "dataset", "limit", "accuracy", "latency_s_per_image",
            "tokens_per_sec", "peak_vram_gb", "per_category"])
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({"variant": args.variant, "dataset": summary["dataset"], "limit": n,
                         "accuracy": summary["accuracy"],
                         "latency_s_per_image": summary["latency_s_per_image"],
                         "tokens_per_sec": summary["tokens_per_sec"],
                         "peak_vram_gb": summary["peak_vram_gb"],
                         "per_category": json.dumps(summary["per_category"])})
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))

if __name__ == "__main__":
    main()
