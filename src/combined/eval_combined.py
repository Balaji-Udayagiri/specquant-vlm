"""Day 10: Combined winners run.

Combined = INT4 (bnb NF4, Day 3 top-5 sensitive layers FP16) +
FastV pruning (manual loop, after layer 2) at keep 50% and 35%.
Speculative decoding is EXCLUDED from the combined run on purpose:
Day 8-9 measured it as 67-69% slower than baseline for all draft
token counts 3, 5, 7, with acceptance ~0.01-0.02, so including it
would worsen latency, not improve it. The "best draft" in this
project is therefore no draft, i.e. normal greedy generation.

INT4 loading and FP16 restore reuse src.quantization.eval_quantized
exactly as Day 4. Pruning reuses src.pruning.eval_pruned.generate_pruned
exactly as Day 6. No new quantization or pruning method is introduced
here, so the combined number is comparable to the individual rows.

Usage:
  python -m src.combined.eval_combined --limit 200
Writes outputs/combined_int4_pruning_keep050/eval.json and
outputs/combined_int4_pruning_keep035/eval.json, appends to
benchmarks/eval_results.csv
"""
import argparse, csv, json, pathlib, re, time, torch
from datasets import load_dataset
from src.common.config import load_config
from src.common.load_model import make_prompt
from src.evaluation.evaluate import stratified_indices
from src.pruning.eval_pruned import generate_pruned
from src.quantization.eval_quantized import load_quantized, restore_fp16_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    indices = stratified_indices(args.limit)
    for keep_ratio, variant in [(0.5, "combined_int4_pruning_keep050"),
                                (0.35, "combined_int4_pruning_keep035")]:
        model, processor = load_quantized(cfg["model"]["name"], 4)
        print(f"Restoring FP16 layers: [31, 1, 16, 10, 7]", flush=True)
        model = restore_fp16_layers(model, cfg["model"]["name"], [31, 1, 16, 10, 7])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        correct = 0
        total_latency = 0.0
        total_tokens = 0
        per_category = {}
        out_examples = []
        t_start = time.time()
        for idx in indices:
            ex = ds[idx]
            prompt = make_prompt(processor, ex["question"])
            inputs = processor(text=prompt, images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                gen_ids = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"],
                                          keep_ratio=keep_ratio, prune_after_layer=2)
            secs = time.time() - t0
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            m = re.search(r"\b(yes|no)\b", text.lower())
            pred = m.group(1) if m else "other"
            gold = ex["answer"].strip().lower()
            ok = pred == gold
            correct += int(ok)
            total_latency += secs
            total_tokens += len(gen_ids)
            cat = ex["category"]
            c = per_category.setdefault(cat, {"n": 0, "correct": 0})
            c["n"] += 1
            c["correct"] += int(ok)
            out_examples.append({"index": int(idx), "category": cat, "question": ex["question"],
                                 "gold": gold, "pred": pred, "correct": ok,
                                 "latency_s": round(secs, 3), "tokens_generated": len(gen_ids),
                                 "generated_text": text})
            if len(out_examples) % 50 == 0:
                print(f"{variant} {len(out_examples)}/{len(indices)} correct: {correct}/{len(out_examples)}", flush=True)
        n = len(out_examples)
        try:
            footprint = round(model.get_memory_footprint() / (1024 ** 3), 3)
        except Exception:
            footprint = None
        summary = {"variant": variant, "keep_ratio": keep_ratio, "bits": 4,
                   "keep_fp16_layers": [31, 1, 16, 10, 7], "speculative": "none (Day 8-9 draft was slower)",
                   "model": cfg["model"]["name"], "dataset": "lmms-lab-encoder/POPE test, stratified",
                   "limit": n, "accuracy": round(correct / n, 4), "correct": correct,
                   "latency_s_per_image": round(total_latency / n, 3),
                   "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency else None,
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
                   "memory_footprint_gb": footprint,
                   "wall_time_s": round(time.time() - t_start, 1),
                   "per_category": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)} for k, v in per_category.items()},
                   "examples": out_examples}
        out_dir = pathlib.Path("outputs") / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))
        csv_path = pathlib.Path("benchmarks/eval_results.csv")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "dataset", "limit", "accuracy", "latency_s_per_image",
                "tokens_per_sec", "peak_vram_gb", "per_category"])
            if csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({"variant": variant, "dataset": summary["dataset"], "limit": n,
                             "accuracy": summary["accuracy"],
                             "latency_s_per_image": summary["latency_s_per_image"],
                             "tokens_per_sec": summary["tokens_per_sec"],
                             "peak_vram_gb": summary["peak_vram_gb"],
                             "per_category": json.dumps(summary["per_category"])})
        print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2), flush=True)
        del model, processor
        torch.cuda.empty_cache()
    print("COMBINED_DONE", flush=True)

if __name__ == "__main__":
    main()
