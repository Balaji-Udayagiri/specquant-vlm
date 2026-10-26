"""Evaluation harness. Same prompts, same limit, same decoding for every variant.

POPE (Polling-based Object Probing Evaluation), via lmms-lab-encoder/POPE:
  - test split is 9000 examples: 3000 adversarial (0-2999),
    3000 popular (3000-5999), 3000 random (6000-8999).
  - Each example: a COCO image, a question "Is there a <object> in the image?",
    and a gold answer "yes" or "no".

For a limit of 200, this harness does NOT take the first 200 rows, which
would be adversarial-only. It takes a stratified slice so the number is
comparable across variants and closer to full POPE:
  - 67 adversarial (indices 0-66)
  - 67 popular (indices 3000-3066)
  - 66 random (indices 6000-6065)
For other limits it splits proportionally across the three blocks.

Usage:
  python -m src.evaluation.evaluate --config configs/baseline.yaml --limit 200
Writes outputs/<variant>/eval.json and benchmarks/eval_results.csv
"""
import argparse, csv, json, pathlib, re, time
from src.common.config import load_config
from src.common.load_model import load_llava, make_prompt
from src.common.metrics import timed, peak_vram_gb, reset_peak

RESULT_FIELDS = ["variant", "accuracy", "latency_s", "tokens_per_sec", "peak_vram_gb", "tokens_generated"]

def stratified_indices(limit):
    blocks = [(0, 3000), (3000, 3000), (6000, 3000)]
    per = [limit // 3, limit // 3, limit - 2 * (limit // 3)]
    indices = []
    for (start, _), n in zip(blocks, per):
        indices.extend(range(start, start + n))
    return indices

def parse_yes_no(text):
    # Assistant part only, then first yes/no word wins.
    assistant = text.split("ASSISTANT:")[-1].strip().lower()
    m = re.search(r"\b(yes|no)\b", assistant)
    return m.group(1) if m else "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    dtype = cfg["model"].get("dtype", "float16")

    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    indices = stratified_indices(args.limit)

    model, processor = load_llava(model_name, dtype)
    reset_peak()

    examples = []
    correct = 0
    total_latency = 0.0
    total_tokens = 0
    per_category = {}
    t_start = time.time()
    for idx in indices:
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        prompt = make_prompt(processor, ex["question"])
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        out, secs = timed(lambda: model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False))
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
        examples.append({
            "index": int(idx), "id": ex["id"], "question_id": ex["question_id"],
            "category": cat, "question": ex["question"], "gold": gold,
            "pred": pred, "correct": ok, "latency_s": round(secs, 3),
            "tokens_generated": n_tokens, "raw": text,
        })
        if len(examples) % 50 == 0:
            print(f"{len(examples)}/{len(indices)} correct so far: {correct}/{len(examples)}", flush=True)

    n = len(examples)
    summary = {
        "variant": args.variant,
        "config": args.config,
        "model": model_name,
        "dataset": "lmms-lab-encoder/POPE test, stratified",
        "limit": n,
        "indices": indices,
        "accuracy": round(correct / n, 4) if n else None,
        "correct": correct,
        "latency_s_per_image": round(total_latency / n, 3) if n else None,
        "total_latency_s": round(total_latency, 3),
        "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency else None,
        "avg_tokens_generated": round(total_tokens / n, 2) if n else None,
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "wall_time_s": round(time.time() - t_start, 1),
        "per_category": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                         for k, v in per_category.items()},
        "examples": examples,
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
        writer.writerow({
            "variant": args.variant, "dataset": summary["dataset"], "limit": n,
            "accuracy": summary["accuracy"],
            "latency_s_per_image": summary["latency_s_per_image"],
            "tokens_per_sec": summary["tokens_per_sec"],
            "peak_vram_gb": summary["peak_vram_gb"],
            "per_category": json.dumps(summary["per_category"]),
        })
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))

if __name__ == "__main__":
    main()
