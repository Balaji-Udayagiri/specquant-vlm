"""Day 5: Layer 2 attention analysis for FastV.

Loads LLaVA with attn_implementation="eager" (SDPA does not support
output_attentions in this transformers version), runs 20 POPE
examples, and for each one captures the language model layer 2
attention weights.

Confirmed offset (Oct 25 2026, Day 5 check):
  - input_ids length 597 for the example checked
  - BOS + "USER:" at positions 0-4
  - 576 vision tokens, id 32000, at positions 5-580
  - 16 text tokens after that, positions 581-596
The scaffold comment ("vision tokens are the first tokens after BOS")
is WRONG for this processor. Do not use it.

Score for a vision token: mean attention it receives from text
tokens, averaged over heads, at layer 2. This is the FastV score.
Writes outputs/pruning_attention.json with per-example stats and
the aggregate distribution, so Day 6 can use the same scoring.
"""
import argparse, json, pathlib, torch
from datasets import load_dataset
from src.common.config import load_config
from src.common.load_model import make_prompt
from transformers import AutoProcessor, LlavaForConditionalGeneration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # 20 examples: first 7 adversarial, 7 popular, 6 random, same POPE blocks as before
    indices = list(range(0, 7)) + list(range(3000, 3007)) + list(range(6000, 6006))
    indices = indices[:args.limit]
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    model.eval()
    results = []
    all_scores = []
    for idx in indices:
        ex = ds[idx]
        prompt = make_prompt(processor, ex["question"])
        inputs = processor(text=prompt, images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
        ids = inputs["input_ids"][0].tolist()
        vision_pos = [i for i, t in enumerate(ids) if t == 32000]
        text_pos = list(range(vision_pos[-1] + 1, len(ids))) if vision_pos else []
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        # attentions: tuple per layer? In transformers 5, output.attentions is for language model
        att = out.attentions
        print(f"example {idx}: attentions type={type(att)}, len={len(att) if att else 0}", flush=True)
        if not att:
            raise RuntimeError("No attentions returned; check attn_implementation=eager")
        # att[layer] shape: [batch, heads, seq, seq] or list?
        a2 = att[2]
        if isinstance(a2, (list, tuple)):
            a2 = a2[0]
        print(f"  layer2 attn shape: {tuple(a2.shape)}", flush=True)
        # a2: [batch, heads, q_len, k_len]
        scores = a2[0, :, text_pos, :][:, :, vision_pos].mean(dim=(0, 1)).float().cpu()
        all_scores.append(scores)
        s_sorted, _ = torch.sort(scores, descending=True)
        total = float(scores.sum())
        results.append({
            "index": int(idx), "category": ex["category"], "question": ex["question"],
            "seq_len": len(ids), "vision_start": vision_pos[0], "vision_end": vision_pos[-1],
            "num_vision": len(vision_pos), "num_text": len(text_pos),
            "score_mean": float(scores.mean()), "score_std": float(scores.std()),
            "score_min": float(scores.min()), "score_max": float(scores.max()),
            "top10_pct_share": float(s_sorted[:58].sum() / total) if total else 0.0,
            "top35_pct_share": float(s_sorted[:202].sum() / total) if total else 0.0,
            "bottom50_pct_share": float(s_sorted[288:].sum() / total) if total else 0.0,
        })
        print(f"  mean={results[-1]['score_mean']:.6f} top10% share={results[-1]['top10_pct_share']:.3f} top35% share={results[-1]['top35_pct_share']:.3f} bottom50% share={results[-1]['bottom50_pct_share']:.3f}", flush=True)
    agg = torch.cat(all_scores)
    out_path = pathlib.Path("outputs/pruning_attention.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"examples": results,
               "aggregate": {"mean": float(agg.mean()), "std": float(agg.std()),
                             "min": float(agg.min()), "max": float(agg.max()),
                             "mean_top10_pct_share": sum(r["top10_pct_share"] for r in results) / len(results),
                             "mean_top35_pct_share": sum(r["top35_pct_share"] for r in results) / len(results),
                             "mean_bottom50_pct_share": sum(r["bottom50_pct_share"] for r in results) / len(results)}}
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))

if __name__ == "__main__":
    main()
