"""Day 8-9: Speculative decoding eval on POPE, HF assisted generation.

Draft: LLaVA-1.5-7B with every other language layer deepcopied and
renumbered 0-15 (16 layers), other components (vision tower,
embeddings, norm, lm_head) shared with the target. Sharing layer
objects directly fails: draft layers keep their original layer_idx
(0,2,4,...,30) and the 16-entry draft KV cache indexes out of range,
and shallow-copying a nn.Module shares its _modules dict, so
assigning draft layers also truncates the target to 16 layers
(both bugs found in single-example tests, Oct 26 2026).

Target verification means assisted greedy output should match
baseline greedy output, except for rare floating-point flips from
parallel verification. Accuracy is therefore reported, not assumed.

Acceptance is estimated, not directly exposed by HF generate:
  target forwards counted by wrapping the target language model
  forward (draft has a separate language model object, so it is not
  counted). V = target forwards - 1 prefill.
  tokens per verification = generated tokens / V
  acceptance approx = (tokens per verification - 1) / num_assistant_tokens
Last round truncation makes this a slight underestimate.

Usage:
  python -m src.speculative.eval_speculative --limit 200
Writes outputs/speculative_nat{3,5,7}/eval.json and a baseline
outputs/speculative_baseline/eval.json in the same run, and appends
to benchmarks/eval_results.csv
"""
import argparse, copy, csv, json, pathlib, re, time, torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.config import load_config
from src.common.load_model import make_prompt
from src.evaluation.evaluate import stratified_indices

def build_draft(model):
    draft = copy.copy(model)
    draft._modules = dict(model._modules)
    draft_model = copy.copy(model.model)
    draft_model._modules = dict(model.model._modules)
    draft_lang = copy.copy(model.model.language_model)
    draft_lang._modules = dict(model.model.language_model._modules)
    draft_lang.config = copy.copy(model.model.language_model.config)
    draft_lang.config.num_hidden_layers = 16
    layers_copy = []
    for i, layer in enumerate(model.model.language_model.layers[::2]):
        lc = copy.deepcopy(layer)
        lc.self_attn.layer_idx = i
        layers_copy.append(lc)
    draft_lang.layers = torch.nn.ModuleList(layers_copy)
    draft_model.language_model = draft_lang
    draft.model = draft_model
    draft.config = copy.copy(model.config)
    draft.config.text_config = copy.copy(model.config.text_config)
    draft.config.text_config.num_hidden_layers = 16
    draft.eval()
    return draft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    indices = stratified_indices(args.limit)
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    model.eval()
    draft = build_draft(model)
    assert len(model.model.language_model.layers) == 32, "target mutated by draft build"
    assert len(draft.model.language_model.layers) == 16
    # target forward counter
    lang = model.model.language_model
    orig_forward = lang.forward
    counter = [0]
    def counting_forward(*a, **kw):
        counter[0] += 1
        return orig_forward(*a, **kw)
    lang.forward = counting_forward

    variants = [("speculative_baseline", None)] + [
        (f"speculative_nat{nat}", nat) for nat in [3, 5, 7]]
    for variant, nat in variants:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        correct = 0
        total_latency = 0.0
        total_tokens = 0
        total_target_forwards = 0
        per_category = {}
        out_examples = []
        t_start = time.time()
        for idx in indices:
            ex = ds[idx]
            prompt = make_prompt(processor, ex["question"])
            inputs = processor(text=prompt, images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
            counter[0] = 0
            t0 = time.time()
            with torch.no_grad():
                if nat is None:
                    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                else:
                    out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                         assistant_model=draft, num_assistant_tokens=nat)
            secs = time.time() - t0
            # generate returns full sequence; generated part is after input length
            gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            m = re.search(r"\b(yes|no)\b", text.lower())
            pred = m.group(1) if m else "other"
            gold = ex["answer"].strip().lower()
            ok = pred == gold
            correct += int(ok)
            total_latency += secs
            total_tokens += len(gen_ids)
            total_target_forwards += counter[0]
            cat = ex["category"]
            c = per_category.setdefault(cat, {"n": 0, "correct": 0})
            c["n"] += 1
            c["correct"] += int(ok)
            out_examples.append({"index": int(idx), "category": cat, "question": ex["question"],
                                 "gold": gold, "pred": pred, "correct": ok,
                                 "latency_s": round(secs, 3), "tokens_generated": len(gen_ids),
                                 "target_forwards": counter[0], "generated_text": text})
            if len(out_examples) % 50 == 0:
                print(f"{variant} {len(out_examples)}/{len(indices)} correct: {correct}/{len(out_examples)}", flush=True)
        n = len(out_examples)
        v = max(1, total_target_forwards - n)  # verification forwards, minus 1 prefill per example
        tokens_per_verif = total_tokens / v if v else None
        acceptance = None
        if nat is not None and tokens_per_verif is not None:
            acceptance = round(max(0.0, min(1.0, (tokens_per_verif - 1) / nat)), 3)
        summary = {"variant": variant, "num_assistant_tokens": nat,
                   "model": cfg["model"]["name"], "dataset": "lmms-lab-encoder/POPE test, stratified",
                   "limit": n, "accuracy": round(correct / n, 4), "correct": correct,
                   "latency_s_per_image": round(total_latency / n, 3),
                   "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency else None,
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
                   "target_forwards_per_image": round(total_target_forwards / n, 2),
                   "tokens_per_verification": round(tokens_per_verif, 3) if tokens_per_verif else None,
                   "acceptance_rate_approx": acceptance,
                   "wall_time_s": round(time.time() - t_start, 1),
                   "per_category": {k: {"n": vv["n"], "accuracy": round(vv["correct"] / vv["n"], 4)} for k, vv in per_category.items()},
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
        print(json.dumps({k: v2 for k, v2 in summary.items() if k != "examples"}, indent=2), flush=True)
    print("SPECULATIVE_DONE", flush=True)

if __name__ == "__main__":
    main()
