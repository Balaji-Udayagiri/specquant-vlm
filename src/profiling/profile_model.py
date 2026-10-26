"""Day 1-2: Baseline profiling.

Usage:
  python -m src.profiling.profile_model --config configs/baseline.yaml --image data/sample.jpg --question "What is in this image?"

Outputs a JSON line with TTFT proxy, total latency, tokens/sec, peak VRAM.
Replace the single generate call with a proper TTFT split once baseline works:
stream the first token separately, then time the rest.
"""
import argparse, json
from PIL import Image
from src.common.config import load_config
from src.common.load_model import load_llava, make_prompt
from src.common.metrics import timed, peak_vram_gb, reset_peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image in one sentence.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, processor = load_llava(cfg["model"]["name"], cfg["model"].get("dtype", "float16"))
    image = Image.open(args.image).convert("RGB")
    prompt = make_prompt(processor, args.question)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    reset_peak()
    out, secs = timed(lambda: model.generate(**inputs, max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 64), do_sample=False))
    n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    result = {
        "config": args.config,
        "tokens_generated": int(n_tokens),
        "latency_s": round(secs, 3),
        "tokens_per_sec": round(n_tokens / secs, 2) if secs else None,
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "text": processor.decode(out[0], skip_special_tokens=True),
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
