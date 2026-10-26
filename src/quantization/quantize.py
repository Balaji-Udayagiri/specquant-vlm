"""Day 3-4: Produce INT8 and INT4 variants.

Paths to try, in order:
  A. bitsandbytes 8-bit: fastest to get working, good baseline.
  B. AWQ 4-bit: better quality at 4-bit, needs calibration data.
  C. ONNX Runtime static quantization: use export_onnx.py.

Do not try all three on day 1. Get A measured, then B, then compare.
"""
import argparse
from transformers import BitsAndBytesConfig
from src.common.config import load_config
from src.common.load_model import load_llava

def bnb_config(bits: int):
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    raise ValueError("bits must be 8 or 4")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    q = cfg.get("quantization", {})
    if q.get("method") == "bitsandbytes":
        qc = bnb_config(int(q.get("bits", 8)))
        model, processor = load_llava(cfg["model"]["name"], quantization_config=qc)
        print(f"Loaded {cfg['model']['name']} with bitsandbytes {q.get('bits')} bit")
        print(model.get_memory_footprint())
    else:
        print(f"Method {q.get('method')} is a Day 4 task: wire AWQ / GPTQ here using sensitivity.json")

if __name__ == "__main__":
    main()
