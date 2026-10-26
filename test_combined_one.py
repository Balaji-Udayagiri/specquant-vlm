import time, torch
from datasets import load_dataset
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import generate_pruned
from src.quantization.eval_quantized import load_quantized, restore_fp16_layers
model, processor = load_quantized("llava-hf/llava-1.5-7b-hf", 4)
print("Restoring FP16 layers", flush=True)
model = restore_fp16_layers(model, "llava-hf/llava-1.5-7b-hf", [31,1,16,10,7])
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
ex = ds[0]
inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
print("Q", ex["question"], "gold", ex["answer"], flush=True)
with torch.no_grad():
    t0=time.time()
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"INT4 no pruning time {time.time()-t0:.2f}s text {processor.decode(out[0], skip_special_tokens=True)!r}", flush=True)
    for kr in [0.5, 0.35]:
        t0=time.time()
        try:
            gen = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"], keep_ratio=kr)
            print(f"INT4 pruning keep={kr} time {time.time()-t0:.2f}s text {processor.tokenizer.decode(gen, skip_special_tokens=True)!r}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"INT4 pruning keep={kr} FAILED {e}", flush=True)
print("DONE", flush=True)
