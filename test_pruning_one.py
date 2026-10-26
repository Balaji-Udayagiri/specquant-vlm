import time, torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import generate_pruned
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
ex = ds[0]
prompt = make_prompt(processor, ex["question"])
inputs = processor(text=prompt, images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
print("Q", ex["question"], "gold", ex["answer"], flush=True)
for kr in [1.0, 0.35, 0.25]:
    t0=time.time()
    try:
        gen = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"], keep_ratio=kr)
        print(f"keep={kr} time={time.time()-t0:.2f}s text={processor.tokenizer.decode(gen, skip_special_tokens=True)!r}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"keep={kr} FAILED: {e}", flush=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print("baseline text:", processor.decode(out[0], skip_special_tokens=True), flush=True)
print("DONE", flush=True)
