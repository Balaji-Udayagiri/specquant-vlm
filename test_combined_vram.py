import torch
from datasets import load_dataset
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import generate_pruned
from src.quantization.eval_quantized import load_quantized, restore_fp16_layers
from src.evaluation.evaluate import stratified_indices
model, processor = load_quantized("llava-hf/llava-1.5-7b-hf", 4)
model = restore_fp16_layers(model, "llava-hf/llava-1.5-7b-hf", [31,1,16,10,7])
print("footprint", round(model.get_memory_footprint()/(1024**3),3), flush=True)
torch.cuda.reset_peak_memory_stats()
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
indices = stratified_indices(200)
correct=0
for i, idx in enumerate(indices):
    ex = ds[idx]
    inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"], keep_ratio=0.35)
    import re
    text = processor.tokenizer.decode(gen, skip_special_tokens=True)
    m = re.search(r"\b(yes|no)\b", text.lower())
    pred = m.group(1) if m else "other"
    if pred == ex["answer"].strip().lower():
        correct+=1
    if i % 50 == 49:
        print(f"{i+1} correct {correct} peak {round(torch.cuda.max_memory_allocated()/(1024**3),3)}", flush=True)
print(f"FINAL correct {correct}/200 peak {round(torch.cuda.max_memory_allocated()/(1024**3),3)}", flush=True)
