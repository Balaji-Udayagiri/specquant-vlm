import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import generate_pruned
# Monkeypatch to print first token: just call generate and also print gen ids
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
ex=ds[0]
inputs=processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
print("Q", ex["question"], flush=True)
for kr in [0.35]:
    gen = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"], keep_ratio=kr)
    print(f"keep {kr} gen_ids {gen}", flush=True)
    print(f"keep {kr} decoded tokens {[processor.tokenizer.decode([t]) for t in gen]}", flush=True)
    print(f"keep {kr} text {processor.tokenizer.decode(gen, skip_special_tokens=True)!r}", flush=True)
print("DONE", flush=True)
