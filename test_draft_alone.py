import copy, torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.load_model import make_prompt
from src.speculative.eval_speculative import build_draft
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
draft = build_draft(model)
print(f"target layers {len(model.model.language_model.layers)} draft layers {len(draft.model.language_model.layers)}", flush=True)
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
for idx in [0,1,2,3,4]:
    ex = ds[idx]
    inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_t = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        out_d = draft.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"idx {idx} gold {ex['answer']} target {processor.decode(out_t[0], skip_special_tokens=True)[-80:]!r}", flush=True)
    print(f"idx {idx} draft  {processor.decode(out_d[0], skip_special_tokens=True)[-80:]!r}", flush=True)
print("DONE", flush=True)
