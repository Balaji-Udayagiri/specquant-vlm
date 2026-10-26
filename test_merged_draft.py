import copy, time, torch, re
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.load_model import make_prompt
from src.evaluation.evaluate import stratified_indices
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
# Merged draft: average pairs of consecutive layers (0+1, 2+3, ..., 30+31) -> 16 layers
def build_merged_draft(model):
    draft = copy.copy(model); draft._modules = dict(model._modules)
    dm = copy.copy(model.model); dm._modules = dict(model.model._modules)
    dl = copy.copy(model.model.language_model); dl._modules = dict(model.model.language_model._modules)
    dl.config = copy.copy(model.model.language_model.config); dl.config.num_hidden_layers = 16
    layers=[]
    orig = model.model.language_model.layers
    for i in range(0, 32, 2):
        lc = copy.deepcopy(orig[i])
        other = orig[i+1]
        with torch.no_grad():
            for (n1,p1), (n2,p2) in zip(lc.named_parameters(), other.named_parameters()):
                p1.data.copy_((p1.data + p2.data)/2)
            for (n1,b1), (n2,b2) in zip(lc.named_buffers(), other.named_buffers()):
                if b1.dtype.is_floating_point:
                    b1.data.copy_((b1.data + b2.data)/2)
        lc.self_attn.layer_idx = i//2
        layers.append(lc)
    dl.layers = torch.nn.ModuleList(layers)
    dm.language_model = dl; draft.model = dm
    draft.config = copy.copy(model.config); draft.config.text_config = copy.copy(model.config.text_config); draft.config.text_config.num_hidden_layers = 16
    draft.eval(); return draft
draft = build_merged_draft(model)
print(f"target layers {len(model.model.language_model.layers)} merged draft layers {len(draft.model.language_model.layers)}", flush=True)
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
# Draft alone on 5 examples
for idx in [0,1,2,3,4]:
    ex = ds[idx]
    inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_d = draft.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"draft alone idx {idx} gold {ex['answer']} text {processor.decode(out_d[0], skip_special_tokens=True)[-100:]!r}", flush=True)
# Assisted on POPE 50 subset, nat 5, plus baseline for comparison in same run
indices = stratified_indices(50)
for variant, nat in [("baseline", None), ("merged_nat5", 5)]:
    correct=0; total_lat=0; total_tok=0
    for idx in indices:
        ex = ds[idx]
        inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
        t0=time.time()
        with torch.no_grad():
            if nat is None:
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False, assistant_model=draft, num_assistant_tokens=nat)
        secs=time.time()-t0
        gen = out[0][inputs["input_ids"].shape[1]:].tolist()
        text = processor.tokenizer.decode(gen, skip_special_tokens=True)
        m=re.search(r"\b(yes|no)\b", text.lower()); pred=m.group(1) if m else "other"
        correct+= int(pred==ex["answer"].strip().lower()); total_lat+=secs; total_tok+=len(gen)
    print(f"{variant} POPE50 accuracy {correct}/50={correct/50:.3f} latency {total_lat/50:.3f} tok/s {total_tok/total_lat:.2f}", flush=True)
print("DONE", flush=True)
