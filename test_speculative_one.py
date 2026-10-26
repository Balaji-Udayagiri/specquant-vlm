import copy, time, torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.common.load_model import make_prompt
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
# Draft: separate module objects, but with _modules dicts copied so target is NOT mutated.
# Layers are deepcopied (every other layer, 16 layers, renumbered 0-15) to fix cache layer_idx.
# Other components (vision, embeddings, norm, lm_head) are shared with target, no extra VRAM for them.
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
print(f"target layers {len(model.model.language_model.layers)} draft layers {len(draft.model.language_model.layers)}", flush=True)
print(f"target num_hidden {model.model.language_model.config.num_hidden_layers} draft num_hidden {draft.model.language_model.config.num_hidden_layers}", flush=True)
print(f"target layer0 idx {model.model.language_model.layers[0].self_attn.layer_idx} draft layer0 idx {draft.model.language_model.layers[0].self_attn.layer_idx}", flush=True)
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
ex = ds[0]
inputs = processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
print("Q", ex["question"], "gold", ex["answer"], flush=True)
with torch.no_grad():
    t0=time.time()
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"baseline time {time.time()-t0:.2f}s text {processor.decode(out[0], skip_special_tokens=True)!r}", flush=True)
    print(f"after baseline target layers {len(model.model.language_model.layers)}", flush=True)
    for nat in [5]:
        t0=time.time()
        try:
            out2 = model.generate(**inputs, max_new_tokens=16, do_sample=False, assistant_model=draft, num_assistant_tokens=nat)
            print(f"assisted nat={nat} time {time.time()-t0:.2f}s text {processor.decode(out2[0], skip_special_tokens=True)!r}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"assisted nat={nat} FAILED {e}", flush=True)
print("DONE", flush=True)
