import torch, math
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import get_inputs_embeds, layer2_scores
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()
llava=model.model; lang=llava.language_model
ds = load_dataset("lmms-lab-encoder/POPE", split="test")
ex=ds[0]
inputs=processor(text=make_prompt(processor, ex["question"]), images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
ids=inputs["input_ids"]; pix=inputs["pixel_values"]
emb=get_inputs_embeds(llava, ids, pix)
seq=emb.shape[1]
print("seq",seq, flush=True)
vision=[i for i,t in enumerate(ids[0].tolist()) if t==32000]
print("vision",vision[0],vision[-1],len(vision), flush=True)
pos=torch.arange(seq, device=model.device).unsqueeze(0)
cm=create_causal_mask(config=lang.config, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
print("causal_mask type",type(cm), "shape", tuple(cm.shape) if hasattr(cm,"shape") else None, "dim", cm.dim() if hasattr(cm,"dim") else None, flush=True)
if cm is not None:
    print("cm sample [0,0,0,:5]", cm[0,0,0,:5] if cm.dim()==4 else cm, flush=True)
    print("cm sample last row first 5, last 5:", cm[0,0,-1,:5], cm[0,0,-1,-5:], flush=True)
pe=lang.rotary_emb(emb, position_ids=pos)
print("pe shapes", pe[0].shape, pe[1].shape, flush=True)
# run layers 0-2 unpruned, capture hidden after layer2, and cache info
past=DynamicCache(config=lang.config)
hidden=emb
with torch.no_grad():
    for li,layer in enumerate(lang.layers[:3]):
        if li==2:
            attn=layer2_scores(layer, hidden, pe, 32, 128)
            text_q=[i for i in range(seq) if i>vision[-1]]
            scores=attn[0,:,text_q,:][:,:,vision].mean(dim=(0,1))
            print("scores mean",float(scores.mean()),"topk",torch.topk(scores,k=5).values.tolist(), flush=True)
        hidden=layer(hidden, attention_mask=cm, position_embeddings=pe, position_ids=pos, past_key_values=past, use_cache=True)
        print(f"after layer {li} hidden shape {tuple(hidden.shape)} hidden last mean {float(hidden[0,-1].mean()):.4f}", flush=True)
    print("cache layers len", len(past.layers) if hasattr(past,"layers") else "no layers attr", flush=True)
    if hasattr(past,"layers"):
        lc=past.layers[0]
        print("cache layer0 attrs", [a for a in dir(lc) if not a.startswith("_")], flush=True)
        for attr in ["keys","values","key_cache","value_cache"]:
            v=getattr(lc, attr, None)
            if v is not None:
                print(f"  {attr} shape {tuple(v.shape)}", flush=True)
    print("cache key_cache attr?", hasattr(past,"key_cache"), flush=True)
print("DONE", flush=True)
