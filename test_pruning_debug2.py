import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache
from src.common.load_model import make_prompt
from src.pruning.eval_pruned import get_inputs_embeds, layer2_scores, slice_cache
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
vision=[i for i,t in enumerate(ids[0].tolist()) if t==32000]
def run_prefill(keep_ratio):
    pos=torch.arange(seq, device=model.device).unsqueeze(0)
    cm=create_causal_mask(config=lang.config, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
    pe=lang.rotary_emb(emb, position_ids=pos)
    past=DynamicCache(config=lang.config)
    hidden=emb
    kept=None
    with torch.no_grad():
        for li,layer in enumerate(lang.layers):
            if li==2 and keep_ratio<1.0:
                attn=layer2_scores(layer, hidden, pe, 32, 128)
                text_q=[i for i in range(seq) if i>vision[-1]]
                scores=attn[0,:,text_q,:][:,:,vision].mean(dim=(0,1))
                k=int(len(vision)*keep_ratio)
                topk=torch.topk(scores,k=k).indices.tolist()
                kept_vision=sorted([vision[i] for i in topk])
                kept=sorted([i for i in range(seq) if i not in set(vision)]+kept_vision)
                print(f"keep {keep_ratio} kept len {len(kept)} vision kept {len(kept_vision)} first kept vision {kept_vision[:5]}", flush=True)
                # share of attention mass in kept
                s_sorted,_=torch.sort(scores, descending=True)
                print(f"  top{k} attention share {float(s_sorted[:k].sum()/scores.sum()):.3f}", flush=True)
            hidden=layer(hidden, attention_mask=cm, position_embeddings=pe, position_ids=pos, past_key_values=past, use_cache=True)
            if kept is not None and li==2:
                kept_t=torch.tensor(kept, device=model.device)
                hidden=hidden[:,kept_t,:]
                pos=pos[:,kept_t]
                cos,sin=pe
                pe=(cos[:,kept_t,:], sin[:,kept_t,:])
                print(f"  after prune pe shape {pe[0].shape} pos last {pos[0,-1].item()} pos first {pos[0,:6].tolist()}", flush=True)
                slice_cache(past, kept_t, 3)
                print(f"  cache layer0 keys shape after slice {tuple(past.layers[0].keys.shape)}", flush=True)
            if li in (2,3,5,10,31):
                print(f"  layer {li} last hidden mean {float(hidden[0,-1].mean()):.6f} norm {float(hidden[0,-1].norm()):.3f}", flush=True)
        hidden=lang.norm(hidden)
        logits=model.lm_head(hidden)
        top5=torch.topk(logits[0,-1], k=5)
        print(f"keep {keep_ratio} first-token top5: {[(int(i), processor.tokenizer.decode([int(i)]), float(v)) for i,v in zip(top5.indices, top5.values)]}", flush=True)
        return logits[0,-1]
for kr in [1.0, 0.99, 0.35]:
    print(f"\n=== keep {kr} ===", flush=True)
    run_prefill(kr)
print("DONE", flush=True)
