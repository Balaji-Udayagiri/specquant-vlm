"""Day 6: FastV-style vision token pruning, evaluated on POPE.

Method, implemented as a manual language-model loop (not model.generate),
because HF's LlamaModel.forward computes position embeddings and the
causal mask once for the full sequence, before the layer loop, so a
mid-loop hook cannot prune correctly:

  - Inputs: processor input_ids / pixel_values, merged into
    inputs_embeds exactly as LlavaModel.forward does (text embeddings
    with image features scattered into placeholder positions).
  - Vision positions: input_ids == 32000. Confirmed Day 5: positions
    5-580 for the example checked, 576 tokens, not "first after BOS".
  - Layers 0-2 run normally. At layer 2, attention scores are computed
    manually from Q/K (after input_layernorm, with rotary embeddings):
    score(vision token) = mean attention it receives from text tokens,
    averaged over heads. This is the FastV score, and it does not
    require output_attentions / eager attention.
  - After layer 2, keep top-k vision tokens by that score (k =
    576 * keep_ratio), plus all non-vision tokens, in original order.
    Hidden states, position_ids, position embeddings (sliced, so
    original positions are preserved, not renumbered), and the causal
    mask (sliced on both query and key dims) are all pruned to the
    kept indices. Past KV for layers 0-2 is sliced the same way.
  - Layers 3-31 run on the pruned sequence. Generation then proceeds
    greedily for up to 16 tokens, with the pruned KV cache, using the
    same manual loop (no vision tokens in later steps, so no further
    pruning).
  - keep_ratio=1.0 should reproduce the unpruned baseline generation
    for the same example (a sanity check run before any ablation).

Usage:
  python -m src.pruning.eval_pruned --keep_ratio 0.35 --limit 200 --variant pruning_keep035
Writes outputs/<variant>/eval.json and appends to benchmarks/eval_results.csv
"""
import argparse, csv, json, math, pathlib, time, torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from src.common.config import load_config
from src.common.load_model import make_prompt
from src.common.metrics import timed
from src.evaluation.evaluate import parse_yes_no, stratified_indices

def get_inputs_embeds(llava_model, input_ids, pixel_values):
    text_embeds = llava_model.get_input_embeddings()(input_ids)
    image_outputs = llava_model.get_image_features(
        pixel_values=pixel_values, return_dict=True)
    image_features = image_outputs.pooler_output
    image_features = torch.cat(image_features, dim=0).to(
        text_embeds.device, text_embeds.dtype)
    mask = llava_model.get_placeholder_mask(
        input_ids, inputs_embeds=text_embeds, image_features=image_features)
    return text_embeds.masked_scatter(mask, image_features)

def layer2_scores(layer, hidden, position_embeddings, num_heads, head_dim):
    """Return full attn weights for layer 2, computed manually from Q/K."""
    ln = layer.input_layernorm(hidden)
    q = layer.self_attn.q_proj(ln).view(hidden.shape[0], hidden.shape[1], num_heads, head_dim).transpose(1, 2)
    k = layer.self_attn.k_proj(ln).view(hidden.shape[0], hidden.shape[1], num_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    return attn  # [batch, heads, seq, seq]

def slice_cache(cache, kept, num_layers):
    # DynamicCache in transformers 5 stores per-layer key/value in .layers
    # Fall back to .key_cache / .value_cache lists if present.
    matched = False
    try:
        for i in range(min(num_layers, len(cache.layers))):
            layer_cache = cache.layers[i]
            if hasattr(layer_cache, "keys"):
                layer_cache.keys = layer_cache.keys[:, :, kept, :]
                layer_cache.values = layer_cache.values[:, :, kept, :]
                matched = True
            elif hasattr(layer_cache, "key_cache"):
                layer_cache.key_cache = layer_cache.key_cache[:, :, kept, :]
                layer_cache.value_cache = layer_cache.value_cache[:, :, kept, :]
                matched = True
            else:
                print(f"cache layer {i} attrs: {[a for a in dir(layer_cache) if not a.startswith('_')][:30]}", flush=True)
    except Exception as e:
        print(f"cache slice warning (layers 0-{num_layers-1}): {e}", flush=True)
    if not matched:
        print("cache slice: no .layers match, trying .key_cache/.value_cache lists", flush=True)
    for attr in ["key_cache", "value_cache"]:
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            for i in range(min(num_layers, len(lst))):
                if lst[i] is not None:
                    lst[i] = lst[i][:, :, kept, :]

def generate_pruned(model, processor, input_ids, pixel_values, keep_ratio=0.35,
                    prune_after_layer=2, max_new_tokens=16):
    llava = model.model
    lang = llava.language_model
    device = model.device
    inputs_embeds = get_inputs_embeds(llava, input_ids, pixel_values)
    seq_len = inputs_embeds.shape[1]
    vision_pos = [i for i, t in enumerate(input_ids[0].tolist()) if t == 32000]
    text_pos = [i for i in range(seq_len) if i not in set(vision_pos)]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = create_causal_mask(
        config=lang.config, inputs_embeds=inputs_embeds,
        attention_mask=None, past_key_values=None, position_ids=position_ids)
    position_embeddings = lang.rotary_emb(inputs_embeds, position_ids=position_ids)
    past = DynamicCache(config=lang.config)
    hidden = inputs_embeds
    kept_indices = None
    original_last_pos = seq_len - 1
    num_heads = getattr(lang.config, "num_attention_heads", 32)
    head_dim = getattr(lang.config, "hidden_size", 4096) // num_heads
    eos_token_id = getattr(lang.config, "eos_token_id", 2)
    with torch.no_grad():
        for li, layer in enumerate(lang.layers):
            if li == prune_after_layer and keep_ratio < 1.0 and vision_pos:
                attn = layer2_scores(layer, hidden, position_embeddings, num_heads, head_dim)
                # text queries are positions after the last vision token
                text_q = [i for i in range(seq_len) if i > vision_pos[-1]]
                scores = attn[0, :, text_q, :][:, :, vision_pos].mean(dim=(0, 1))
                k = max(1, int(len(vision_pos) * keep_ratio))
                topk = torch.topk(scores, k=k).indices.tolist()
                kept_vision = sorted([vision_pos[i] for i in topk])
                # Kept = non-vision + kept vision, in original order
                kept_indices = sorted([i for i in range(seq_len) if i not in set(vision_pos)] + kept_vision)
            hidden = layer(hidden, attention_mask=causal_mask,
                           position_embeddings=position_embeddings,
                           position_ids=position_ids, past_key_values=past, use_cache=True)
            if kept_indices is not None and li == prune_after_layer:
                kept = torch.tensor(kept_indices, device=device)
                hidden = hidden[:, kept, :]
                position_ids = position_ids[:, kept]
                cos, sin = position_embeddings
                # cos/sin shape: [batch, seq, dim] in this version; slice seq dim if needed
                if cos.shape[1] == seq_len:
                    position_embeddings = (cos[:, kept, :], sin[:, kept, :])
                else:
                    position_embeddings = lang.rotary_emb(hidden, position_ids=position_ids)
                # Causal mask: for this SDPA model the original mask is
                # None (SDPA handles causality internally, in sequence
                # order, which is preserved by pruning). Recomputing it
                # with non-contiguous position_ids produced a wrong
                # mask and broke the first token (debug3: 'O)' instead
                # of 'Yes'), while keeping it None reproduced the
                # correct prefill (debug2). So only slice a real mask,
                # otherwise leave it as None.
                if causal_mask is not None and causal_mask.dim() == 4 and causal_mask.shape[2] == seq_len:
                    causal_mask = causal_mask[:, :, kept, :][:, :, :, kept]
                slice_cache(past, kept, prune_after_layer + 1)
                seq_len = len(kept_indices)
        hidden = lang.norm(hidden)
        logits = model.lm_head(hidden)
        next_token = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
        generated = [int(next_token[0, 0])]
        # Generation steps, no further pruning
        for step in range(max_new_tokens - 1):
            if int(next_token[0, 0]) == eos_token_id:
                break
            tok_emb = lang.embed_tokens(next_token)
            pos = torch.tensor([[original_last_pos + 1 + step]], device=device)
            cm = create_causal_mask(
                config=lang.config, inputs_embeds=tok_emb,
                attention_mask=None, past_key_values=past, position_ids=pos)
            pe = lang.rotary_emb(tok_emb, position_ids=pos)
            h = tok_emb
            for layer in lang.layers:
                h = layer(h, attention_mask=cm, position_embeddings=pe,
                          position_ids=pos, past_key_values=past, use_cache=True)
            h = lang.norm(h)
            lg = model.lm_head(h)
            next_token = torch.argmax(lg[0, -1]).unsqueeze(0).unsqueeze(0)
            generated.append(int(next_token[0, 0]))
    return generated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--keep_ratio", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--prune_after_layer", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_dataset("lmms-lab-encoder/POPE", split="test")
    indices = stratified_indices(args.limit)
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    correct = 0
    total_latency = 0.0
    total_tokens = 0
    per_category = {}
    out_examples = []
    t_start = time.time()
    for idx in indices:
        ex = ds[idx]
        prompt = make_prompt(processor, ex["question"])
        inputs = processor(text=prompt, images=ex["image"].convert("RGB"), return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen_ids = generate_pruned(model, processor, inputs["input_ids"], inputs["pixel_values"],
                                      keep_ratio=args.keep_ratio, prune_after_layer=args.prune_after_layer)
        secs = time.time() - t0
        text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # parse like Day 2: first yes/no word in generated text
        import re
        m = re.search(r"\b(yes|no)\b", text.lower())
        pred = m.group(1) if m else "other"
        gold = ex["answer"].strip().lower()
        ok = pred == gold
        correct += int(ok)
        total_latency += secs
        total_tokens += len(gen_ids)
        cat = ex["category"]
        c = per_category.setdefault(cat, {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(ok)
        out_examples.append({"index": int(idx), "category": cat, "question": ex["question"],
                             "gold": gold, "pred": pred, "correct": ok,
                             "latency_s": round(secs, 3), "tokens_generated": len(gen_ids),
                             "generated_text": text})
        if len(out_examples) % 50 == 0:
            print(f"{len(out_examples)}/{len(indices)} correct: {correct}/{len(out_examples)}", flush=True)
    n = len(out_examples)
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None
    summary = {"variant": args.variant, "keep_ratio": args.keep_ratio,
               "prune_after_layer": args.prune_after_layer, "model": cfg["model"]["name"],
               "dataset": "lmms-lab-encoder/POPE test, stratified", "limit": n,
               "accuracy": round(correct / n, 4), "correct": correct,
               "latency_s_per_image": round(total_latency / n, 3),
               "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency else None,
               "peak_vram_gb": peak_vram_gb,
               "wall_time_s": round(time.time() - t_start, 1),
               "per_category": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                                for k, v in per_category.items()},
               "examples": out_examples}
    out_dir = pathlib.Path("outputs") / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    csv_path = pathlib.Path("benchmarks/eval_results.csv")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "dataset", "limit", "accuracy", "latency_s_per_image",
            "tokens_per_sec", "peak_vram_gb", "per_category"])
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({"variant": args.variant, "dataset": summary["dataset"], "limit": n,
                         "accuracy": summary["accuracy"],
                         "latency_s_per_image": summary["latency_s_per_image"],
                         "tokens_per_sec": summary["tokens_per_sec"],
                         "peak_vram_gb": summary["peak_vram_gb"], "per_category": json.dumps(summary["per_category"])})
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))

if __name__ == "__main__":
    main()
