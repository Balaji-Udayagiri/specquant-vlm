# Results (copy your real numbers here)

| Variant | Accuracy (POPE) | Latency / image | Tokens/sec | Peak VRAM | Notes |
|---|---:|---:|---:|---:|---|
| FP16 baseline | 83.5% (167/200) | 0.412 s / image (POPE eval); 1.084 s (single image profile) | 38.24 (POPE eval) | 13.553 GB | LLaVA-1.5-7B, RTX 4090, stratified POPE 200, Oct 25 2026 |
| INT8 (bnb) | 81.0% (162/200, -2.5 vs FP16) | 1.519 s / image | 10.29 | 7.262 GB (footprint 6.825 GB) | LLaVA-1.5-7B, RTX 4090, stratified POPE 200, Oct 25 2026 |
| INT4 (bnb NF4, sensitive layers FP16) | 82.5% (165/200, -1.0 vs FP16) | 0.543 s / image | 28.89 | 6.426 GB (footprint 5.072 GB) | LLaVA-1.5-7B, RTX 4090, stratified POPE 200, FP16 layers 31,1,16,10,7, Oct 25 2026, NOT AWQ |
| FP16 + pruning, manual loop keep 100% (sanity) | 83.5% (167/200) | 0.406 s / image | 38.84 | 13.521 GB | Manual loop reproduces Day 2 baseline exactly, Oct 26 2026, stratified POPE 200 |
| FP16 + pruning keep 50% | 81.5% (163/200, -2.0 vs FP16) | 0.386 s / image | 40.85 | 13.378 GB | FastV layer 2 top-k, after layer 2, Oct 26 2026, stratified POPE 200; adv 77.27%, pop 84.85%, rand 82.35% |
| FP16 + pruning keep 35% | 80.0% (160/200, -3.5 vs FP16) | 0.381 s / image | 41.45 | 13.331 GB | FastV layer 2 top-k, after layer 2, Oct 26 2026, stratified POPE 200; adv 75.76%, pop 84.85%, rand 79.41% |
| FP16 + pruning keep 25% | 79.5% (159/200, -4.0 vs FP16) | 0.384 s / image | 40.99 | 13.315 GB | FastV layer 2 top-k, after layer 2, Oct 26 2026, stratified POPE 200; adv 75.76%, pop 84.85%, rand 77.94% |
| FP16 + speculative, draft 16 layers (every other), nat 3 | 83.5% (167/200) | 0.689 s / image | 22.87 | 19.731 GB | HF assisted, Oct 26 2026, accuracy preserved exactly, but SLOWER than baseline; tokens per verification ~1.07, acceptance approx 0.023 |
| FP16 + speculative, draft 16 layers, nat 5 | 83.5% (167/200) | 0.683 s / image | 23.08 | 19.731 GB | HF assisted, Oct 26 2026, accuracy preserved exactly, but SLOWER; tokens per verification ~1.07, acceptance approx 0.014 |
| FP16 + speculative, draft 16 layers, nat 7 | 83.5% (167/200) | 0.682 s / image | 23.10 | 19.731 GB | HF assisted, Oct 26 2026, accuracy preserved exactly, but SLOWER; tokens per verification ~1.07, acceptance approx 0.010 |
| INT4 + pruning keep 50% (combined, no speculative) | 84.5% (169/200, +1.0 vs FP16) | 0.516 s / image | 30.32 | 6.257 GB peak (footprint 5.072 GB) | INT4 bnb NF4, FP16 layers 31,1,16,10,7, FastV keep 50%, Oct 26 2026, stratified POPE 200; adv 80.30%, pop 89.39%, rand 83.82% |
| INT4 + pruning keep 35% (combined, no speculative) | 85.0% (170/200, +1.5 vs FP16) | 0.515 s / image | 30.38 | 6.21 GB peak in fresh-process re-measurement, footprint 5.072 GB | INT4 bnb NF4, FP16 layers 31,1,16,10,7, FastV keep 35%, Oct 26 2026, stratified POPE 200; adv 83.33%, pop 89.39%, rand 82.35%; original sequential-run peak 9.803 GB was a measurement artifact, fresh run reproduced accuracy 170/200 exactly with peak 6.21 GB |
| INT4 + pruning + speculative | - | - | - | - | Excluded on purpose: Day 8-9 speculative was 67-69% slower, acceptance ~0.01-0.02, draft generates blank tokens |
| Merged draft, POPE 50 only, nat 5 | 90.0% (45/50, same as baseline on same 50) | 0.703 s/image vs baseline 0.408 | 22.66 vs 39.08 | - | Day 9, Oct 26 2026: draft alone generates repeating 'et' tokens, not Yes/No; merging layers does not fix draft, lm_head expects layer 32 states |
| Serving, combined keep 35%, FastAPI | - | median 0.554 s/request single, 2.248 s at conc 4, 4.466 s at conc 8 | throughput saturates ~1.78 req/s | - | Day 11, Oct 26 2026: 48/48 requests ok, same output, concurrency queues only, no throughput gain on one GPU |

Targets, not guarantees: 2 to 3x faster than FP16, VRAM under 6 GB, accuracy drop under 2 points.
If a target is missed, report why. That analysis is part of the project.
