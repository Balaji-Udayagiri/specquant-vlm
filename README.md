# SpecQuant-VLM

Adaptive quantization, vision token pruning, and speculative decoding for a vision language model.

Goal: make LLaVA-1.5-7B run 2 to 3x faster on a single consumer GPU, fit in under 6 GB VRAM, and lose less than 2 points of accuracy. All numbers are measured, not claimed. If a target is missed, the repo should explain why.

Inspired by adaptive mixed precision quantization and efficient multimodal inference work with Qwen2.5-Omni and LLaVA: layer wise sensitivity, post hoc token compression, and speculative decoding, combined into one reproducible benchmark.

## What you build

1. A profiled FP16 baseline with latency, tokens per second, and peak VRAM
2. An INT8 and INT4 model where sensitive layers stay in FP16, based on a measured sensitivity ranking
3. Vision token pruning that drops 50 to 70 percent of the 576 image tokens after layer 2
4. Speculative decoding with a small draft model, plus an experiment using a merged draft model
5. One benchmark table and one short write up comparing every variant on the same prompts

## Repo layout

- `configs/` - one YAML per variant: baseline, quant_int8, quant_int4, pruning, speculative
- `src/common/` - model loading, config loading, timing and VRAM helpers. Use these everywhere so numbers stay comparable
- `src/profiling/` - Day 1 baseline profiler
- `src/quantization/` - sensitivity ranking, bitsandbytes and AWQ paths, ONNX export stub
- `src/pruning/` - vision token scoring and selection
- `src/speculative/` - draft model list and speculative decoding loop
- `src/evaluation/` - POPE / VQAv2 harness that appends rows to `benchmarks/results.csv`
- `src/serving/` - minimal FastAPI server for the winning variant only
- `benchmarks/results_template.md` - the final table to fill in
- `PLAN.md` - the day by day 2 week plan

## Quickstart, cloud only on RunPod

This project runs only on RunPod, not on a local machine. Follow ENVIRONMENT.md first to deploy an RTX 4090 pod with a 50GB Network Volume, then in the JupyterLab terminal on the pod:

```bash
cd /workspace/specquant-vlm
source /workspace/venv/bin/activate
pip install -r requirements.txt
pytest tests/test_smoke.py -v

# Day 1: add an image at data/sample.jpg, then
python -m src.profiling.profile_model --config configs/baseline.yaml --image data/sample.jpg --question "Describe this image in one sentence."
```

## Results, final, Oct 26 2026

Model: llava-hf/llava-1.5-7b-hf, RTX 4090, POPE 200 stratified
(66 adversarial, 66 popular, 68 random), greedy, max 16 tokens.
Full table in `benchmarks/results_template.md`, per-example data
in `outputs/*/eval.json`.

| Variant | Accuracy | Latency | Tokens/sec | Peak VRAM |
|---|---|---|---|---|
| FP16 baseline | 83.5% (167/200) | 0.412 s/image | 38.24 | 13.553 GB |
| INT8 bnb | 81.0% (-2.5) | 1.519 s/image | 10.29 | 7.262 GB |
| INT4 bnb NF4, top 5 layers FP16 | 82.5% (-1.0) | 0.543 s/image | 28.89 | 6.426 GB, footprint 5.072 GB |
| FP16 + pruning keep 50% | 81.5% (-2.0) | 0.386 s/image | 40.85 | 13.378 GB |
| FP16 + pruning keep 35% | 80.0% (-3.5) | 0.381 s/image | 41.45 | 13.331 GB |
| FP16 + pruning keep 25% | 79.5% (-4.0) | 0.384 s/image | 40.99 | 13.315 GB |
| FP16 + speculative nat 3/5/7 | 83.5% (all three) | 0.689 / 0.683 / 0.682 s/image | 22.87 / 23.08 / 23.10 | 19.731 GB |
| **Combined INT4 + pruning keep 50%** | **84.5% (+1.0)** | **0.516 s/image** | **30.32** | **6.257 GB, footprint 5.072 GB** |
| **Combined INT4 + pruning keep 35%, winning variant** | **85.0% (+1.5)** | **0.515 s/image** | **30.38** | **6.21 GB corrected, footprint 5.072 GB** |

Targets, honestly scored:
- 2 to 3x faster: **not met by any variant**. Best latency is
  FP16 pruning 0.381 s/image vs baseline 0.412, only about 8% faster.
  Combined is 25% slower than baseline.
- VRAM under 6 GB: **met on footprint** (5.072 GB for INT4 and
  combined), peaks 6.21 to 6.43 GB just above 6.
- Accuracy drop under 2 points: **met by INT4 alone, pruning
  keep 50%, and combined**, not by pruning 35/25 alone or INT8.
- Accuracy gains of 2 to 3 examples over baseline are within
  noise for n=200 (SE about 2.6 points). Defensible claim for
  combined is it does not hurt accuracy, not that it improves it.

Why speedup failed, three distinct reasons, all measured:
- Quantization: bitsandbytes dequantizes to FP16 on the fly for
  every matmul, paying FP16 compute plus dequant overhead.
  FP16 uses tuned Tensor Core kernels and already fits on 24 GB.
- Pruning: batch 1 generation is weight bandwidth bound. Each of
  16 steps reads the 14 GB weights regardless of KV length, so
  prefill savings do not move end to end latency.
- Speculative: draft acceptance about 0.01 to 0.02, tokens per
  verification about 1.07. Draft alone generates blanks (skipped
  layers) or repeating "et" tokens (merged layers, Day 9, POPE 50:
  90.0% accuracy, same as baseline, but 0.703 vs 0.408 s/image),
  because lm_head expects layer 32 hidden states.

Serving, Day 11, FastAPI, combined keep 35%:
- 48/48 requests succeeded, median single request 0.554 s.
- Throughput saturates at about 1.8 req/s on one GPU.
  Concurrency 4 median 2.248 s, concurrency 8 median 4.466 s,
  throughput unchanged at 1.77 to 1.78 req/s. Concurrency only
  queues, it does not scale a weight bound model.
- See `outputs/failure_analysis.md` for Day 12, and
  `outputs/review_note_bharath_draft.md` for the Day 7 review
  draft. Neither was sent without approval.
- Charts: `outputs/chart_latency_accuracy.png`,
  `outputs/chart_sensitivity.png`.

## Rules that keep this honest

- Same prompts, same max tokens, same evaluation limit for every variant
- Report accuracy drop next to every speedup, never speedup alone
- Log draft acceptance rate for speculative decoding, not just final latency
- One config file per run, committed with the results it produced

## Stretch goals, only after the table is done

- KV cache quantization and a vLLM serving benchmark
- ONNX Runtime benchmark for the vision tower
- Qwen2.5-VL-7B as a second target model with the same pipeline
