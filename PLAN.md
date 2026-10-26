# SpecQuant-VLM: 2 week plan

Time box: about 2 to 3 hours a day for 14 days. Each day ends with a commit and a number written down, even if the number is bad.

## Week 1: Measure, quantize, prune

### Day 1: Baseline runs end to end
- Set up venv, install requirements, run `pytest tests/test_smoke.py`
- Add `data/sample.jpg`, get `src/profiling/profile_model.py` generating text
- Record FP16 latency, tokens per second, peak VRAM for 10 prompts
- Done when: baseline numbers are in `benchmarks/results_template.md`

### Day 2: Proper profiling and eval harness
- Split TTFT from decode time in the profiler (stream first token, time the rest)
- Wire POPE evaluation, limit 200, in `src/evaluation/evaluate.py`
- Append the first real row to `benchmarks/results.csv`
- Done when: you can rerun eval and get the same accuracy within noise

### Day 3: Layer sensitivity analysis
- Implement fake quantization in `src/quantization/sensitivity.py`
- Quantize one transformer block at a time to INT8, measure the drop on 50 to 100 examples
- Save the ranking to `outputs/sensitivity.json`, plot it in a notebook
- Done when: you have a ranked list and can name the 5 most sensitive layers

### Day 4: INT8 and INT4 models
- Get bitsandbytes INT8 working first with `configs/quant_int8.yaml`, measure it
- Then AWQ INT4 with `configs/quant_int4.yaml`, keeping the top 10 to 15 percent sensitive layers in FP16
- Done when: INT8 and INT4 rows are in the results table with accuracy next to them

### Day 5: Vision token pruning, exploration
- Run with `output_attentions=True` on 20 images, visualize attention on vision tokens
- Confirm the exact vision token offset for the LLaVA processor you are using. Do not trust the scaffold comment blindly
- Done when: you have a heatmap and a clear keep ratio hypothesis

### Day 6: Vision token pruning, implementation
- Implement the pruning patch in `src/pruning/vision_token_pruning.py`
- Ablate keep_ratio: 1.0, 0.5, 0.35, 0.25
- Done when: pruning row is in the table, with the accuracy vs latency tradeoff plotted

### Day 7: Buffer and write up week 1
- Fix whatever slipped, rerun the full table once cleanly
- Write a short week 1 note: what worked, what surprised you, what you would drop
- Ask Bharath to review the sensitivity ranking and pruning results at this point. That is the highest value review moment

## Week 2: Speculative decoding, combine, ship

### Day 8: Speculative decoding, easy path
- Use transformers assisted generation with a small draft model from `src/speculative/draft_models.py`
- Log tokens proposed, tokens accepted, acceptance rate, and wall time vs target only
- Tune `num_draft_tokens` in `configs/speculative.yaml`: try 3, 5, and 7
- Done when: you know your acceptance rate, even if speedup is modest

### Day 9: Merged draft experiment
- Merge 2 to 3 small instruction models using your model merging approach, or if merging takes too long, fine tune nothing and just compare 2 off the shelf drafts
- Run the same prompts, compare acceptance rate: single draft vs merged draft
- Done when: you have a yes or no answer, with numbers, on whether merging helps the draft

### Day 10: Combine the winners
- Best quantization + best keep ratio + best draft, in one config
- Run the full eval, check that accuracy losses do not stack badly
- Done when: the combined row is in the table. This is your headline result

### Day 11: Serving
- Serve only the winning variant with `src/serving/app.py` or vLLM
- Benchmark concurrent requests: 1, 4, and 8 at a time
- Done when: `/generate` works and you have a throughput number under load

### Day 12: Failure analysis
- Collect 20 examples where the optimized model is wrong and the baseline is right
- Group them: quantization errors, pruning dropped the key token, other
- Done when: you can explain your accuracy drop, not just report it

### Day 13: README and charts
- Finalize the README results table, add latency and accuracy plots
- Clean the repo: no large files, configs committed, instructions tested from scratch
- Done when: a friend can clone and reproduce the baseline in under 15 minutes

### Day 14: Ship it
- Write a short blog or LinkedIn post: problem, 4 techniques, final table, failure analysis
- Send the repo to Bharath for a final look
- Optional: record a 2 minute demo video showing FP16 vs optimized side by side

## Deliverables checklist

- [ ] Benchmark table with 5 variants, same eval for all
- [ ] Sensitivity ranking plot
- [ ] Accuracy vs latency plot for pruning keep ratios
- [ ] Speculative decoding acceptance rate log
- [ ] Failure analysis with 20 grouped examples
- [ ] Reproducible README

## Cut list, if you run out of time

Cut in this order, keep the benchmark table intact:
1. ONNX export
2. vLLM serving (keep the simple FastAPI server)
3. MMMU eval (POPE plus VQAv2 subset is enough)
4. Manual speculative loop (assisted generation numbers are fine)

Never cut: baseline profiling, sensitivity ranking, or failure analysis. Those are what separate this from a tutorial follow along.
