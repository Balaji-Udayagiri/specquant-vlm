#!/usr/bin/env bash
set -e
for cfg in configs/baseline.yaml configs/quant_int8.yaml configs/quant_int4.yaml configs/pruning.yaml configs/speculative.yaml; do
  echo "=== $cfg ==="
  python -m src.evaluation.evaluate --config "$cfg"
done
cat benchmarks/results.csv
