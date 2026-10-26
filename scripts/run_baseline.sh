#!/usr/bin/env bash
set -e
python -m src.profiling.profile_model --config configs/baseline.yaml --image data/sample.jpg
python -m src.evaluation.evaluate --config configs/baseline.yaml
