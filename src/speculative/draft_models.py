"""Day 7-9: Draft model experiments.

Experiment A: off the shelf small model as draft (Qwen2.5-0.5B class).
Experiment B: Balaji twist. Merge 2 to 3 small instruction models with
your model merging method and use the merged model as the draft.
Hypothesis to test: a merged draft has a higher token acceptance rate
than any single draft because it covers more styles / prompts.

Log for every run: draft tokens proposed, tokens accepted, acceptance
rate, wall time vs target-only decoding.
"""

DRAFT_CANDIDATES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    # Add your merged checkpoint path here for Experiment B
]
