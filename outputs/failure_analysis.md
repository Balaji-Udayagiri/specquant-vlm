# Day 12 failure analysis: combined INT4 + pruning keep 35% vs FP16 baseline

POPE 200, same stratified indices as all other variants.
Baseline: 167/200 = 83.5%. Combined keep 35%: 170/200 = 85.0%.

## Counts

- Both right: 156
- Combined breaks baseline: 11
- Combined fixes baseline: 14
- Both wrong: 19
- Net: +3 examples, within noise for n=200 (SE ~2.6 points).
  Honest claim: combined does not hurt accuracy, not that it truly improves it.

## Broken: 11 examples, only 5 distinct questions

POPE categories reuse the same COCO images. The 11 broken examples
collapse to 5 image-question pairs:

- Truck, gold yes, baseline yes, combined no: indices 6, 3006, 6006
  (adversarial, popular, random, same image and question)
- Backpack, gold yes, baseline yes, combined no: 48, 3048, 6048
- Person, gold yes, baseline yes, combined no: 56, 3056, 6056
- Toaster, gold no, baseline no, combined yes: 6031
- Bowl, gold no, baseline no, combined yes: 6043

So the combined model is not failing on 11 independent images.
It fails on 3 present-but-missed objects and 2 hallucinated objects.

## Attribution: pruning, not quantization, causes most breaks

Compare each broken example against the two single-technique variants:

- Pruning-only (pruning keep 35% wrong, INT4 right, combined wrong): 9
  All 9 are the truck / backpack / person Yes-to-No cases.
  Interpretation: FastV pruning at layer 2 dropped vision tokens
  that carried the object. The object is present but the model
  can no longer see it after pruning.
- Quantization-only (INT4 wrong, pruning right, combined wrong): 1
  Bowl, No-to-Yes.
- Both pruning and INT4 wrong: 1
  Toaster, No-to-Yes.
- Interaction-only (both single variants right, combined wrong): 0

Conclusion: do not blame INT4 for the combined failures.
Pruning is the main risk, quantization is a secondary risk.

## Fixed: 14 examples, mostly baseline hallucinations removed

- 11 of 14 fixed are baseline false positives: baseline said Yes,
  gold was No, combined correctly says No.
  Car (7, 3009), bus (25), cup (39), handbag (49, 55),
  chair (59, 3053, 3059), orange (6023), potted plant (6067).
- 3 of 14 fixed are baseline false negatives: baseline said No,
  gold was Yes, combined correctly says Yes.
  Car (22, 3022, 6022, same image repeated across categories).

Pattern: the combined model is more conservative than baseline.
It says No more often. That helps when baseline hallucinates an
object, and hurts when the object is present but weak in the
pruned vision tokens. That single tendency explains both the
broken and the fixed sets.

## Both wrong: 19 examples

Baseline errors the optimization does not fix:
adversarial 8, random 7, popular 4.
These are hard baseline errors, not optimization-introduced.
Baseline errors total 33 = 14 fixed + 19 both wrong.
An optimization that broke nothing and fixed the same 14 would
score 167 + 14 = 181/200 = 90.5%. The 19 both-wrong are the
remaining headroom that neither quantization nor pruning addresses;
fixing them would need a better base model or a different
technique, not a better quantizer or pruner.

## Union across all optimized variants

26 examples are broken by at least one optimized variant
(INT8, INT4, pruning 50/35/25, combined 50/35, speculative 3/5/7):
1, 6, 16, 20, 23, 40, 48, 56, 63, 65,
3006, 3011, 3016, 3020, 3040, 3048, 3056, 3061,
6006, 6016, 6020, 6031, 6040, 6043, 6048, 6056.
Speculative variants break 0, because they verify against the
full target model, so their accuracy equals baseline exactly.

## What to do next if accuracy mattered more than VRAM

- Raise keep ratio from 35% to 50%: pruning alone at 50% breaks
  only 8 vs 15 at 35%, and combined at 50% breaks only 5 vs 11
  at 35%, at the cost of a little more KV cache.
- Protect object tokens explicitly: keep tokens with high
  attention from the question token, not just from all text tokens,
  which is what FastV does now.
- Do not spend more time on INT4: it breaks only 6 alone, and
  only 1 combined failure is quantization-only.
