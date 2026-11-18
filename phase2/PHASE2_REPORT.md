# Phase 2: Counterfactual Role Swaps and SAE Feature Steering

**Project:** IOI Sparse Autoencoder  
**Model:** GPT-2 Small  
**SAE:** saved Phase 1 Top-K SAE, layer 8, selected feature 260  
**Run:** 48 counterfactual pairs (96 prompts), seed 1234  
**Status:** Phase 2 output was checked against the submitted row-level CSV. The verifier did not rerun GPT-2.

## Research question

Does the SAE feature selected in Phase 1 respond to changing which name plays the indirect-object role? Does subtracting or adding its decoder contribution consistently affect GPT-2's next-name logit difference more than edits along eight control decoder directions?

## Method

The experiment reuses Phase 1's SAE and eight previously selected control feature IDs, without training or selecting features again. Each counterfactual pair keeps the same two names and sentence template, while changing which name is repeated as the subject. The indirect object changes accordingly. The 48 pairs use held-out test names but the original four training sentence templates.

For each of 96 prompts, we read the selected feature's SAE activation at the last token after GPT-2 block 8. We modified the residual stream by adding multiples of that activation times a unit-length SAE decoder direction. Strengths were -1, -0.5, +0.5, and +1; -1 subtracts the selected feature's decoded contribution and +1 adds one copy. Control directions used exactly the same **per-prompt selected-feature amplitude** (but their own decoder direction), allowing a matched edit-norm comparison. These are residual-stream edits; we did not retrain the model.

We evaluate the difference between GPT-2's next-token logit for the correct indirect object and that for the repeated subject. Positive values favor the correct name.

## Results

The baseline pairwise accuracy was **100%** and the baseline mean logit difference was **+3.2770**.

### Counterfactual response

- **Order AUROC:** 0.8125.
- **Pairs with a response in the expected direction:** 30/48 (62.5%). The other 18 pairs had zero difference, rather than an opposite response.
- **Selected feature active in:** 30/96 prompts (31.25%).
- **Template dependence:** All 30 affected prompts used template 0 (11 cases) or template 1 (19 cases) with the indirect object second. No effect was observed on the 18 corresponding template-2/3 pairs or on the 48 prompts with the indirect object first.

AUROC summarizes discrimination on these specific paired prompts. Because these prompts use the same templates as training, this is **not evidence that the result generalizes to new sentence templates**.

### Feature steering

| Strength | Mean change in correct-minus-subject logit gap | Pairwise accuracy |
| ---: | ---: | ---: |
| -1.0 | -0.10538 | 100% |
| -0.5 | -0.05171 | 100% |
| +0.5 | +0.04970 | 100% |
| +1.0 | +0.09731 | 100% |

The signs were consistent with subtracting versus adding the decoder contribution. The mean response was approximately proportional to intervention strength over the values tested. **No prompt changed its preferred name.** For the -1.0 setting, 30/96 prompts had a nonzero change, all negative.

### Matched-direction controls

At strength -1.0 the selected feature had an RMS gap change of **0.20531**. Across the eight control directions, the RMS changes ranged from **0.02419** to **0.13424**. Thus, the selected direction had a larger RMS effect than each of these eight controls, but this is not a general statistical guarantee against all possible controls or directions. Some controls affected the logit difference in the same direction and others in the opposite direction.

A pair-cluster bootstrap (10,000 resamples, seed 2026), computed as a separate exploratory analysis, gave an approximate 95% interval of **[-0.1337, -0.0772]** for the mean gap change after -1.0 steering. This interval describes resampling across the 48 pairs in this one run; it does not account for variability from retraining the SAE.

## Validation and provenance

The submitted archive contained `phase2_summary.json` and `steering_examples.csv`. The row table has **3,456 rows = 96 prompts × 9 directions × 4 strengths**, no missing fields, and no duplicate condition/strength/pair/order keys. Every stored per-condition mean, absolute mean, RMS, nonzero fraction, mean steered gap, and pairwise accuracy recomputed from the CSV matched the JSON summary. The pair structure, baseline consistency, and the identity `gap_change = steered_gap - original_gap` also passed.

SHA-256 of the submitted evidence files:

- `phase2_summary.json`: `cba5ec931600fe5c16f8a5f3624dc007f48d24320e5546d4c404d03007ab5e0d`
- `steering_examples.csv`: `31bf0787d3527b1c53342caa0934a6e8e88d465ab0e2b5dfc6e3371223c76cfd`

This was an **internal consistency check, not an independent rerun of GPT-2**, and the saved output alone cannot prove model identity or the full sequence of computations.

## Interpretation and limitations

The evidence supports a **template-dependent, directional causal effect** of one SAE decoder direction on GPT-2's logit gap on this prompt collection. It does **not** establish that feature 260 represents an abstract, general-purpose indirect-object concept. Role swaps change repeated words and multiple input tokens; they do not isolate a single semantic variable. The feature did not flip GPT-2's pairwise decision. We only tested one model, one layer, one SAE seed, eight controls, and four already-seen templates. Phase 1 had especially poor reconstruction on new OOD templates, and this experiment does not fix or retest that problem.

## Next recommended experiments

1. Add new sentence templates and balanced counterfactuals that reduce surface-form confounds.
2. Repeat SAE training with more seeds and compare features across independent runs.
3. Compare against matched random residual-stream directions and appropriately trained linear directions.
4. Report per-template effect sizes and confidence intervals, not just averages.

The results above were copied only after checking them against the uploaded data; no new GPT-2 runs were invented.