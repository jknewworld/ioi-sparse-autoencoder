# Phase 2: Counterfactual Role Swaps and SAE Feature Steering

**Status: implementation added; Phase 2 experiments have not been run or verified yet.**

Phase 1 selected SAE feature 260 using development prompts, and found that removing it lowered the correct-minus-subject logit gap on some held-out IOI prompts. The model's pairwise choice never changed. The next question is whether this feature follows the indirect object's role and whether its direction has a reliable effect beyond an ordinary perturbation.

## Research questions

1. **Counterfactual response:** With the same two names and sentence template, does feature 260 respond differently when the subject and indirect object swap roles?
2. **Dose response:** Does subtracting or adding multiples of the selected feature's contribution produce a systematic change in the correct-minus-subject logit gap?
3. **Controls:** Is the selected direction different from matched SAE decoder directions when all directions are given the same intervention magnitude on each prompt?

These are stronger tests than visual inspection, but no single one proves that a feature represents the semantic IO role.

## Run in Google Colab

Open a GPU runtime, then run these cells:

```python
!git clone --branch phase2-counterfactual-steering https://github.com/jknewworld/ioi-sparse-autoencoder.git
%cd /content/ioi-sparse-autoencoder
!pip install -q -r requirements.txt
!python -m pytest -q
```

Run a small experimental batch first:

```python
!python -m phase2.run_phase2 --pairs 8 --out runs/phase2_smoke
```

Then run the main paired study:

```python
!python -m phase2.run_phase2 --pairs 48 --out runs/phase2
```

The first run downloads GPT-2 Small. You need access to the saved Phase 1 weights under `results/seed42/` (already present in this repository).

## Outputs

- `phase2_summary.json`: paired feature statistics, intervention effects and stated limitations.
- `steering_examples.csv`: one row per prompt, intervention strength and control direction.

**Do not copy any Phase 2 numbers into README or the research report before checking that both output files are complete and the experiment ran without errors.**

## Method

The study uses the held-out test-name pool and Phase 1 sentence templates. It produces two prompts for each ordered pair of names. Both prompts retain the same first and second names and template; the repeated subject changes, so the indirect object also changes.

It uses the saved SAE and the original selected feature and control IDs without selecting a new feature. Steering strengths are `-1`, `-0.5`, `+0.5`, and `+1`, relative to the original selected feature's activation. A strength of `-1` removes its decoded contribution from the residual stream. A strength of `+1` adds one more copy. Controls use their own decoder direction but the selected feature's per-prompt activation magnitude, which makes the edit norm comparable.

The output reports mean and RMS logit-gap change, pairwise accuracy, and an AUROC for the role-swap prompts. Because the role swap changes text tokens as well as the semantic role, it is not a perfect single-variable causal intervention. The test also does not establish whether feature 260 generalizes to novel templates, new model layers, or other training seeds.

## Before calling Phase 2 complete

- Run the code on GPU and check that no errors occur.
- Verify the generated JSON and CSV, including the number of prompts and conditions.
- Examine effects separately for each template and each IO order.
- Confirm the selected direction is not merely matched by control directions.
- Add checked plots and a short report with negative as well as positive findings.
- Repeat with new SAE training seeds and consider new templates if feasible.

All code, comments, file names, and report text in this phase are English.
