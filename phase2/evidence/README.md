# Phase 2 evidence

These files preserve the 3,456 rows of the submitted Phase 2 `steering_examples.csv` experiment. The rows are split by steering direction to keep GitHub uploads small:

- `selected.csv`: SAE feature 260
- `control_0.csv` through `control_7.csv`: the eight fixed Phase 1 control directions

Every file has a header and 384 evidence rows (96 prompts × four strengths). Concatenating the rows in the order above, using the first header only and CRLF line endings, reconstructs the originally uploaded CSV **byte for byte**.

Original uploaded CSV SHA-256: `31bf0787d3527b1c53342caa0934a6e8e88d465ab0e2b5dfc6e3371223c76cfd`.

The `phase2_summary.json` file was reconstructed from the CSV metrics and the original saved run metadata. Its intervention means, RMS values and accuracies are checked against every raw row. It is not byte-identical to the uploaded summary JSON, whose original SHA-256 was `cba5ec931600fe5c16f8a5f3624dc007f48d24320e5546d4c404d03007ab5e0d`. Counterfactual feature-activation statistics are copied from that original run summary; they cannot be independently recomputed from the saved logit CSV alone.

From the project root, run:

```bash
python -m phase2.verify_results
```

This checks structural consistency, the uploaded original CSV hash and intervention summary values. It does **not** rerun GPT-2, authenticate model provenance or estimate variation over independent SAE training runs.
