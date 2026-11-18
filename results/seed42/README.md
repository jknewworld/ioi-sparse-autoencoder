# Data from the saved run

This folder contains the prompts, model weights, and scores used in the report. The number 42 is the random seed for this run.

| File | What it contains |
| --- | --- |
| [sae_weights.npz](sae_weights.npz) | The six arrays that store the SAE weights and the values used to scale its inputs |
| [residuals.npz](residuals.npz) | The GPT-2 values after block 8, with one row per prompt. Each set has the same row order as its prompt file. |
| [sae_config.json](sae_config.json) | The model name, layer, size of the SAE, number of steps, and seed |
| [prompts/](prompts/) | The sentences, names, sentence patterns, and correct name order |
| [test_examples.csv](test_examples.csv), [ood_examples.csv](ood_examples.csv) | Each prompt's score gap and how much that gap changed when we removed feature 260 |
| [provenance.json](provenance.json) | The original file names, software versions, and hashes. A hash helps check that a file is an exact copy. |

The arrays hold the same numbers as the original PyTorch files. NumPy can open them with `allow_pickle=False`. The SAE learned from the training set. The development set helped us choose the feature. The test and OOD sets share names and use different sentence patterns.

Run `python -m analysis.verify` in the main project folder to check the numbers. The original CSV files give scores to five decimal places. [Checked results](../verified_metrics.json)
