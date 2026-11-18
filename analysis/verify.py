"""Check the results on a CPU using the saved data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_stats import bootstrap_ci, heldout_auc, matched_controls, select_feature
from src.ioi_data import make_examples


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "seed42"
REFERENCE = ROOT / "results" / "verified_metrics.json"
SPLITS = ("train", "dev", "test", "ood")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs():
    provenance = load_json(RUN / "provenance.json")
    for name, expected in provenance["evidence_files_sha256"].items():
        digest = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"File checksum mismatch: {name}")

    config = load_json(RUN / "sae_config.json")
    prompt_dir = RUN / "prompts"
    if hashlib.sha256((prompt_dir / "train.json").read_bytes()).hexdigest() != config["train_file_sha256"]:
        raise ValueError("The training prompts do not match the ones used to save these weights")
    metadata = load_json(prompt_dir / "metadata.json")
    pools = metadata["names"]
    all_names = [name for pool in pools.values() for name in pool]
    if len(all_names) != len(set(all_names)) or metadata["seed"] != config["seed"]:
        raise ValueError("Unexpected name overlap or seed mismatch")

    rows = {}
    for offset, split in enumerate(SPLITS, start=1):
        rows[split] = load_json(prompt_dir / f"{split}.json")
        cached_digest = hashlib.sha256(json.dumps(rows[split], sort_keys=True).encode()).hexdigest()
        if cached_digest != provenance["cache_prompt_sha256"][split]:
            raise ValueError(f"Activation rows do not match the {split} prompts")
        names = pools["test" if split == "ood" else split]
        templates = metadata["templates_ood" if split == "ood" else "templates_train"]
        generated = make_examples(names, templates, len(rows[split]), config["seed"] + offset)
        if rows[split] != generated:
            raise ValueError(f"The new prompts for {split} differ from the saved prompts")
    return config, rows


def encode_and_score(raw, weights, k):
    """Use NumPy to rebuild the inputs from the saved SAE weights and measure the error."""
    x = (raw - weights["mean"]) / weights["scale"]
    scores = x @ weights["encoder.weight"].T + weights["encoder.bias"]
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    values = np.take_along_axis(scores, indices, axis=1)
    z = np.zeros_like(scores)
    np.put_along_axis(z, indices, np.maximum(values, 0), axis=1)
    decoder = weights["decoder_weight"]
    decoder = decoder / np.linalg.norm(decoder, axis=1, keepdims=True)
    reconstruction = z @ decoder + weights["decoder_bias"]
    mse = np.mean((reconstruction - x) ** 2, dtype=np.float64)
    split_variance = np.mean((x - x.mean(axis=0)) ** 2, dtype=np.float64)
    return z, {
        "mse_normalized": float(mse),
        "explained_variance": float(1.0 - mse / split_variance),
        "mean_active_features": float((z > 0).sum(axis=1).mean()),
        "inactive_feature_fraction": float((z == 0).all(axis=0).mean()),
    }


def score_logit_rows(split, rows, feature_active):
    with (RUN / f"{split}_examples.csv").open(encoding="utf-8", newline="") as stream:
        logged = list(csv.DictReader(stream))
    for row, obs in zip(rows, logged, strict=True):
        keys = (obs["prompt"], obs["correct_io"], obs["subject"], int(obs["io_first"]))
        if keys != (row["prompt"], row["io"], row["subject"], row["io_first"]):
            raise ValueError(f"Logit rows do not match the {split} prompt order")

    gap = np.array([float(r["original_logit_gap"]) for r in logged])
    change = np.array([float(r["selected_feature_ablation_gap_change"]) for r in logged])
    patched = gap + change
    if not np.isfinite(gap).all() or not np.isfinite(change).all():
        raise ValueError(f"Non-finite logit data on {split}")
    # Check that rounding could not change which name won.
    if np.min(np.abs(gap)) <= 5e-6 or np.min(np.abs(patched)) <= 1e-5:
        raise ValueError("A gap is too close to zero to tell which name had the higher score")
    changed = change != 0
    if not np.array_equal(changed, feature_active):
        raise ValueError(f"The active feature and the changed gaps do not match on {split}")

    order = np.array([r["io_first"] for r in rows])
    templates = np.array([r["template_id"] for r in rows])

    def group(mask):
        return {
            "n": int(mask.sum()),
            "changed": int(changed[mask].sum()),
            "negative": int((change[mask] < 0).sum()),
            "mean_change": float(change[mask].mean()),
        }

    by_template = {}
    for template_id in np.unique(templates):
        mask = templates == template_id
        by_template[str(template_id)] = {
            "io_first": group(mask & (order == 1)),
            "io_second": group(mask & (order == 0)),
        }
    return {
        "baseline_mean_gap": float(gap.mean()),
        "ablated_mean_gap": float(patched.mean()),
        "baseline_pairwise_accuracy": float((gap > 0).mean()),
        "ablated_pairwise_accuracy": float((patched > 0).mean()),
        "pairwise_decisions_changed": int(np.count_nonzero((gap > 0) != (patched > 0))),
        "mean_change": float(change.mean()),
        "mean_change_ci95": bootstrap_ci(change),
        "rms_change": float(np.sqrt(np.mean(change ** 2))),
        "by_order": {"io_first": group(order == 1), "io_second": group(order == 0)},
        "by_template": by_template,
    }


def recompute():
    config, rows = check_inputs()
    with np.load(RUN / "residuals.npz", allow_pickle=False) as cached:
        residuals = {split: cached[split] for split in SPLITS}
    with np.load(RUN / "sae_weights.npz", allow_pickle=False) as checkpoint:
        weights = dict(checkpoint)
    for split, x in residuals.items():
        if x.shape != (len(rows[split]), config["d_in"]):
            raise ValueError(f"The saved GPT-2 values have the wrong shape for {split}")
    train = residuals["train"]
    train_mean = train.mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(weights["mean"], train_mean, atol=5e-5, rtol=2e-5)
    np.testing.assert_allclose(weights["scale"], np.sqrt(np.mean((train - train_mean) ** 2)), rtol=2e-5)

    z = {}
    scores = {}
    for split in ("dev", "test", "ood"):
        z[split], reconstruction = encode_and_score(residuals[split], weights, config["k"])
        scores[split] = {"reconstruction": reconstruction}

    selected, sign, strength, frequency = select_feature(z["dev"], rows["dev"])
    controls = matched_controls(z["dev"], selected, strength, frequency, n=8, seed=config["seed"])
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500))
    probe.fit(residuals["dev"], [row["io_first"] for row in rows["dev"]])
    for split in ("dev", "test", "ood"):
        active = z[split][:, selected] > 0
        scores[split]["order_auroc"] = heldout_auc(z[split], rows[split], selected, sign)
        scores[split]["active_prompts"] = int(active.sum())
        if split in ("test", "ood"):
            y = [r["io_first"] for r in rows[split]]
            scores[split]["probe_order_auroc"] = float(roc_auc_score(y, probe.predict_proba(residuals[split])[:, 1]))
            scores[split]["logits"] = score_logit_rows(split, rows[split], active)
    return {
        "split_sizes": {split: len(rows[split]) for split in SPLITS},
        "feature_id": selected,
        "feature_sign_for_io_first": sign,
        "matched_control_ids": controls,
        "splits": scores,
    }


def compare(computed, expected, path="metrics"):
    if isinstance(expected, dict):
        if computed.keys() != expected.keys():
            raise ValueError(f"Metric fields differ at {path}")
        for key in expected:
            compare(computed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        for i, (a, b) in enumerate(zip(computed, expected, strict=True)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(computed, expected, rtol=1e-5, atol=5e-6, err_msg=path)
    elif computed != expected:
        raise ValueError(f"{path}: expected {expected}, got {computed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, help="Save both figures in this folder after checking the results")
    args = parser.parse_args()
    metrics = recompute()
    compare(metrics, load_json(REFERENCE))
    print("Checks passed for the files, prompts, selected feature, SAE errors, and scores.")
    for split in ("test", "ood"):
        s = metrics["splits"][split]
        gap = s["logits"]
        print(f"{split}: feature {metrics['feature_id']}, order AUROC {s['order_auroc']:.4f}, "
              f"average change in the gap {gap['mean_change']:.5f}, "
              f"{gap['pairwise_decisions_changed']} prompts switched to the other name")
    if args.figures:
        from .figures import draw_figures
        draw_figures(metrics, args.figures)


if __name__ == "__main__":
    main()
