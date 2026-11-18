"""Phase 2: paired role swaps and feature steering for the saved SAE.

Requires the Phase 1 saved weights. This script does NOT retrain GPT-2 or the SAE.
No Phase 2 results are included in the repository until the experiment is run.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.ioi_data import TRAIN_TEMPLATES
from src.sae import TopKSAE
from src.experiment import collect_activations, feature_activations, gaps_and_accuracy

MODEL_ID = "openai-community/gpt2"


def load_sae(path, config_path, device):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        weights = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
    sae = TopKSAE(config["d_in"], config["d_sae"], config["k"]).to(device)
    state = sae.state_dict()
    for key in state:
        if key not in weights:
            raise ValueError(f"Missing SAE weight: {key}")
        if state[key].shape != weights[key].shape:
            raise ValueError(f"Wrong shape for {key}: {tuple(weights[key].shape)}")
        state[key] = weights[key].to(device=device, dtype=state[key].dtype)
    sae.load_state_dict(state)
    return sae.eval(), config


def make_role_pairs(names, n_pairs, seed):
    """Each pair keeps the two names and template, but changes who is repeated."""
    rng = random.Random(seed)
    candidates = []
    for template_id, template in enumerate(TRAIN_TEMPLATES):
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                candidates.append((template_id, template, first, second))
    rng.shuffle(candidates)
    if n_pairs > len(candidates):
        raise ValueError(f"Only {len(candidates)} unique name/template pairs available")
    rows = []
    for pair_id, (template_id, template, first, second) in enumerate(candidates[:n_pairs]):
        # The first prompt has the second person as the indirect object.
        for io_first, subject, io in [(0, first, second), (1, second, first)]:
            rows.append({
                "pair_id": pair_id,
                "template_id": template_id,
                "prompt": template.format(first=first, second=second, subject=subject),
                "io_first": io_first,
                "subject": subject,
                "io": io,
            })
    return rows


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "mean_abs": float(np.abs(a).mean()),
            "rms": float(np.sqrt(np.mean(a ** 2))),
            "fraction_nonzero": float(np.mean(np.abs(a) > 1e-7))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", default="results/seed42")
    parser.add_argument("--out", default="runs/phase2")
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    saved = Path(args.saved_run)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    verified = json.loads(Path("results/verified_metrics.json").read_text(encoding="utf-8"))
    selected = int(verified["feature_id"])
    controls = [int(x) for x in verified["matched_control_ids"]]
    metadata = json.loads((saved / "prompts" / "metadata.json").read_text(encoding="utf-8"))
    sae, config = load_sae(saved / "sae_weights.npz", saved / "sae_config.json", device)
    assert selected < sae.d_sae and all(0 <= i < sae.d_sae for i in controls)
    layer = int(config["layer"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device).eval()

    rows = make_role_pairs(metadata["names"]["test"], args.pairs, args.seed)
    acts = collect_activations(model, tokenizer, rows, layer, args.batch)
    z = feature_activations(sae, acts, device)
    baseline, base_accuracy = gaps_and_accuracy(model, tokenizer, rows, args.batch)
    selected_z = z[:, selected]

    # Compare the same ordered name pair across two different repeated-subject roles.
    first = np.asarray([i for i, r in enumerate(rows) if r["io_first"] == 0])
    second = first + 1
    assert all(rows[a]["pair_id"] == rows[b]["pair_id"] for a, b in zip(first, second))
    direction = int(verified["feature_sign_for_io_first"])
    paired_difference = direction * (selected_z[second] - selected_z[first])
    labels = np.array([r["io_first"] for r in rows])
    auc = float(roc_auc_score(labels, direction * selected_z))

    # Edit the original residual stream using a single SAE decoder direction.
    # Scale by each prompt's ORIGINAL selected-feature activation; controls receive
    # exactly the same per-prompt magnitude so that intervention sizes are matched.
    decoder = torch.nn.functional.normalize(sae.decoder_weight, dim=1).detach().cpu()
    scale = float(sae.scale.detach().cpu())
    magnitudes = torch.from_numpy(selected_z.astype(np.float32))[:, None] * scale
    conditions = [("selected", selected)] + [(f"control_{j}", fid) for j, fid in enumerate(controls)]
    strengths = (-1.0, -0.5, 0.5, 1.0)
    results = {}
    per_prompt = []

    for name, feature_id in conditions:
        results[name] = {"feature_id": int(feature_id), "steering": {}}
        for strength in strengths:
            delta = float(strength) * magnitudes * decoder[feature_id][None, :]
            patched, accuracy = gaps_and_accuracy(
                model, tokenizer, rows, args.batch, layer=layer, deltas=delta
            )
            changes = patched - baseline
            results[name]["steering"][str(strength)] = {
                **summarize(changes), "pairwise_accuracy": float(accuracy),
                "mean_gap_after": float(patched.mean()),
            }
            for row, before, after, change in zip(rows, baseline, patched, changes):
                per_prompt.append({
                    "condition": name, "feature_id": int(feature_id),
                    "strength": strength, "pair_id": row["pair_id"],
                    "template_id": row["template_id"], "io_first": row["io_first"],
                    "subject": row["subject"], "io": row["io"],
                    "prompt": row["prompt"], "original_gap": float(before),
                    "steered_gap": float(after), "gap_change": float(change),
                })
            print(f"{name:12s} strength={strength:+.1f} mean_gap_change={changes.mean():+.5f}")

    with (output / "steering_examples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_prompt[0]))
        writer.writeheader()
        writer.writerows(per_prompt)

    report = {
        "status": "completed", "model": MODEL_ID, "activation_site": f"layer {layer} block output",
        "selection": "Feature 260 and control IDs fixed from Phase 1; no reselection",
        "dataset": "Paired role swaps, held-out test-name pool, original training templates",
        "pairs": len(first), "prompts": len(rows), "seed": args.seed,
        "baseline_pairwise_accuracy": float(base_accuracy),
        "baseline_mean_gap": float(baseline.mean()),
        "feature_id": selected, "counterfactual": {
            "order_auc": auc,
            "signed_pair_difference": summarize(paired_difference),
            "fraction_pairs_in_expected_direction": float(np.mean(paired_difference > 0)),
            "feature_active_fraction": float(np.mean(selected_z > 0)),
        },
        "interventions": results,
        "limitations": [
            "Role swaps change the repeated name and therefore several input tokens.",
            "Paired tests assess role sensitivity, not semantic specificity by themselves.",
            "The SAE was trained with Phase 1 data and is not retrained here.",
            "Controls use the selected feature's per-prompt amplitude to match edit size.",
            "The result uses one SAE training seed and one model layer.",
        ],
    }
    (output / "phase2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output / "phase2_summary.json", "and", output / "steering_examples.csv")


if __name__ == "__main__":
    main()
