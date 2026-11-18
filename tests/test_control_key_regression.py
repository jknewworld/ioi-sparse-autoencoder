"""Check how the code names comparison features and draws their scores."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

import run_pipeline


class SAEStub:
    def reconstruction_metrics(self, activations):
        return {"mse": 0.1, "explained_variance": 0.9, "l0": 2.0, "dead_feature_fraction": 0.0}


def test_evaluate_split_control_keys_are_zero_based(tmp_path):
    rows = [{"io_first": i % 2, "prompt": "test", "io": " Alice", "subject": " Bob"} for i in range(4)]
    acts = torch.randn(4, 3)
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    patched = [(baseline + 0.3, 1.0), (baseline + 0.1, 1.0), (baseline + 0.2, 1.0)]
    with (patch.object(run_pipeline, "gaps_and_accuracy", side_effect=[(baseline, 1.0)] + patched),
          patch.object(run_pipeline, "feature_activations", return_value=np.ones((4, 15))),
          patch.object(run_pipeline, "heldout_auc", return_value=0.7),
          patch.object(run_pipeline, "feature_ablation_deltas", return_value=acts)):
        result, _ = run_pipeline.evaluate_split(
            None, None, SAEStub(), rows, acts, 8, 2, 9,
            [4, 5], 1, tmp_path, "test", "cpu"
        )
    assert result["selected_feature"]["feature_id"] == 9
    assert result["control_0"]["feature_id"] == 4
    assert result["control_1"]["feature_id"] == 5
    assert result["control_median_rms_gap_change"] > 0


def test_plot_filters_out_control_summary_scalar(tmp_path):
    def stat(rms):
        return {"rms_gap_change": rms}
    report = {"test": {
        "selected_feature": stat(1.0),
        "control_0": stat(0.3),
        "control_1": stat(0.2),
        "control_median_rms_gap_change": 0.25,
    }}
    run_pipeline.make_plots(report, tmp_path)
    assert (tmp_path / "causal_controls.png").exists()
