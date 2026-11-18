"""Verify Phase 2 evidence using the saved row-level steering results.

This checks stored outputs, NOT an independent GPT-2 forward pass.
Run: python -m phase2.verify_results
"""
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / "evidence"
ORDER = ["selected"] + [f"control_{i}" for i in range(8)]
STRENGTHS = {-1.0, -0.5, 0.5, 1.0}
EXPECTED_SHA256 = "31bf0787d3527b1c53342caa0934a6e8e88d465ab0e2b5dfc6e3371223c76cfd"


def avg(values):
    return sum(values) / len(values)


def check():
    summary = json.loads((EVIDENCE / "phase2_summary.json").read_text(encoding="utf-8"))
    assert summary["pairs"] == 48 and summary["prompts"] == 96
    raw_lines = []
    rows = []
    header = None
    for name in ORDER:
        path = EVIDENCE / f"{name}.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 385, (name, len(lines))
        if header is None:
            header = lines[0]
        assert lines[0] == header
        raw_lines.extend(lines[1:])
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == 384
        assert all(r["condition"] == name for r in records)
        rows.extend(records)

    raw = (header + "\r\n" + "\r\n".join(raw_lines) + "\r\n").encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == EXPECTED_SHA256, f"Original evidence SHA-256 mismatch: {sha}"
    assert len(rows) == 3456 and len(set(
        (r["condition"], r["strength"], r["pair_id"], r["io_first"]) for r in rows
    )) == len(rows)

    prompts = {}
    for row in rows:
        cond = row["condition"]
        strength = float(row["strength"])
        pair = int(row["pair_id"])
        io_first = int(row["io_first"])
        assert strength in STRENGTHS and 0 <= pair < 48 and io_first in (0, 1)
        assert int(row["feature_id"]) == summary["interventions"][cond]["feature_id"]
        before = float(row["original_gap"])
        after = float(row["steered_gap"])
        delta = float(row["gap_change"])
        assert all(math.isfinite(x) for x in (before, after, delta))
        assert math.isclose(after - before, delta, abs_tol=1e-8)
        key = (pair, io_first)
        signature = (row["prompt"], row["subject"], row["io"], row["template_id"], before)
        if key in prompts:
            assert prompts[key] == signature
        else:
            prompts[key] = signature

    assert len(prompts) == 96
    for pair in range(48):
        a, b = prompts[(pair, 0)], prompts[(pair, 1)]
        assert a[1] == b[2] and a[2] == b[1] and a[3] == b[3]

    for name in ORDER:
        info = summary["interventions"][name]
        for strength in STRENGTHS:
            sub = [r for r in rows if r["condition"] == name and float(r["strength"]) == strength]
            assert len(sub) == 96
            changes = [float(r["gap_change"]) for r in sub]
            afters = [float(r["steered_gap"]) for r in sub]
            befores = [float(r["original_gap"]) for r in sub]
            got = {
                "mean": avg(changes),
                "mean_abs": avg([abs(x) for x in changes]),
                "rms": math.sqrt(avg([x*x for x in changes])),
                "fraction_nonzero": avg([float(abs(x) > 1e-7) for x in changes]),
                "pairwise_accuracy": avg([float(x > 0) for x in afters]),
                "mean_gap_after": avg(afters),
            }
            expected = info["steering"][str(strength)]
            for metric, value in got.items():
                assert math.isclose(value, expected[metric], abs_tol=1e-7), (name, strength, metric)
            assert math.isclose(avg(befores), summary["baseline_mean_gap"], abs_tol=1e-7)

    print("PASS: original CSV SHA-256, 3456 rows, 48 pairs, 9 directions, 4 strengths")
    print("PASS: all per-condition aggregate measurements match the row-level evidence")
    print("NOTE: this does not independently rerun GPT-2 or verify model provenance")


if __name__ == "__main__":
    check()
