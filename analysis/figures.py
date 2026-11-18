"""Draw the figures from the checked results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


INK = "#203343"
MUTED = "#516270"
GRID = "#dee6e8"
TEAL = "#146b71"
ORANGE = "#a95018"


def style_axis(ax):
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)


def draw_figures(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 12,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        representation_figure(metrics, output_dir / "reconstruction_and_order.png")
        ablation_figure(metrics, output_dir / "ablation_by_template.png")


def representation_figure(metrics, output):
    splits = ("dev", "test", "ood")
    rows = np.arange(3)
    labels = ["Development", "Test", "New patterns"]
    colors = [TEAL, TEAL, ORANGE]
    fig, (variance, auc) = plt.subplots(1, 2, figsize=(8.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.16, right=0.96, top=0.81, bottom=0.24, wspace=0.24)
    for ax in (variance, auc):
        style_axis(ax)
        ax.set_ylim(2.5, -0.5)
        ax.set_yticks(rows)
    variance.set_yticklabels(labels)
    auc.set_yticklabels([])
    variance.set_title("Rebuilding GPT-2 values", loc="left", weight="bold", pad=12)
    auc.set_title(f"Name order with feature {metrics['feature_id']}", loc="left", weight="bold", pad=12)
    ev = [metrics["splits"][s]["reconstruction"]["explained_variance"] for s in splits]
    variance.barh(rows, ev, height=0.36, color=colors)
    variance.axvline(0, c=INK, lw=0.9)
    variance.set_xlim(-0.58, 1.21)
    variance.set_xticks([-0.5, 0, 0.5, 1])
    variance.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    variance.set_xlabel("Explained variance", labelpad=9)
    for y, x in zip(rows, ev):
        # Put labels next to the bars, with room at each outer edge.
        variance.text(x + (0.025 if x >= 0 else -0.025), y, f"{x:.1%}",
                      va="center", ha="left" if x >= 0 else "right", fontsize=9)
    auc.set_xlim(0.455, 1.02)
    auc.set_xticks([0.5, 0.75, 1.0], ["0.50", "0.75", "1.00"])
    auc.axvline(0.5, color=MUTED, linestyle=":", linewidth=1)
    auc.set_xlabel("Name order score (AUROC, chance = 0.5)", labelpad=9)
    for y, s, color in zip(rows, splits, colors):
        x = metrics["splits"][s]["order_auroc"]
        auc.plot(x, y, "o", ms=8, color=color)
        auc.text(x + 0.022, y, f"{x:.4f}", va="center", fontsize=9)
    fig.savefig(output, dpi=180, metadata={"Title": "Reconstruction and name-order prediction"})
    plt.close(fig)


def ablation_figure(metrics, output):
    templates = [("test", n) for n in range(4)] + [("ood", n) for n in range(2)]
    names = ["Store", "Park", "School", "Market", "Hiking", "Trip"]
    fig, (mean_ax, counts) = plt.subplots(1, 2, figsize=(8.2, 3.55), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.145, right=0.98, top=0.76, bottom=0.21, wspace=0.47)
    style_axis(mean_ax)
    mean_ax.set_title("Average change in the gap", loc="left", weight="bold", pad=15)
    mean_ax.set_ylim(1.55, -0.55)
    mean_ax.set_yticks([0, 1], ["Test", "New patterns"])
    mean_ax.set_xlim(-0.14, 0.011)
    mean_ax.set_xticks([-0.12, -0.08, -0.04, 0.0], ["−0.12", "−0.08", "−0.04", "0"])
    mean_ax.axvline(0, c=INK, linewidth=0.8)
    mean_ax.set_xlabel("After minus before (logits)", labelpad=9)
    for y, (s, color) in enumerate((("test", TEAL), ("ood", ORANGE))):
        v = metrics["splits"][s]["logits"]
        mean = v["mean_change"]
        lo, hi = v["mean_change_ci95"]
        mean_ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o",
                         color=color, markersize=7, capsize=4, linewidth=2)
        mean_ax.text(mean, y - 0.22, f"{mean:+.3f}", ha="center", va="center", fontsize=9)
    counts.set_title("How many gaps changed?", loc="center", weight="bold", pad=26)
    counts.set_xlim(-0.55, 1.55)
    counts.set_ylim(5.5, -0.5)
    counts.set_xticks([0, 1], ["Recipient first", "Recipient second"])
    counts.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                       length=0, colors=MUTED, labelsize=9)
    counts.set_yticks(range(6), names)
    counts.tick_params(axis="y", pad=6)
    counts.spines[:].set_visible(False)
    for i, (split, template_id) in enumerate(templates):
        color = TEAL if split == "test" else ORANGE
        for column, order in enumerate(("io_first", "io_second")):
            v = metrics["splits"][split]["logits"]["by_template"][str(template_id)][order]
            alpha = 0.06 if v["changed"] == 0 else 0.22 + 0.68 * v["changed"] / v["n"]
            counts.add_patch(plt.Rectangle((column - 0.48, i - 0.46), 0.96, 0.92,
                                            facecolor=color, alpha=alpha, edgecolor="none"))
            counts.text(column, i, f"{v['changed']}/{v['n']}", ha="center", va="center",
                        fontsize=9, color="white" if alpha > 0.6 else INK)
    counts.axhline(3.5, color=GRID, linewidth=1)
    fig.savefig(output, dpi=180, metadata={"Title": "What changed after removing the feature?"})
    plt.close(fig)
