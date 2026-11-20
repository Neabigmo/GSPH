# -*- coding: utf-8 -*-
"""Main figure for the common-support mathematics and barcode evidence.

Two steps:

  collect  Reduce full union-audit outputs (experiments/union_audit_batch.py)
           to the small frozen tables the figure needs.
  draw     Draw the main figure (symmetrized graph) and the supplementary
           directed-graph figure from frozen tables under results/tables/ only.

Usage:
  python scripts/figure_common_support.py collect --rna <audit dir> --pd <audit dir>
  python scripts/figure_common_support.py draw
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

JUNG = float(np.sqrt(1.5))
BIN_WIDTH = 0.005
BIN_EDGES = np.arange(0.1975, 1.2575 + BIN_WIDTH / 2, BIN_WIDTH)  # ratio 1 sits mid-bin
NUMERICAL_BAR_LENGTH = 1e-6  # Å; shorter bars are solver round-off
COUNT_THRESHOLDS = (0.0, 1e-6, 1e-3, 1e-2)
MIN_BAR_LENGTH = 1e-2  # Å; bar counts on the figures use bars longer than this
EXAMPLES = {"rna": "FARFAR2-5SWD-S_000001_091_0001",
            "pd": "1HJC_ftd142s0d681551n4n17p23p18p126p10"}
TASK_LABEL = {"rna": "RNA", "pd": "Protein\u2013DNA"}
TASK_COLOR = {"rna": "#C2410C", "pd": "#1D4ED8"}
PAIRWISE_COLOR = "#8A8F98"
INK = "#1F2328"
LABEL_BOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)


# ---------------------------------------------------------------- collect

def bar_length(bar):
    birth, death = bar
    return np.inf if death is None else death - birth


def collect(audit_dirs):
    histogram, thresholds, examples = [], [], {}
    for task, directory in audit_dirs.items():
        ratio = pd.read_parquet(directory / "path_metrics.parquet", columns=["ratio"]).ratio.to_numpy()
        if ratio.min() < BIN_EDGES[0] or ratio.max() >= BIN_EDGES[-1]:
            raise ValueError(f"{task}: ratio outside histogram range")
        counts, _ = np.histogram(ratio, BIN_EDGES)
        histogram += [dict(task=task, bin_left=round(left, 4), bin_right=round(left + BIN_WIDTH, 4), n_paths=int(n))
                      for left, n in zip(BIN_EDGES[:-1], counts)]
        for path in sorted(directory.glob("*.barcodes.json")):
            record = json.loads(path.read_text())
            for order in (0, 1, 2):
                lengths = {name: np.array([bar_length(b) for b in record[name + "_A"][str(order)]])
                           for name in ("common", "pairwise")}
                for threshold in COUNT_THRESHOLDS:
                    thresholds.append(dict(
                        task=task, structure_id=record["structure_id"], graph=record["graph"], order=order,
                        min_length_A=threshold,
                        common_count=int((lengths["common"] > threshold).sum()),
                        pairwise_count=int((lengths["pairwise"] > threshold).sum())))
            if record["structure_id"] == EXAMPLES[task]:
                examples.setdefault(task, {})[record["graph"]] = dict(
                    structure_id=record["structure_id"], graph=record["graph"], unit="angstrom",
                    min_length_A=NUMERICAL_BAR_LENGTH,
                    **{name: {str(order): [[round(b, 6), None if d is None else round(d, 6)]
                                           for b, d in record[name + "_A"][str(order)]
                                           if bar_length((b, d)) > NUMERICAL_BAR_LENGTH]
                              for order in (1, 2)}
                       for name in ("common", "pairwise")})
    pd.DataFrame(histogram).to_csv(TABLES / "union_ratio_histogram.csv", index=False)
    table = pd.DataFrame(thresholds)
    table["count_delta"] = table.common_count - table.pairwise_count
    table.to_csv(TABLES / "union_barcode_count_by_min_length.csv", index=False)
    (TABLES / "union_barcode_intervals_examples.json").write_text(json.dumps(examples, indent=1))


# ------------------------------------------------------------------- draw

def registered():
    """Numbers printed on the figure come from the audited tables only."""
    ratio = {"rna": pd.read_csv(TABLES / "union_rna_ratio_distribution.csv"),
             "pd": pd.read_csv(TABLES / "union_audit_pd_ratio_distribution.csv")}
    bars = {"rna": pd.read_csv(TABLES / "union_rna_barcode_summary.csv"),
            "pd": pd.read_csv(TABLES / "union_audit_pd_barcodes.csv")}
    counts = pd.read_csv(TABLES / "union_barcode_count_by_min_length.csv")
    counts = counts[np.isclose(counts.min_length_A, MIN_BAR_LENGTH)]
    keys = ["structure_id", "graph", "order"]
    for task, table in bars.items():
        merged = table[keys + ["different", "finite_total_persistence_delta_A"]].merge(
            counts[keys + ["common_count", "pairwise_count", "count_delta"]], on=keys, validate="one_to_one")
        if len(merged) != len(table):
            raise ValueError(f"{task}: thresholded bar counts do not cover the audited barcodes")
        bars[task] = merged
    return ratio, bars


def panel_label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="baseline", color=INK)
    fig.text(x + 0.022, y, title, fontsize=8.4, fontweight="bold", va="baseline", color=INK)


def draw_supports(ax, radius, verdicts, title, witness=False):
    from matplotlib.patches import Circle, Polygon
    centers = np.array([[-1, 0], [1, 0], [0, np.sqrt(3)]]) - [0, 1 / np.sqrt(3)]
    colors = ["#F59E0B", "#10B981", "#6366F1"]
    for center, color in zip(centers, colors):
        ax.add_patch(Circle(center, radius, facecolor=color, alpha=0.26, edgecolor=color, lw=1.1))
    ax.add_patch(Polygon(centers, closed=True, facecolor="#111827" if witness else "none",
                         alpha=0.10 if witness else 1, edgecolor=INK, lw=1.0, ls="-" if witness else (0, (3, 2))))
    ax.scatter(*centers.T, s=13, color=INK, zorder=5)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        ax.scatter(*((centers[i] + centers[j]) / 2), s=11, marker="o", facecolor="white", edgecolor=INK, lw=0.7, zorder=6)
    if witness:
        ax.scatter(0, 0, s=95, marker="*", color="#DC2626", edgecolor="white", lw=0.5, zorder=7)
    else:
        ax.scatter(0, 0, s=38, marker="x", color="#DC2626", lw=1.3, zorder=7)
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-1.95, 2.45); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 2.62, title, ha="center", va="bottom", fontsize=7.2, color=INK)
    for k, (name, value, good) in enumerate(verdicts):
        y = -2.3 - 0.5 * k
        ax.text(0.35, y, name, ha="right", va="center", fontsize=6.8, color=INK)
        ax.text(0.5, y, value, ha="left", va="center", fontsize=6.8, fontweight="bold",
                color="#15803D" if good else "#DC2626")


def draw_ratio(ax, histogram, ratio_tables):
    from matplotlib.patches import Rectangle
    floor, ceiling, right = 0.5, 3e6, 1.33
    for task in ("rna", "pd"):
        rows = histogram[histogram.task == task]
        counts = rows.n_paths.to_numpy().astype(float)
        edges = np.append(rows.bin_left.to_numpy(), rows.bin_right.to_numpy()[-1])
        ax.stairs(np.where(counts > 0, counts, floor), edges, baseline=floor, fill=True,
                  color=TASK_COLOR[task], alpha=0.8 if task == "rna" else 0.9, lw=0, label=TASK_LABEL[task])
    ax.set_yscale("log"); ax.set_ylim(floor, ceiling); ax.set_xlim(0.2, right)
    ax.set_yticks([1, 1e2, 1e4])
    ax.axvspan(1.0025, JUNG, color="#FDE68A", alpha=0.5, lw=0, zorder=0)
    ax.add_patch(Rectangle((JUNG, floor), right - JUNG, ceiling, facecolor="#FEE2E2", edgecolor="#F87171",
                           hatch="////", lw=0, zorder=0))
    ax.axvline(JUNG, color="#DC2626", lw=1.6)
    ax.axvline(1.0, color=INK, lw=0.7, ls=(0, (3, 2)))
    n_paths = sum(int(t.n_paths.sum()) for t in ratio_tables.values())
    violations = sum(int(t.bound_violations.sum()) for t in ratio_tables.values())
    largest = max(float(t["max"].max()) for t in ratio_tables.values())
    ax.annotate(f"largest\n{largest:.3f}", xy=(largest, 4), xytext=(1.15, 1.5e3), fontsize=6.6,
                ha="center", color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkB=1))
    ax.text(JUNG, ceiling * 1.25, r"$\sqrt{3/2}$ bound", ha="center", va="bottom", fontsize=7.4,
            color="#DC2626", fontweight="bold")
    ax.text((JUNG + right) / 2 + 0.004, 40, f"{violations}\npaths", ha="center", va="center", fontsize=7.4,
            color="#DC2626", fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.text((1.0 + JUNG) / 2, ceiling * 0.6, "beyond\npairwise", ha="center", va="top", fontsize=6.4, color="#92400E",
            fontweight="bold")
    ax.text(0.225, ceiling * 0.6, f"all {n_paths:,} paths of 16 structures", ha="left", va="top", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.set_xlabel(r"support ratio  $(r_{\min}+\tau)\,/\,(r_{\max}+\tau_{\rm pw}+\delta)$", fontsize=7.6)
    ax.set_ylabel("paths per bin", fontsize=7.6)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.8, frameon=False, handlelength=1.0, ncol=2,
              columnspacing=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_barcode(ax, intervals, task, order, row):
    common, pairwise = (sorted((b for b in intervals[name][str(order)] if bar_length(b) > MIN_BAR_LENGTH),
                               key=lambda b: (b[0], bar_length(b))) for name in ("common", "pairwise"))
    if (len(common), len(pairwise)) != (int(row.common_count), int(row.pairwise_count)):
        raise ValueError(f"{task}: drawn bars do not match the counted bars")
    finite = [d for _, d in common + pairwise if d is not None]
    right = max(finite) * 1.03
    for bars, sign, color in ((common, 1, TASK_COLOR[task]), (pairwise, -1, PAIRWISE_COLOR)):
        for k, (birth, death) in enumerate(bars):
            ax.plot([birth, right if death is None else death], [sign * (k + 1)] * 2,
                    color=color, lw=0.85, solid_capstyle="butt")
    ax.axhline(0, color=INK, lw=0.7)
    top = max(len(common), len(pairwise), 4)
    ax.set_ylim(-top * 1.05, top * 1.05); ax.set_xlim(0, right)
    ax.set_yticks([]); ax.tick_params(axis="x", labelsize=6.6, length=2)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.text(0.02, 0.97, f"{TASK_LABEL[task]}  H$_{order}$", transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="top", color=INK)
    ax.text(0.02, 0.74, f"common support\n{int(row.common_count)} bars", transform=ax.transAxes, fontsize=6.8,
            va="center", color=TASK_COLOR[task], fontweight="bold", bbox=LABEL_BOX)
    ax.text(0.02, 0.26, f"pairwise support\n{int(row.pairwise_count)} bars", transform=ax.transAxes, fontsize=6.8,
            va="center", color="#5B616B", fontweight="bold", bbox=LABEL_BOX)
    ax.text(0.98, 0.04, f"+{int(row.count_delta)} bars\n+{float(row.finite_total_persistence_delta_A):.2f} \u00c5 persistence", transform=ax.transAxes, fontsize=7.2,
            ha="right", va="bottom", color=TASK_COLOR[task], fontweight="bold", bbox=LABEL_BOX)


def draw_gain(ax, bar_tables):
    for task in ("pd", "rna"):
        rows = bar_tables[task]
        rows = rows[(rows.graph == "und") & (rows.order == 2)]
        if not (rows.different.all() and (rows.count_delta > 0).all() and len(rows) == 8):
            raise ValueError(f"{task}: panel d states that all 8 structures gain bars")
        ax.scatter(rows.count_delta, rows.finite_total_persistence_delta_A, s=34, color=TASK_COLOR[task],
                   edgecolor="white", lw=0.6, label=f"{TASK_LABEL[task]} (8/8 gain)", zorder=3)
        star = rows[rows.structure_id == EXAMPLES[task]]
        ax.scatter(star.count_delta, star.finite_total_persistence_delta_A, s=95, facecolor="none",
                   edgecolor=TASK_COLOR[task], lw=0.9, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.7, 200); ax.set_ylim(0.02, 12)
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
    ax.grid(True, which="major", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xlabel("extra H$_2$ bars longer than 0.01 \u00c5", fontsize=7.6)
    ax.set_ylabel("extra total persistence (\u00c5)", fontsize=7.6)
    ax.legend(loc="upper left", fontsize=6.6, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    ax.text(0.97, 0.05, "symmetrized graph, H$_2$:\nevery structure gains bars", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.8, color=INK, fontweight="bold")


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.7,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "axes.edgecolor": INK,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    return matplotlib, plt


def save(fig, out_stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=600)
    return out_stem


def draw_main(out_stem):
    matplotlib, plt = style()
    ratio_tables, bar_tables = registered()
    histogram = pd.read_csv(TABLES / "union_ratio_histogram.csv")
    for task, table in ratio_tables.items():  # frozen histogram must describe the audited paths
        if int(histogram[histogram.task == task].n_paths.sum()) != int(table.n_paths.sum()):
            raise ValueError(f"{task}: histogram does not match the audited path count")
    examples = json.loads((TABLES / "union_barcode_intervals_examples.json").read_text())

    fig = plt.figure(figsize=(7.2, 6.3))
    fig.suptitle("Common support sees what pairwise support cannot, and never by more than $\\sqrt{3/2}$",
                 fontsize=10.2, fontweight="bold", x=0.5, y=0.985)
    top = fig.add_gridspec(1, 2, left=0.035, right=0.985, top=0.855, bottom=0.575, width_ratios=[1.0, 1.08], wspace=0.2)
    left = top[0].subgridspec(1, 2, wspace=0.02)
    ax_pair, ax_common = fig.add_subplot(left[0]), fig.add_subplot(left[1])
    draw_supports(ax_pair, 1.06, [("pairwise support", "present", True),
                                  ("common support", "absent", False)],
                  r"scale $\tau_{\rm pw}$: every pair meets")
    draw_supports(ax_common, 2 / np.sqrt(3) + 0.03, [("pairwise support", "present", True),
                                                     ("common support", "born here", True)],
                  r"scale $\tau$: one point in all three", witness=True)
    fig.patches.append(matplotlib.patches.FancyArrowPatch(
        (0.238, 0.745), (0.276, 0.745), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
        color=INK, lw=0.9))
    fig.text(0.257, 0.757, r"gap $\eta$", ha="center", va="bottom", fontsize=7, color="#DC2626", fontweight="bold")

    ax_ratio = fig.add_subplot(top[1])
    draw_ratio(ax_ratio, histogram, ratio_tables)

    bottom = fig.add_gridspec(1, 3, left=0.035, right=0.985, top=0.435, bottom=0.075,
                              width_ratios=[1, 1, 1.12], wspace=0.14)
    bottom_axes = []
    for k, task in enumerate(("rna", "pd")):
        ax = fig.add_subplot(bottom[k])
        rows = bar_tables[task]
        row = rows[(rows.structure_id == EXAMPLES[task]) & (rows.graph == "und") & (rows.order == 2)].iloc[0]
        draw_barcode(ax, examples[task]["und"], task, 2, row)
        ax.set_xlabel("filtration scale (\u00c5)", fontsize=7.6)
        bottom_axes.append(ax)
    gain = bottom[2].subgridspec(1, 1)
    ax_gain = fig.add_subplot(gain[0])
    box = ax_gain.get_position()
    ax_gain.set_position([box.x0 + 0.05, box.y0, box.width - 0.05, box.height])
    draw_gain(ax_gain, bar_tables)

    panel_label(fig, 0.035, 0.915, "a", "Every pair can meet while no point is common")
    panel_label(fig, 0.515, 0.915, "b", "In molecules the gap stays under the bound")
    panel_label(fig, 0.035, 0.468, "c", "On the same paths, common support gives the richer H$_2$ barcode")
    panel_label(fig, 0.685, 0.468, "d", "16 of 16 structures gain")

    return save(fig, out_stem)


def short_name(task, structure_id):
    if task == "rna":
        return "-".join(structure_id.split("-")[:2])
    target, decoy = structure_id.split("_ftd")
    return f"{target} {decoy.split('s0d')[0]}"


def draw_counts(ax, bar_tables, order):
    labels, ticks, y = [], [], 0.0
    for task in ("rna", "pd"):
        rows = bar_tables[task]
        rows = rows[(rows.graph == "dir") & (rows.order == order)].sort_values("structure_id")
        ax.annotate(f"{TASK_LABEL[task]}: {int(rows.different.sum())} of 8 barcodes differ", (1.0, y - 0.95),
                    xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=7.2, fontweight="bold",
                    color=TASK_COLOR[task])
        for row in rows.itertuples():
            ax.plot([row.pairwise_count, row.common_count], [y, y], color=TASK_COLOR[task], lw=1.6, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.scatter(row.pairwise_count, y, s=20, color=PAIRWISE_COLOR, zorder=3)
            ax.scatter(row.common_count, y, s=20, color=TASK_COLOR[task], zorder=4)
            if not row.different:
                ax.annotate("identical barcode", (max(row.common_count, row.pairwise_count), y), xytext=(6, 0),
                            textcoords="offset points", fontsize=6.2, va="center", color="#5B616B", style="italic")
            labels.append(short_name(task, row.structure_id)); ticks.append(y); y += 1
        y += 1.4
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.4)
    ax.margins(x=0.08)
    ax.set_ylim(ticks[-1] + 0.7, -1.6)
    ax.grid(True, axis="x", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xlabel(f"H$_{order}$ bars longer than 0.01 \u00c5", fontsize=7.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_directed(out_stem):
    matplotlib, plt = style()
    from matplotlib.lines import Line2D
    _, bar_tables = registered()
    examples = json.loads((TABLES / "union_barcode_intervals_examples.json").read_text())
    differ = {task: int(t[(t.graph == "dir") & (t.order == 1)].different.sum()) for task, t in bar_tables.items()}
    fig = plt.figure(figsize=(7.2, 6.6))
    fig.suptitle(f"Directed graph: common support changes H$_1$ in {differ['rna']} of 8 RNA "
                 f"and {differ['pd']} of 8 protein\u2013DNA structures", fontsize=9.6, fontweight="bold", y=0.985)
    top = fig.add_gridspec(1, 2, left=0.035, right=0.985, top=0.895, bottom=0.6, wspace=0.12)
    for k, task in enumerate(("rna", "pd")):
        ax = fig.add_subplot(top[k])
        rows = bar_tables[task]
        row = rows[(rows.structure_id == EXAMPLES[task]) & (rows.graph == "dir") & (rows.order == 1)].iloc[0]
        draw_barcode(ax, examples[task]["dir"], task, 1, row)
        ax.set_xlabel("filtration scale (\u00c5)", fontsize=7.6)
    bottom = fig.add_gridspec(1, 2, left=0.13, right=0.985, top=0.475, bottom=0.07, wspace=0.34)
    for k, order in enumerate((1, 2)):
        ax = fig.add_subplot(bottom[k])
        draw_counts(ax, bar_tables, order)
        if k == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=PAIRWISE_COLOR, markersize=4, label="pairwise support"),
                               Line2D([], [], marker="o", ls="", color=INK, markersize=4, label="common support")],
                      loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=6.6, frameon=False, handletextpad=0.2)
    panel_label(fig, 0.035, 0.925, "a", "H$_1$ barcodes on the directed graph, same structures as the main figure")
    panel_label(fig, 0.035, 0.508, "b", "H$_1$ bar counts per structure")
    panel_label(fig, 0.535, 0.508, "c", "H$_2$ bar counts per structure")
    return save(fig, out_stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--rna", type=Path, required=True)
    collect_parser.add_argument("--pd", type=Path, required=True)
    commands.add_parser("draw")
    args = parser.parse_args()
    if args.command == "collect":
        collect({"rna": args.rna, "pd": args.pd})
    else:
        print("wrote", draw_main(FIGURES / "fig_common_support_main").name)
        print("wrote", draw_directed(FIGURES / "fig_common_support_directed").name)


if __name__ == "__main__":
    main()
