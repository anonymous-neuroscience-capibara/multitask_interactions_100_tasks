#!/usr/bin/env python
"""Per-task test accuracy vs N (nonlinear units), one line per P, seeds averaged.

Small-multiples grid: one panel per task. Within each panel, best-epoch test
accuracy is plotted against N for every P value (P is ordinal, so lines are
colored on a sequential light->dark = low->high P scale). Accuracy is averaged
over the available seeds; a faint band shows the seed min-max spread.

Reads the per-run metadata.json files written by the foundation training
(fields: nonlinear_units=N, num_individual_params=P, seed,
best_test_accuracy_per_task). Pure matplotlib/numpy.

Usage:
    python "foundation model/plot_acc_vs_N.py"
    python "foundation model/plot_acc_vs_N.py" --data-dir "data/M64_lr5e4_NPsweep" --out acc_vs_N.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_runs(data_dir):
    """acc[task][P][N] = list of per-seed accuracies; plus sorted N, P, task order."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    Ns, Ps = set(), set()
    task_order = None
    metas = sorted(glob.glob(os.path.join(data_dir, "*", "metadata.json")))
    n_ok = 0
    for m in metas:
        try:
            d = json.load(open(m))
        except Exception:
            continue
        N, P = d.get("nonlinear_units"), d.get("num_individual_params")
        per_task = d.get("best_test_accuracy_per_task")
        if N is None or P is None or not per_task:
            continue
        # keep a stable task order from the first fully-populated run
        if task_order is None and d.get("task_names"):
            task_order = list(d["task_names"])
        Ns.add(N)
        Ps.add(P)
        for task, a in per_task.items():
            acc[task][P][N].append(float(a))
        n_ok += 1
    if task_order is None:
        task_order = sorted(acc.keys())
    # include any tasks not in the reference name list, appended alphabetically
    extra = sorted(set(acc.keys()) - set(task_order))
    task_order = [t for t in task_order if t in acc] + extra
    return acc, sorted(Ns), sorted(Ps), task_order, n_ok, len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/M64_lr5e4_NPsweep")
    ap.add_argument(
        "--out",
        default=None,
        help="output png (default: <data-dir>/acc_vs_N_perTask.png)",
    )
    ap.add_argument("--ncols", type=int, default=10)
    a = ap.parse_args()

    acc, Ns, Ps, tasks, n_ok, n_all = load_runs(a.data_dir)
    if not tasks:
        print(f"No usable metadata under {a.data_dir}")
        return
    out = a.out or os.path.join(a.data_dir, "acc_vs_N_perTask.png")
    print(f"{n_ok}/{n_all} runs loaded | {len(tasks)} tasks | N={Ns} | P={Ps}")

    # sequential color per (ordinal) P: low P light, high P dark
    cmap = matplotlib.colormaps["viridis"]
    pcolor = {P: cmap(i / max(1, len(Ps) - 1)) for i, P in enumerate(Ps)}
    xpos = {
        n: i for i, n in enumerate(Ns)
    }  # even spacing so 8->16 gap doesn't dominate

    ncols = a.ncols
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.0 * ncols, 1.7 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_visible(False)
    for k, task in enumerate(tasks):
        ax = axes[k]
        ax.set_visible(True)
        for P in Ps:
            xs, ys, lo, hi = [], [], [], []
            for n in Ns:
                vals = acc[task][P].get(n)
                if not vals:
                    continue
                xs.append(xpos[n])
                ys.append(np.mean(vals))
                lo.append(np.min(vals))
                hi.append(np.max(vals))
            if not xs:
                continue
            xs = np.array(xs)
            ax.fill_between(xs, lo, hi, color=pcolor[P], alpha=0.12, linewidth=0)
            ax.plot(xs, ys, color=pcolor[P], lw=1.3, marker="o", ms=2.5)
        ax.set_title(task, fontsize=6.5, pad=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(Ns, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5, length=2)
        ax.tick_params(axis="x", length=2)
        ax.grid(True, lw=0.3, alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # shared legend (P is ordinal -> ordered swatches)
    handles = [
        Line2D([0], [0], color=pcolor[P], lw=2, marker="o", ms=4, label=f"P={P}")
        for P in Ps
    ]
    fig.legend(
        handles=handles,
        title="individual params P",
        ncol=len(Ps),
        loc="lower center",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Per-task test accuracy vs N (nonlinear units) — seeds averaged, band = seed min–max",
        fontsize=12,
        y=0.997,
    )
    fig.supxlabel("N (nonlinear units)", y=0.028, fontsize=10)
    fig.supylabel("best-epoch test accuracy", fontsize=10)
    fig.tight_layout(rect=[0.012, 0.045, 1, 0.985])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
