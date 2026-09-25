"""
Merge partial seed sweep results from parallel runs into a single file and plot.

Usage:
    python merge_sweep_results.py
    python merge_sweep_results.py --save_dir seed_sweep_results
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hierachical_model_task.run_experiments import TASKS_MAP

ALL_TASK_NAMES = list(TASKS_MAP.keys())


def main():
    parser = argparse.ArgumentParser(description="Merge partial sweep results")
    parser.add_argument("--save_dir", type=str, default="seed_sweep_results")
    cfg = parser.parse_args()
    save_dir = Path(cfg.save_dir)

    # Find all partial result files
    partial_files = sorted(save_dir.glob("results_all_seeds*.pkl"))
    if not partial_files:
        partial_files = sorted(save_dir.glob("results_intermediate*.pkl"))
    if not partial_files:
        print("No result files found!")
        return

    print(f"Found {len(partial_files)} result files:")
    for f in partial_files:
        print(f"  {f.name}")

    # Merge
    merged_before = {}
    merged_after = {}
    all_loo_seeds = set()
    all_ft_seeds = set()
    all_conditions = set()

    for f in partial_files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)

        for excl, keys_dict in data["before"].items():
            if excl not in merged_before:
                merged_before[excl] = {}
            merged_before[excl].update(keys_dict)

        for cond, excl_dict in data["after"].items():
            if cond not in merged_after:
                merged_after[cond] = {}
            for excl, keys_dict in excl_dict.items():
                if excl not in merged_after[cond]:
                    merged_after[cond][excl] = {}
                merged_after[cond][excl].update(keys_dict)

        if "loo_seeds" in data:
            all_loo_seeds.update(data["loo_seeds"])
        if "ft_seeds" in data:
            all_ft_seeds.update(data["ft_seeds"])
        if "conditions" in data:
            all_conditions.update(data["conditions"])

    print(
        f"\nMerged: {len(merged_before)} excluded tasks, "
        f"{len(merged_after)} conditions"
    )

    # Save merged
    merged_path = save_dir / "results_merged.pkl"
    with open(merged_path, "wb") as f:
        pickle.dump(
            {
                "before": merged_before,
                "after": merged_after,
                "loo_seeds": sorted(all_loo_seeds),
                "ft_seeds": sorted(all_ft_seeds),
                "conditions": sorted(all_conditions),
            },
            f,
        )
    print(f"Saved {merged_path}")

    # Plot
    conditions = {c: None for c in sorted(all_conditions)}
    plot_results(
        merged_before,
        merged_after,
        conditions,
        sorted(all_loo_seeds),
        sorted(all_ft_seeds),
        save_dir,
    )


def plot_results(
    results_before, results_after, conditions, loo_seeds, ft_seeds, save_dir
):
    cond_names = list(conditions.keys())
    n_conds = len(cond_names)

    # ---- PLOT 1: Excluded task accuracy ----
    tasks_with_data = [
        t
        for t in ALL_TASK_NAMES
        if t in results_before
        and any(t in results_after.get(c, {}) for c in cond_names)
    ]
    x = np.arange(len(tasks_with_data), dtype=float)
    width = 0.8 / n_conds

    fig1, ax1 = plt.subplots(figsize=(18, 7))
    colors = ["steelblue", "coral", "goldenrod", "seagreen", "mediumpurple"]

    for i, cond_name in enumerate(cond_names):
        cond_data = results_after.get(cond_name, {})
        means = []
        stds = []
        for excluded in tasks_with_data:
            vals = []
            for key, task_accs in cond_data.get(excluded, {}).items():
                vals.append(task_accs.get(excluded, 0.0))
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        ax1.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=cond_name,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
        )

    ax1.set_xticks(x + width * (n_conds - 1) / 2)
    ax1.set_xticklabels(tasks_with_data, rotation=45, ha="right")
    ax1.set_ylabel("Test Accuracy on Excluded Task (200 trials)")
    ax1.set_title(
        "Transfer Learning: Excluded Task Accuracy by Condition",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fig1.savefig(save_dir / "excluded_task_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_dir / 'excluded_task_accuracy.png'}")

    # ---- PLOT 2: Trained task degradation ----
    fig2, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 6), sharey=True)
    if n_conds == 1:
        axes = [axes]

    for idx, cond_name in enumerate(cond_names):
        ax = axes[idx]
        cond_data = results_after.get(cond_name, {})
        tasks_plotted = [
            t for t in ALL_TASK_NAMES if t in cond_data and t in results_before
        ]

        before_stats = []
        after_stats = []
        for excluded in tasks_plotted:
            b_vals = []
            a_vals = []
            trained_tasks = [t for t in ALL_TASK_NAMES if t != excluded]
            for key in results_before[excluded]:
                if key not in cond_data.get(excluded, {}):
                    continue
                b_vals.append(
                    np.mean(
                        [results_before[excluded][key].get(t, 0) for t in trained_tasks]
                    )
                )
                a_vals.append(
                    np.mean([cond_data[excluded][key].get(t, 0) for t in trained_tasks])
                )
            before_stats.append(
                (np.mean(b_vals) if b_vals else 0, np.std(b_vals) if b_vals else 0)
            )
            after_stats.append(
                (np.mean(a_vals) if a_vals else 0, np.std(a_vals) if a_vals else 0)
            )

        x2 = np.arange(len(tasks_plotted), dtype=float)
        w = 0.35
        b_means = [v[0] for v in before_stats]
        b_stds = [v[1] for v in before_stats]
        a_means = [v[0] for v in after_stats]
        a_stds = [v[1] for v in after_stats]

        ax.bar(
            x2 - w / 2,
            b_means,
            w,
            yerr=b_stds,
            label="Before FT",
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
        )
        ax.bar(
            x2 + w / 2,
            a_means,
            w,
            yerr=a_stds,
            label="After FT",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
        )
        ax.set_xticks(x2)
        ax.set_xticklabels(tasks_plotted, rotation=90, fontsize=8)
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        if idx == 0:
            ax.set_ylabel("Mean Accuracy on 18 Trained Tasks")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Trained-task degradation after finetuning "
        f"(LOO seeds={loo_seeds}, FT seeds={ft_seeds})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig(
        save_dir / "trained_task_degradation.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"Saved {save_dir / 'trained_task_degradation.png'}")


if __name__ == "__main__":
    main()
