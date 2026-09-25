#!/usr/bin/env python
"""
Train one hierarchical PLRNN model per task (default: p=4, n=2, h=64).
For each task, sweeps sample size in {50, 100, 200} by default.

Output layout:
    <output_dir>/<task_name>/seed_<S>/p<P>_n<N>_h<H>_s<sample_size>/
        model.pt
        results.npz
        metadata.json
"""

import argparse
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Make project importable when invoked as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import hierachical_model_task.run_experiments as rx
from hierachical_model_task.run_experiments import (
    run_single_experiment,
    get_num_workers,
    TASKS_MAP,
    task_names as ALL_TASKS,
)


def run_one(cfg):
    """Worker entry point: train on a single task."""
    (task_name, seed, sample_size, base_output_dir,
     num_individual_params, nonlinear_units, hidden_size,
     num_epochs, early_stopping_patience,
     latent_states, hierarchisation, verbose) = cfg

    # Restrict run_single_experiment's selection to just this task.
    # The function does `task_names[:num_tasks]` against its module globals,
    # so patching here is enough; each worker call re-applies its own patch.
    rx.task_names = [task_name]

    output_dir = base_output_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the joint script's per-task threshold convention (NoiseCleaner uses 10 deg).
    task_thresholds = {0: 10 * np.pi / 180} if task_name == "NoiseCleaner" else None

    try:
        return run_single_experiment(
            num_tasks=1,
            num_individual_params=num_individual_params,
            nonlinear_units=nonlinear_units,
            hidden_size=hidden_size,
            seed=seed,
            output_dir=output_dir,
            task_thresholds=task_thresholds,
            sample_size=sample_size,
            verbose=verbose,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            latent_states=latent_states,
            per_task_sample_sizes=None,
            hierarchisation=hierarchisation,
            finetune_path=None,
            unfreeze_params=None,
            finetune_new_task=False,
            exclude_task=None,
        )
    except Exception as e:
        print(f"[FAIL] task={task_name} seed={seed} N={sample_size}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical PLRNN on each task individually (defaults: p=4, n=2)."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--num-individual-params", type=int, default=4, help="p")
    parser.add_argument("--nonlinear-units", type=int, default=2, help="n")
    parser.add_argument("--hidden-size", type=int, default=64, help="h")
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--hierarchisation", default="all",
                        choices=["all", "AW", "CD"])
    parser.add_argument("--latent-states", action="store_true",
                        help="Save latent states + bitcodes after each run.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of task names to run. Default: all 19.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output dir (default: data/20260504_individual_p4n2).")
    parser.add_argument("--num-processes", type=int, default=None,
                        help="Parallel worker count (default: auto-detect via "
                             "SLURM_CPUS_PER_TASK / OMP_NUM_THREADS / cpu_count, capped at 8).")
    args = parser.parse_args()

    selected = args.tasks if args.tasks is not None else list(ALL_TASKS)
    unknown = [t for t in selected if t not in TASKS_MAP]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Known: {list(TASKS_MAP.keys())}")

    base_output = (Path(args.output_dir) if args.output_dir
                   else Path(__file__).parent.parent / "data" / "20260504_individual_p4n2")
    base_output.mkdir(parents=True, exist_ok=True)

    configs = [
        (task, seed, N, base_output,
         args.num_individual_params, args.nonlinear_units, args.hidden_size,
         args.num_epochs, args.early_stopping_patience,
         args.latent_states, args.hierarchisation, args.verbose)
        for task in selected
        for seed in args.seeds
        for N in args.sample_sizes
    ]
    print(f"Total runs: {len(configs)} "
          f"({len(selected)} tasks x {len(args.seeds)} seeds x {len(args.sample_sizes)} sample sizes)")

    num_processes = max(1, args.num_processes if args.num_processes is not None
                        else get_num_workers())
    print(f"Running in parallel with {num_processes} worker process(es).")
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_one, configs)
    n_ok = sum(1 for r in results if r is not None)
    print(f"\nDone: {n_ok}/{len(configs)} succeeded.")


if __name__ == "__main__":
    main()
