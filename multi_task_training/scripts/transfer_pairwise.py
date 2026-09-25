"""Pairwise transfer-learning experiment across all task pairs.

For each ordered pair (task1, task2):
  - Train a model from scratch on task1            -> `_pro`     fields
  - Train a model from scratch on task2 (baseline) -> `_anti`    fields
  - Take the task1 model, deepcopy, fine-tune on task2 (transfer) -> `_transfer` fields

Saves per-pair training history at:
  results/transfer/<task1>_<task2>/training_history.npz

The first-task model is trained ONCE per task1 and reused across all task2 values,
unlike the older `multi_task_training copy.py` which retrained it inside the inner loop.

Early stopping is enabled (mean test accuracy across the training task(s), patience 20).
The saved loss / accuracy arrays mirror the legacy schema (train loss + train accuracies).
"""

import os
import sys
import logging
import argparse
from copy import deepcopy
from multiprocessing import Pool

# Cap OpenMP/MKL threads BEFORE importing torch so multiprocessing Pool workers don't
# oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path when running as a script.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from multi_task_training.rnn_model import PLRNN, train_multitask
from multi_task_training.utils import build_tasks, epochs_to_threshold
from tasks.dataset import MultiTaskDataset, collate_fn

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transfer_pairwise.log"),
    ],
)
logger = logging.getLogger(__name__)


ALL_TASKS = [
    "NoiseCleaner",
    "DelayPro",
    "DelayAnti",
    "CatPro",
    "CatAnti",
    "Match2Sample",
    "NonMatch2Sample",
    "CtxIntMod1",
    "CtxIntMod2",
    "ArithMultiply",
    "ArithAdd",
    "CopyTask",
    "GoNogo",
    "PerceptualDM",
    "DelayedComparison",
    # "DualDelayMatchSample",
    "DurationPro",
    "DurationAnti",
    "IntDisc",
    "MultiSens",
]

DURATION_PARAMS = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}

# Hyperparameters (mirror the original `multi_task_training copy.py`)
M = 32
DEFAULT_L = 2  # nonlinear unit count; override per-run via --L
N = 5  # output channel count — tasks share OUTPUT_DIM=5 (see tasks/dataset.py)
TRIALS_PER_TASK = 200
N_TEST_TRIALS = 150
EPOCHS_SCRATCH = 200
EPOCHS_TRANSFER = 150
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
SEED = 0
# Early stopping
ES_PATIENCE = 50
ES_METRIC = "accuracy"


def results_base_for(L):
    return os.path.join(
        repo_root, "multi_task_training", "scripts", "results", "transfer", f"L={L}"
    )


# Backward-compatible alias (used by plot_transfer_heatmap.py as default base path).
RESULTS_BASE = results_base_for(DEFAULT_L)


def make_loaders(tasks, task_id, n_train=TRIALS_PER_TASK, n_test=N_TEST_TRIALS):
    """Build train + fixed test loaders that sample only `task_id` from `tasks`."""
    train_ds = MultiTaskDataset(
        tasks, n_trials=n_train, task_indices=[task_id] * n_train
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_ds = MultiTaskDataset(
        tasks, n_trials=n_test, task_indices=[task_id] * n_test, fixed=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=min(n_test, 1000), shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def train_one(model, train_loader, test_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    out = train_multitask(
        model,
        train_loader,
        optimizer,
        device,
        tau=0.01,
        M_reg=int(model.M / 2),
        num_epochs=num_epochs,
        test_loader=test_loader,
        verbose=False,
        early_stopping_patience=ES_PATIENCE,
        early_stopping_metric=ES_METRIC,
    )
    loss_history = out[0]
    train_task_accuracies = out[1]
    return loss_history, train_task_accuracies


def run_task1(args_tuple):
    """Worker: train first model on task1 once, then loop over all task2 != task1."""
    task1, skip_existing, task2_filter, L = args_tuple

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_base = results_base_for(L)
    task2_list = task2_filter if task2_filter else ALL_TASKS
    logger.info(
        f"[{task1}] L={L} starting (first-task scratch + "
        f"{len([t for t in task2_list if t != task1])} transfers)"
    )

    first_task = build_tasks([task1, task1], DURATION_PARAMS)
    first_train_loader, first_test_loader = make_loaders(first_task, task_id=0)
    input_dim = next(iter(first_train_loader))[0].shape[-1]

    first_model = PLRNN(M, L, N, input_dim).to(device)
    loss_hist_pro, acc_pro = train_one(
        first_model, first_train_loader, first_test_loader, EPOCHS_SCRATCH, device
    )

    for task2 in task2_list:
        if task2 == task1:
            continue

        pair_dir = os.path.join(results_base, f"{task1}_{task2}")
        out_path = os.path.join(pair_dir, "training_history.npz")
        if skip_existing and os.path.exists(out_path):
            logger.info(f"[{task1}->{task2}] L={L} exists, skipping")
            continue

        second_task = build_tasks([task1, task2], DURATION_PARAMS)
        # Both [task1, task1] and [task1, task2] have len 2 in the task-id channel,
        # so input_dim is consistent and the deep-copied first_model is weight-compatible.
        second_train_loader, second_test_loader = make_loaders(second_task, task_id=1)

        # scratch baseline on task2
        second_model = PLRNN(M, L, N, input_dim).to(device)
        loss_hist_anti, acc_anti = train_one(
            second_model,
            second_train_loader,
            second_test_loader,
            EPOCHS_SCRATCH,
            device,
        )

        # transfer task1 -> task2
        transfer_model = deepcopy(first_model)
        loss_hist_transfer, acc_transfer = train_one(
            transfer_model,
            second_train_loader,
            second_test_loader,
            EPOCHS_TRANSFER,
            device,
        )

        epochs_to_scratch = epochs_to_threshold(acc_anti[1], threshold=0.9)
        epochs_to_transfer = epochs_to_threshold(acc_transfer[1], threshold=0.9)

        os.makedirs(pair_dir, exist_ok=True)
        np.savez(
            out_path,
            task_names=np.array([task1, task2]),
            duration_params=DURATION_PARAMS,
            loss_history_pro=np.array(loss_hist_pro),
            loss_history_anti=np.array(loss_hist_anti),
            loss_history_transfer=np.array(loss_hist_transfer),
            task_accuracies_pro=acc_pro,
            task_accuracies_anti=acc_anti,
            task_accuracies_transfer=acc_transfer,
            epochs_to_scratch=epochs_to_scratch,
            epochs_to_transfer=epochs_to_transfer,
        )
        logger.info(
            f"[{task1}->{task2}] L={L} saved (scratch_epochs={epochs_to_scratch}, "
            f"transfer_epochs={epochs_to_transfer})"
        )

    logger.info(f"[{task1}] L={L} done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of pool workers (each handles one task1 at a time). "
        "Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Restrict to a subset of task1 names (default: all 19). "
        "task2 still iterates over all tasks != task1 unless --task2 is set.",
    )
    parser.add_argument(
        "--task2",
        nargs="*",
        default=None,
        help="Restrict task2 (target task) to these names. Default: iterate full ALL_TASKS.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs whose training_history.npz already exists "
        "(useful for resuming an interrupted run).",
    )
    parser.add_argument(
        "--L",
        type=int,
        nargs="+",
        default=[DEFAULT_L],
        help=f"One or more nonlinear-unit counts to run sequentially (default {DEFAULT_L}). "
        f"Example: --L 1 2  runs L=1 then L=2. Each writes to results/transfer/L=<L>/.",
    )
    args = parser.parse_args()

    task1_list = args.tasks if args.tasks else ALL_TASKS
    task2_filter = args.task2  # may be None
    for t in task1_list:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task1: {t!r}. Valid: {ALL_TASKS}")
    if task2_filter:
        for t in task2_filter:
            if t not in ALL_TASKS:
                raise ValueError(f"Unknown task2: {t!r}. Valid: {ALL_TASKS}")

    task2_count = len([t for t in (task2_filter or ALL_TASKS)])
    n_pairs_per_L = sum(
        1 for t1 in task1_list for t2 in (task2_filter or ALL_TASKS) if t1 != t2
    )
    logger.info(
        f"Pairs per L: {len(task1_list)} task1 x {task2_count} task2 = {n_pairs_per_L}    "
        f"L values: {args.L}    Total pairs: {n_pairs_per_L * len(args.L)}"
    )
    logger.info(f"Workers: {args.workers}    Skip existing: {args.skip_existing}")

    for L in args.L:
        logger.info(f"=== Starting L={L}  (output base: {results_base_for(L)}) ===")
        work = [(t, args.skip_existing, task2_filter, L) for t in task1_list]

        if args.workers <= 1:
            for w in work:
                run_task1(w)
        else:
            with Pool(processes=args.workers) as pool:
                pool.map(run_task1, work)

        logger.info(f"=== Finished L={L} ===")

    logger.info("All L values processed.")


if __name__ == "__main__":
    main()
