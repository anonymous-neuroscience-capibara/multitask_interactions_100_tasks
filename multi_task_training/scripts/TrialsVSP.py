import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

# Ensure project root is on path when running as a script
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tasks.dataset import MultiTaskDataset, collate_fn
from multi_task_training.rnn_model import PLRNN, train_multitask
from multi_task_training.utils import (
    Params,
    build_tasks,
)

logging.basicConfig(
    level=logging.INFO,  # Show INFO+
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("training.log"),  # File too
    ],
)
logger = logging.getLogger(__name__)

duration_params = {
    "context": (5, 10),  # Shorter
    "stimulus": (10, 15),  # Shorter, less variable
    "delay": (10, 15),  # Shorter delays
    "response": (5, 10),
}

task_names = [
    "DelayPro",
    "DelayAnti",
    "ReactPro",
    "ReactAnti",
    "CatPro",
    "CatAnti",
    "Match2Sample",
    "NonMatch2Sample",
    "CtxIntMod1",
    "CtxIntMod2",
    "GoNogo",
]


def save_training_history_locked(
    params,
    tasks=None,
    duration_params=None,
    loss_history=None,
    task_accuracies=None,
    epochs_to_90=None,
    best_test_accuracies=None,
    best_test_accuracy=None,
    best_loss=None,
    best_epoch=None,
    results_dir="results",
):
    """Save training history (multi-task run) with file locking to avoid concurrent write conflicts.

    This wrapper accepts the keyword arguments used by the multi-task script variant and
    stores them into a consolidated NPZ per P value. It intentionally does not call the
    older `save_training_history` helper (which expects a pair-wise transfer-learning
    signature); instead it creates/updates a consolidated file keyed by this run's
    (trials_per_task, P) to avoid signature mismatch errors.
    """
    lock_path = os.path.join(
        results_dir, f"P={params.model_params['L']}_training_history.npz.lock"
    )
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(
        results_dir, f"P={params.model_params['L']}_training_history.npz"
    )

    with FileLock(lock_path, timeout=60):
        # Load existing data if present
        data = {}
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                data = dict(np.load(filepath, allow_pickle=True))
        except Exception:
            data = {}

        # Build a stable key for this multi-task run: use trials_per_task (or num_ex) and P
        trials = getattr(params, "trials_per_task", None) or getattr(
            params, "num_ex", None
        )
        key = f"trials={trials}_P={params.model_params['L']}"

        # Convert tasks (task objects) to readable names where possible
        task_names = None
        if tasks is not None:
            try:
                task_names = [
                    getattr(t, "name", None)
                    or getattr(t, "mode", None)
                    or t.__class__.__name__
                    for t in tasks
                ]
            except Exception:
                task_names = None

        data[key] = {
            "task_names": task_names,
            "duration_params": duration_params,
            "loss_history": loss_history,
            "task_accuracies": task_accuracies,
            "epochs_to_90": epochs_to_90,
            "best_test_accuracies": best_test_accuracies,
            "best_test_accuracy": best_test_accuracy,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "model_params": getattr(params, "model_params", None),
            "trials_per_task": trials,
        }

        # Save updated data
        np.savez(filepath, **data)


def run_single_task_pair(task1_info):
    """Wrapper function for parallel processing of task1 x P combinations."""
    num_ex, non_linear_units = task1_info

    params = Params(
        tasks=task_names,
        duration_params=duration_params,
        model_params={
            "M": 64,
            "L": non_linear_units,  # P, nonlinear units
            "N": 3,  # Output dimension (fixation, cos, sin)
        },
        trials_per_task=num_ex,
        n_test_trials=150,
        epochs=200,
        num_ex=num_ex,
    )
    multi_task(params)


def epochs_to_threshold_dict(acc_dict, threshold=0.9):
    """
    acc_dict: mapping -> sequence (list/np.array) of accuracies over epochs
    returns: mapping -> first epoch index (0-based) where accuracy >= threshold, or None
    """
    results = {}
    for k, seq in acc_dict.items():
        if seq is None:
            results[k] = None
            continue
        arr = np.asarray(seq)
        hits = np.where(arr >= threshold)[0]
        results[k] = int(hits[0]) if hits.size > 0 else None
    return results


def multi_task(params):

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = build_tasks(params.tasks, params.duration_params)

    logger.info(
        f"Training on tasks with P={params.model_params['L']} and num_ex={params.trials_per_task}, from scratch..."
    )
    train_dataset = MultiTaskDataset(
        tasks, n_trials=params.trials_per_task * len(tasks)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    test_dataset = MultiTaskDataset(tasks, n_trials=params.n_test_trials * len(tasks))
    test_loader = DataLoader(
        test_dataset, batch_size=5000, shuffle=False, collate_fn=collate_fn
    )

    inputs, targets, masks, task_ids = next(iter(train_loader))
    input_dim = inputs.shape[-1]  # stimulus (5) + task identity (n_tasks)
    # Create model
    model = PLRNN(
        params.model_params["M"],
        params.model_params["L"],
        params.model_params["N"],
        input_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    (
        loss_history,
        task_accuracies,
        _,
        _,
        best_test_accuracies,
        best_test_accuracy,
        best_loss,
        best_epoch,
    ) = train_multitask(
        model,
        train_loader,
        optimizer,
        device,
        tau=0.01,
        M_reg=int(model.M / 2),
        num_epochs=params.epochs,
        test_loader=test_loader,
    )

    epochs_to_90 = epochs_to_threshold_dict(task_accuracies, threshold=0.9)

    # If task_accuracies is a dict of per-task accuracy lists, use the dict-aware helper.

    save_training_history_locked(
        params=params,
        tasks=tasks,
        duration_params=params.duration_params,
        loss_history=loss_history,
        task_accuracies=task_accuracies,
        epochs_to_90=epochs_to_90,
        best_test_accuracies=best_test_accuracies,
        best_test_accuracy=best_test_accuracy,
        best_loss=best_loss,
        best_epoch=best_epoch,
        results_dir="C:\\Users\\garcias\\Documents\\Projects\\Linear_Nonlinear_Memory\\multi_task_training\\results\\PvsTrials",
    )


if __name__ == "__main__":

    P = [1, 2, 3, 4, 8, 16, 32, 64]
    num_processes = 8  # Adjust based on your available GPUs/CPUs

    # Build list of all (task1, other_tasks, P) combinations
    task_combinations = []
    for num_ex in [10, 20, 30, 40, 50, 80, 100, 150, 200]:
        for non_linear_units in P:
            task_combinations.append((num_ex, non_linear_units))

    logger.info(f"Starting parallel processing with {num_processes} processes")
    logger.info(f"Total combinations to process: {len(task_combinations)}")

    # Run experiments in parallel with file locking for safe concurrent writes
    with Pool(processes=num_processes) as pool:
        pool.map(run_single_task_pair, task_combinations)

    logger.info("All experiments completed!")
