import os
import sys
import logging
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader
from filelock import FileLock

from multi_task_training.rnn_model import PLRNN, train_multitask, get_latent_states
from multi_task_training.utils import (
    build_tasks,
    Params,
    epochs_to_threshold,
    save_latent_states,
    save_training_history,
)
from tasks.dataset import collate_fn, MultiTaskDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("training.log"),
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

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

SEEDS = [1, 2]
RESULTS_BASE = os.path.join(repo_root, "multi_task_training", "results")


def save_latent_states_locked(
    task_name, latent_states, trial_info, P, results_dir="results"
):
    """Save latent states with file locking to avoid concurrent write conflicts."""
    lock_path = os.path.join(results_dir, f"P={P}_latent_states.npz.lock")
    os.makedirs(results_dir, exist_ok=True)

    with FileLock(lock_path, timeout=60):
        save_latent_states(
            task_name=task_name,
            latent_states=latent_states,
            trial_info=trial_info,
            P=P,
            results_dir=results_dir,
        )


def save_training_history_locked(
    params,
    task1,
    task2,
    duration_params,
    loss_history_first,
    loss_history_second,
    loss_history_transfer,
    task_accuracies_first,
    task_accuracies_second,
    task_accuracies_transfer,
    epochs_to_scratch,
    epochs_to_transfer,
    results_dir="results",
):
    """Save training history with file locking to avoid concurrent write conflicts."""
    lock_path = os.path.join(
        results_dir, f"P={params.model_params['L']}_training_history.npz.lock"
    )
    os.makedirs(results_dir, exist_ok=True)

    with FileLock(lock_path, timeout=60):
        save_training_history(
            params=params,
            task1=task1,
            task2=task2,
            duration_params=duration_params,
            loss_history_first=loss_history_first,
            loss_history_second=loss_history_second,
            loss_history_transfer=loss_history_transfer,
            task_accuracies_first=task_accuracies_first,
            task_accuracies_second=task_accuracies_second,
            task_accuracies_transfer=task_accuracies_transfer,
            epochs_to_scratch=epochs_to_scratch,
            epochs_to_transfer=epochs_to_transfer,
            results_dir=results_dir,
        )


def run_single_task_pair(task1_info):
    """Wrapper function for parallel processing of task1 x P x seed combinations."""
    task1, other_tasks, non_linear_units, seed = task1_info

    params = Params(
        task1=task1,
        other_tasks=other_tasks,
        duration_params=duration_params,
        model_params={
            "M": 16,
            "L": non_linear_units,  # P, nonlinear units
            "N": 3,  # Output dimension (fixation, cos, sin)
        },
        trials_per_task=200,
        n_test_trials=150,
        epochs=200,
    )
    # determine per-seed results directory
    results_dir = os.path.join(RESULTS_BASE, f"v{seed}")
    multi_task(params, seed=seed, results_dir=results_dir)


def multi_task(params, seed=0, results_dir=None):
    # Set RNG seeds per-run and ensure results directory
    torch.manual_seed(seed)
    np.random.seed(seed)
    if results_dir is None:
        results_dir = os.path.join(RESULTS_BASE, f"v{seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_task = build_tasks([params.task1, params.task1], params.duration_params)

    logger.info(f"Training on task {params.task1} from scratch...")
    first_train_dataset = MultiTaskDataset(
        first_task,
        n_trials=params.trials_per_task,
        task_indices=params.trials_per_task * [0],
    )
    first_train_loader = DataLoader(
        first_train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

    inputs, targets, masks, task_ids = next(iter(first_train_loader))
    input_dim = inputs.shape[-1]  # stimulus (5) + task identity (n_tasks)
    # Create model
    first_model = PLRNN(
        params.model_params["M"],
        params.model_params["L"],
        params.model_params["N"],
        input_dim,
    ).to(device)
    first_optimizer = torch.optim.Adam(first_model.parameters(), lr=1e-3)
    first_model.train()

    loss_history_first, task_accuracies_first, _, _ = train_multitask(
        first_model,
        first_train_loader,
        first_optimizer,
        device,
        tau=0.01,
        M_reg=int(first_model.M / 2),
        num_epochs=params.epochs,
    )

    first_test_dataset = MultiTaskDataset(first_task, n_trials=params.n_test_trials)

    first_test_loader = DataLoader(
        first_test_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn
    )

    latent_states_first, trial_info_first = get_latent_states(
        first_model, first_test_loader, device, [params.task1, params.task1]
    )

    save_latent_states_locked(
        task_name=params.task1,
        latent_states=latent_states_first,
        trial_info=trial_info_first,
        P=params.model_params["L"],
        results_dir=results_dir,
    )

    for task2 in params.other_tasks:
        second_task = build_tasks([params.task1, task2], params.duration_params)
        second_dataset = MultiTaskDataset(
            second_task,
            n_trials=params.trials_per_task,
            task_indices=params.trials_per_task * [1],
        )
        second_loader = DataLoader(
            second_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
        )

        second_model = PLRNN(
            params.model_params["M"],
            params.model_params["L"],
            params.model_params["N"],
            input_dim,
        ).to(device)
        second_optimizer = torch.optim.Adam(second_model.parameters(), lr=1e-3)
        second_model.train()
        loss_history_second, task_accuracies_second, _, _ = train_multitask(
            second_model,
            second_loader,
            second_optimizer,
            device,
            tau=0.01,
            M_reg=int(second_model.M / 2),
            num_epochs=params.epochs,
        )

        transfer_model = deepcopy(first_model)
        transfer_optimizer = torch.optim.Adam(transfer_model.parameters(), lr=1e-3)
        loss_history_transfer, task_accuracies_transfer, _, _ = train_multitask(
            transfer_model,
            second_loader,
            transfer_optimizer,
            device,
            num_epochs=params.epochs,
            tau=0.01,
            M_reg=int(transfer_model.M / 2),
        )

        second_test_dataset = MultiTaskDataset(
            second_task, n_trials=params.n_test_trials
        )
        second_test_loader = DataLoader(
            second_test_dataset,
            batch_size=1000,
            shuffle=False,
            collate_fn=collate_fn,
            **{"num_workers": 0},
        )

        epochs_to_90_second = epochs_to_threshold(
            task_accuracies_second[1], threshold=0.9
        )

        epochs_to_90_transfer = epochs_to_threshold(
            task_accuracies_transfer[1], threshold=0.9
        )

        save_training_history_locked(
            params=params,
            task1=params.task1,
            task2=task2,
            duration_params=params.duration_params,
            loss_history_first=loss_history_first,
            loss_history_second=loss_history_second,
            loss_history_transfer=loss_history_transfer,
            task_accuracies_first=task_accuracies_first,
            task_accuracies_second=task_accuracies_second,
            task_accuracies_transfer=task_accuracies_transfer,
            epochs_to_scratch=epochs_to_90_second,
            epochs_to_transfer=epochs_to_90_transfer,
            results_dir=results_dir,
        )

        latent_states_transfer, trial_info_transfer = get_latent_states(
            transfer_model, second_test_loader, device, [params.task1, task2]
        )

        save_latent_states_locked(
            task_name=f"{params.task1}_{task2}",
            latent_states=latent_states_transfer,
            trial_info=trial_info_transfer,
            P=params.model_params["L"],
            results_dir=results_dir,
        )


if __name__ == "__main__":

    P = [1, 2, 3, 4, 8, 16]
    num_processes = 8  # Adjust based on your available GPUs/CPUs

    # Build list of all (task1, other_tasks, P, seed) combinations
    task_combinations = []
    for seed in SEEDS:
        for task1 in task_names:
            other_tasks = [t for t in task_names if t != task1]
            for non_linear_units in P:
                task_combinations.append((task1, other_tasks, non_linear_units, seed))

    logger.info(f"Starting parallel processing with {num_processes} processes")
    logger.info(f"Total combinations to process: {len(task_combinations)}")

    # Run experiments in parallel with file locking for safe concurrent writes
    with Pool(processes=num_processes) as pool:
        pool.map(run_single_task_pair, task_combinations)

    logger.info("All experiments completed!")
