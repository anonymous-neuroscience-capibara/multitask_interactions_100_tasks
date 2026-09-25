import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader
from multi_task_training.rnn_model import PLRNN, train_multitask, get_latent_states

from multi_task_training.utils import Params, build_tasks, save_model  # noqa: E402
from tasks.dataset import MultiTaskDataset, collate_fn  # noqa: E402

# Ensure project root is on path when running as a script
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(repo_root)
for path in (project_root, repo_root):
    if path not in sys.path:
        sys.path.append(path)

logging.basicConfig(
    level=logging.INFO,  # Show INFO+
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("training.log"),  # File too
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

duration_params = {
    "context": (5, 10),  # Shorter
    "stimulus": (10, 15),  # Shorter, less variable
    "delay": (10, 15),  # Shorter delays
    "response": (5, 10),
}


def run_single_combination(args):
    """Train one (task, M, P) combination. Designed to be called via Pool.map."""
    task_names, M, L, results_dir, n_samples, epochs, verbose, seed = args

    # Normalise: accept both "ReactPro" and ["ReactPro", "ReactAnti"]
    if isinstance(task_names, str):
        task_names = [task_names]
    else:
        task_names = list(task_names)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = Params(
        task_names=task_names,
        duration_params=duration_params,
        model_params={"M": M, "L": L, "N": 5},
        trials_per_task=n_samples,
        n_test_trials=3000,
        epochs=epochs,
    )

    tasks = build_tasks(params.task_names, params.duration_params)

    label = "+".join(task_names)
    logger.info(f"Training {label} with M={M}, P={L}")
    train_dataset = MultiTaskDataset(tasks, n_trials=params.trials_per_task)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    test_dataset = MultiTaskDataset(tasks, n_trials=params.n_test_trials, fixed=True)
    test_loader = DataLoader(
        test_dataset, batch_size=5000, shuffle=False, collate_fn=collate_fn
    )

    inputs, *_ = next(iter(train_loader))
    input_dim = inputs.shape[-1]

    model = PLRNN(M, L, params.model_params["N"], input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # Match run_experiments.py: 10° for NoiseCleaner, default (20°) for others
    task_thresholds = {}
    for i, name in enumerate(task_names):
        if name == "NoiseCleaner":
            task_thresholds[i] = 10 * np.pi / 180  # 10 degrees
        elif name in ("ArithMultiply", "ArithAdd"):
            task_thresholds[i] = 0.05
        # Other tasks use the default THRESHOLD (20°) in train_multitask

    (
        loss_history,
        train_task_accuracies,
        test_loss_history,
        test_task_accuracies,
        test_task_losses,
        best_test_accuracies,
        best_test_accuracy,
        best_loss,
        best_epoch,
        outputs,
    ) = train_multitask(
        model,
        train_loader,
        optimizer,
        device,
        tau=0.01,
        M_reg=int(model.M / 2),
        num_epochs=params.epochs,
        task_thresholds=task_thresholds,
        test_loader=test_loader,
        early_stopping_patience=200,
        verbose=verbose,
    )

    logger.info(f"Evaluating {label} M={M} P={L}...")

    latent_states, trial_info = get_latent_states(
        model, test_loader, device, params.task_names
    )

    # Save to a unique folder per combination
    run_dir = os.path.join(results_dir, f"seed_{seed}", f"M{M}_P{L}_N{n_samples}")
    os.makedirs(run_dir, exist_ok=True)

    save_model(model, optimizer, os.path.join(run_dir, "model.pt"))
    logger.info(f"Saved model to {run_dir}/model.pt")

    filepath = os.path.join(run_dir, "results.npz")
    np.savez(
        filepath,
        latent_states=latent_states,
        trial_info=trial_info,
        task_names=task_names,
        M=M,
        P=L,
        loss_history=loss_history,
        train_task_accuracies=train_task_accuracies,
        test_loss_history=test_loss_history,
        test_task_accuracies=test_task_accuracies,
        test_task_losses=test_task_losses,
        best_test_accuracies=best_test_accuracies,
        best_test_accuracy=best_test_accuracy,
        best_loss=best_loss,
        best_epoch=best_epoch,
        outputs=outputs,
    )
    logger.info(f"Saved {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run individual task training")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        help="Task names to train (default: all tasks)",
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        default=[64, 128, 256],
        help="M values (hidden units)",
    )
    parser.add_argument(
        "--P",
        nargs="+",
        type=int,
        default=[0, 1, 2, 4],
        help="P (L) values (nonlinear units)",
    )
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=[50, 200],
        help="Number of training/test trials per task",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/individual_training/all/20260409_prova_early_stopping_acc_based",
        help="Subdirectory under results/ for output files",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print training progress every 5 epochs",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Random seeds for multiple runs",
    )
    args = parser.parse_args()

    combinations = [
        (args.tasks, M, P, args.results_dir, n, args.epochs, args.verbose, seed)
        for M in args.M
        for P in args.P
        for n in args.n_samples
        for seed in args.seeds
    ]

    logger.info(f"Total combinations: {len(combinations)}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"M values: {args.M}")
    logger.info(f"P values: {args.P}")
    logger.info(f"Samples per task: {args.n_samples}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Running with {args.num_processes} processes")

    with Pool(processes=args.num_processes) as pool:
        pool.map(run_single_combination, combinations)

    logger.info("All experiments completed!")
