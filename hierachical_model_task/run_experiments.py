import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, current_process
import traceback

import numpy as np
import pandas as pd
import torch

# Add parent and current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from hierachical_model_task.model import HierarchicalPLRNN
from hierachical_model_task.bptt import BPTT
from hierachical_model_task.rnn_model import get_latent_states

from torch.utils.data import DataLoader

from tasks import (
    Arithmetics,
    CategoryDecision,
    ContextIntegration,
    CopyTask,
    DelayComparison,
    DelayedMatchToSample,
    DelayedResponse,
    DurationEstimation,
    GoNogo,
    IntervalDiscrimination,
    MultiSensoryIntegration,
    NoiseCleaner,
    PerceptualDecisionMaking,
)
from tasks.dataset import HierarchicalTasksDataset, collate_fn
from hierachical_model_task.utils import HiearchicalModelConfig


# Wrapper for parallel execution
def run_experiment_wrapper(config_tuple):
    """Wrapper to unpack config tuple and run experiment."""
    (
        num_tasks,
        num_individual_params,
        nonlinear_units,
        hidden_size,
        seed,
        output_dir,
        task_thresholds,
        sample_size,
        verbose,
        num_epochs,
        early_stopping_patience,
        latent_states,
        per_task_sample_sizes,
        hierarchisation,
        finetune_path,
        unfreeze_params,
        finetune_new_task,
        exclude_task,
        batch_size,
        m_reg,
    ) = config_tuple
    try:
        return run_single_experiment(
            num_tasks,
            num_individual_params,
            nonlinear_units,
            hidden_size,
            seed,
            output_dir,
            task_thresholds,
            sample_size,
            verbose,
            num_epochs,
            early_stopping_patience,
            latent_states,
            per_task_sample_sizes,
            hierarchisation,
            finetune_path,
            unfreeze_params,
            finetune_new_task,
            exclude_task,
            batch_size,
            m_reg,
        )
    except Exception as e:
        print(
            f"\nError in experiment (seed={seed}, tasks={num_tasks}, params={num_individual_params}, nonlinear={nonlinear_units}, hidden={hidden_size}): {e}"
        )
        traceback.print_exc()
        return None


# Detect available CPU workers (cluster-aware)
def get_num_workers():
    """Detect number of workers from cluster scheduler or local CPU count."""
    # Check SLURM allocation first
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)  # Reserve 1 for main
    # Check OMP_NUM_THREADS (common in HPC)
    if "OMP_NUM_THREADS" in os.environ:
        return max(1, int(os.environ["OMP_NUM_THREADS"]) - 1)
    # Fall back to local CPU count, capped at 8
    cpu_count = os.cpu_count() or 4
    return min(cpu_count - 1, 8)  # Reserve 1 CPU for main process, cap at 8


# Worker init function for deterministic DataLoader
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_task_indices(
    num_tasks, sample_size, per_task_sample_sizes=None, selected_task_names=None
):
    """Build task_indices array for the dataset.

    Args:
        num_tasks: Total number of tasks.
        sample_size: Default number of trials per task.
        per_task_sample_sizes: Optional dict mapping task_name -> sample_size.
        selected_task_names: List of task names (required if per_task_sample_sizes is set).

    Returns:
        (task_indices, n_trials) — task_indices is None when uniform (preserves
        random-assignment behavior), or an explicit np.array otherwise.
    """
    if per_task_sample_sizes is None:
        return None, num_tasks * sample_size

    if selected_task_names is None:
        raise ValueError(
            "selected_task_names is required when per_task_sample_sizes is set"
        )

    indices = []
    for tid in range(num_tasks):
        name = selected_task_names[tid]
        n = per_task_sample_sizes.get(name, sample_size)
        indices.extend([tid] * n)
    return indices, len(indices)


duration_params = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}

TASKS_MAP = {
    "NoiseCleaner": NoiseCleaner(duration_params, noise_level=0.5),
    "DelayPro": DelayedResponse(duration_params, mode="pro"),
    "DelayAnti": DelayedResponse(duration_params, mode="anti"),
    "CatPro": CategoryDecision(duration_params, mode="pro"),
    "CatAnti": CategoryDecision(duration_params, mode="anti"),
    "Match2Sample": DelayedMatchToSample(duration_params, mode="match"),
    "NonMatch2Sample": DelayedMatchToSample(duration_params, mode="nonmatch"),
    "CtxIntMod1": ContextIntegration(duration_params, relevant_modality=1),
    "CtxIntMod2": ContextIntegration(duration_params, relevant_modality=2),
    "ArithMultiply": Arithmetics(duration_params, mode="multiply"),
    "ArithAdd": Arithmetics(duration_params, mode="avg"),
    "CopyTask": CopyTask(duration_params, seq_len=8, num_symbols=4, delay_range=0),
    "GoNogo": GoNogo(duration_params),
    "PerceptualDM": PerceptualDecisionMaking(duration_params),
    "DelayedComparison": DelayComparison(duration_params),
    "DurationPro": DurationEstimation(duration_params, mode="pro"),
    "DurationAnti": DurationEstimation(duration_params, mode="anti"),
    "IntDisc": IntervalDiscrimination(duration_params),
    "MultiSens": MultiSensoryIntegration(duration_params),
}

task_names = list(TASKS_MAP.keys())


def _get_device(args: HiearchicalModelConfig):
    args.device = "cpu"
    if args.use_gpu:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        try:
            args.device = args.device + ":" + str(args.device_id)
        except AttributeError:
            pass
    return args


def _handle_defaults(args: HiearchicalModelConfig):
    # set learning rates
    if args.individual_learning_rate is not None:
        args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    else:
        args.learning_rate = (args.learning_rate, args.learning_rate)
    # set teacher forcing alpha
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args


def load_and_analyze(experiment_dir, sample_size=50, hierarchisation="all"):
    """Load a trained model from experiment_dir and compute latent states + bitcodes."""
    experiment_dir = Path(experiment_dir)

    # Load metadata to reconstruct config
    with open(experiment_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    selected_task_names = metadata["task_names"]
    tasks = [TASKS_MAP[name] for name in selected_task_names]

    args = HiearchicalModelConfig(tasks=tasks)
    args.num_individual_params = metadata["num_individual_params"]
    args.nonlinear_units = metadata["nonlinear_units"]
    args.hidden_size = metadata["hidden_size"]
    args.hierarchisation = hierarchisation
    args = _get_device(args)
    args = _handle_defaults(args)

    # Build model and load weights
    model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
    model.load_state_dict(
        torch.load(experiment_dir / "model.pt", map_location=args.device)
    )
    model.to(args.device)

    # Create test loader
    test_dataset = HierarchicalTasksDataset(
        tasks, n_trials=len(tasks) * sample_size, fixed=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    save_latent_states_and_bitcodes(
        model, test_loader, args.device, selected_task_names, experiment_dir
    )
    print(f"Saved latent states and bitcodes to {experiment_dir}")


def save_latent_states_and_bitcodes(model, test_loader, device, task_names, output_dir):
    """Extract one trial per task, save latent states and bitcode distributions."""
    latent_states, trial_info = get_latent_states(
        model, test_loader, device, task_names
    )

    # Keep only the first trial per task
    for tid in range(len(task_names)):
        if latent_states[tid].size > 0:
            latent_states[tid] = latent_states[tid][:1]  # (1, T, M)
            trial_info[tid] = {k: v[:1] for k, v in trial_info[tid].items()}

    L = model.L
    bitcodes_dir = output_dir / "latent_and_bitcodes"
    bitcodes_dir.mkdir(parents=True, exist_ok=True)
    for tid in range(len(task_names)):
        if latent_states[tid].size == 0:
            continue
        # Single trial: (T, M)
        z_trial = latent_states[tid][0]
        T = z_trial.shape[0]

        # Compute bitcode per timestep from last L units
        if L > 0:
            bitcodes = np.array(
                [
                    "".join("1" if v > 0 else "0" for v in z_trial[t, -L:])
                    for t in range(T)
                ]
            )
        else:
            bitcodes = np.array([""] * T)

        # Save as npz: latent states (T, M) + bitcodes (T,)
        np.savez(
            bitcodes_dir / f"latent_bitcodes_{task_names[tid]}.npz",
            latent_states=z_trial,
            bitcodes=bitcodes,
            masks=trial_info[tid]["masks"][0],
            inputs=trial_info[tid]["inputs"][0],
            targets=trial_info[tid]["targets"][0],
        )

    # Build unified DataFrame: one row per (task, timestep)
    df_rows = []
    for tid in range(len(task_names)):
        if latent_states[tid].size == 0:
            continue
        z_trial = latent_states[tid][0]  # (T, M)
        inputs_trial = trial_info[tid]["inputs"][0]  # (T, D)
        T = z_trial.shape[0]
        if L > 0:
            bitcodes = np.array(
                [
                    "".join("1" if v > 0 else "0" for v in z_trial[t, -L:])
                    for t in range(T)
                ]
            )
        else:
            bitcodes = np.array([""] * T)

        for t in range(T):
            df_rows.append(
                {
                    "task_name": task_names[tid],
                    "timestep": t,
                    "bitcode": str(bitcodes[t]),
                    "latent_state": z_trial[t].tolist(),
                    "input": inputs_trial[t].tolist(),
                }
            )

    df = pd.DataFrame(df_rows)
    df.to_csv(bitcodes_dir / "latent_bitcodes_dataframe.csv", index=False)


def run_single_experiment(
    num_tasks,
    num_individual_params,
    nonlinear_units,
    hidden_size,
    seed,
    output_dir,
    task_thresholds=None,
    sample_size=50,
    verbose=False,
    num_epochs=500,
    early_stopping_patience=50,
    latent_states=False,
    per_task_sample_sizes=None,
    hierarchisation="all",
    finetune_path=None,
    unfreeze_params=None,
    finetune_new_task=False,
    exclude_task=None,
    batch_size=64,
    m_reg=None,
):
    """
    Run a single experiment with specified number of tasks and individual parameters.

    Args:
        num_tasks: Number of tasks to use (1 to len(task_names))
        num_individual_params: Dimension of individual parameter vector
        nonlinear_units: Size of nonlinear layer
        hidden_size: Size of hidden layer
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        task_thresholds: Optional dict mapping task_id -> angle threshold (radians)
        per_task_sample_sizes: Optional dict mapping task_name -> train sample_size

    Returns:
        Dictionary with results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Detect available workers and set thread count
    # If running in a Pool daemon process, must use num_workers=0
    # (daemon processes cannot spawn child processes)
    if current_process().daemon:
        num_workers = 0
        torch.set_num_threads(4)
    else:
        # On cluster without Pool, detect workers normally
        num_workers = get_num_workers()
        torch.set_num_threads(max(1, num_workers))

    # Select tasks
    if exclude_task is not None:
        selected_task_names = [n for n in task_names if n != exclude_task]
        if len(selected_task_names) == len(task_names):
            raise ValueError(
                f"--exclude-task '{exclude_task}' not found. Available: {task_names}"
            )
    else:
        selected_task_names = task_names[:num_tasks]
    tasks = [TASKS_MAP[name] for name in selected_task_names]

    print(
        f"Running experiment: tasks={num_tasks}, params={num_individual_params}, nonlinear={nonlinear_units}, hidden={hidden_size}, seed={seed}"
    )

    # Configure model
    args = HiearchicalModelConfig(tasks=tasks)
    args.num_individual_params = num_individual_params
    args.nonlinear_units = nonlinear_units
    args.hidden_size = hidden_size
    args.num_epochs = num_epochs
    args.early_stopping_patience = early_stopping_patience
    args.hierarchisation = hierarchisation
    args.batch_size = batch_size
    if m_reg is not None:
        args.M_reg = m_reg
    args = _get_device(args)
    args = _handle_defaults(args)

    # Create datasets
    train_indices, n_train = build_task_indices(
        len(tasks), sample_size, per_task_sample_sizes, selected_task_names
    )
    train_dataset = HierarchicalTasksDataset(
        tasks, n_trials=n_train, task_indices=train_indices, fixed=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    test_indices, n_test = build_task_indices(len(tasks), 200)
    test_dataset = HierarchicalTasksDataset(
        tasks,
        n_trials=n_test,
        task_indices=test_indices,
        fixed=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    # Train model
    if finetune_path is not None:
        # Load pretrained weights, freeze all, unfreeze specified params
        pretrained_state = torch.load(finetune_path, map_location=args.device)
        training_alg = BPTT(None, args)

        if finetune_new_task:
            # Extend p_vector and noise_cov to accommodate new task(s)
            old_num_subjects = pretrained_state["p_vector"].shape[0]
            new_num_subjects = args.num_subjects  # current config has more tasks
            if new_num_subjects <= old_num_subjects:
                raise ValueError(
                    f"--finetune-new-task expects more tasks than the pretrained model. "
                    f"Pretrained has {old_num_subjects}, current config has {new_num_subjects}."
                )
            num_new = new_num_subjects - old_num_subjects
            dp = pretrained_state["p_vector"].shape[1]

            # Extend p_vector: keep old rows, add random new rows
            new_p_rows = torch.empty(num_new, dp).uniform_(-1, 1)
            pretrained_state["p_vector"] = torch.cat(
                [pretrained_state["p_vector"], new_p_rows], dim=0
            )
            # Extend noise_cov similarly
            old_noise = pretrained_state["noise_cov"]
            new_noise = torch.zeros(num_new, *old_noise.shape[1:])
            pretrained_state["noise_cov"] = torch.cat([old_noise, new_noise], dim=0)

            training_alg.model.load_state_dict(pretrained_state)

            # Freeze everything
            for param in training_alg.model.parameters():
                param.requires_grad = False
            # Unfreeze p_vector (we'll mask gradients to only update new rows)
            training_alg.model.p_vector.requires_grad = True

            # Gradient hook: zero out gradients for all pretrained rows
            def _mask_old_rows(grad, n_old=old_num_subjects):
                grad[:n_old] = 0
                return grad

            training_alg.model.p_vector.register_hook(_mask_old_rows)

            # Also unfreeze any extra params requested via --unfreeze
            if unfreeze_params:
                unfrozen = set()
                for name, param in training_alg.model.named_parameters():
                    if name in unfreeze_params and name != "p_vector":
                        param.requires_grad = True
                        unfrozen.add(name)
                missing = set(unfreeze_params) - unfrozen - {"p_vector"}
                if missing:
                    all_names = [n for n, _ in training_alg.model.named_parameters()]
                    raise ValueError(
                        f"Parameters not found: {missing}. Available: {all_names}"
                    )
                if unfrozen:
                    print(f"Fine-tuning new task: also unfroze {sorted(unfrozen)}")

            print(
                f"Fine-tuning new task: extended p_vector from {old_num_subjects} to "
                f"{new_num_subjects} rows, training only new row(s)"
            )
        else:
            training_alg.model.load_state_dict(pretrained_state)
            # Freeze everything
            for param in training_alg.model.parameters():
                param.requires_grad = False
            # Unfreeze requested parameters
            if unfreeze_params:
                unfrozen = set()
                for name, param in training_alg.model.named_parameters():
                    if name in unfreeze_params:
                        param.requires_grad = True
                        unfrozen.add(name)
                missing = set(unfreeze_params) - unfrozen
                if missing:
                    all_names = [n for n, _ in training_alg.model.named_parameters()]
                    raise ValueError(
                        f"Parameters not found: {missing}. " f"Available: {all_names}"
                    )
                print(f"Fine-tuning: unfroze {sorted(unfrozen)}")
            else:
                print(
                    "Warning: --finetune specified but no --unfreeze params; all weights are frozen."
                )
        # Rebuild optimizers with only unfrozen params
        unfrozen_shared = []
        unfrozen_individual = []
        _, individual_set = (
            training_alg.model.hierarchisation_scheme.grouped_parameters()
        )
        individual_ids = {id(p) for p in individual_set}
        for param in training_alg.model.parameters():
            if param.requires_grad:
                if id(param) in individual_ids:
                    unfrozen_individual.append(param)
                else:
                    unfrozen_shared.append(param)
        from torch.optim import Adam
        from torch.optim.lr_scheduler import LambdaLR

        if unfrozen_shared:
            training_alg.shared_optimizer = Adam(
                unfrozen_shared,
                lr=args.learning_rate[0],
                weight_decay=args.weight_decay,
            )
        else:
            training_alg.shared_optimizer = Adam(
                [torch.zeros(1, requires_grad=True)], lr=0
            )
        if unfrozen_individual:
            training_alg.individual_optimizer = Adam(
                unfrozen_individual, lr=args.learning_rate[1]
            )
        else:
            training_alg.individual_optimizer = Adam(
                [torch.zeros(1, requires_grad=True)], lr=0
            )
        training_alg.shared_scheduler = LambdaLR(
            training_alg.shared_optimizer, lambda epoch: torch.tensor(0.999) ** epoch
        )
        training_alg.individual_scheduler = LambdaLR(
            training_alg.individual_optimizer,
            lambda epoch: torch.tensor(0.999) ** epoch,
        )
    else:
        training_alg = BPTT(None, args)
    (
        loss_history,
        train_task_accuracies,
        test_loss_history,
        test_task_accuracies,
        train_task_losses,
        test_task_losses,
        best_epoch,
        outputs,
    ) = training_alg.train(
        train_loader,
        test_loader,
        verbose=verbose,
        task_thresholds=task_thresholds,
        task_names=selected_task_names,
    )

    # Compute best metrics
    if test_loss_history:
        best_test_loss_idx = np.argmin(test_loss_history)
        best_test_loss = test_loss_history[best_test_loss_idx]
        # Get accuracy at the epoch with best test loss
        best_avg_test_accuracy = np.mean(
            [
                test_task_accuracies[task_id][best_test_loss_idx]
                for task_id in range(len(tasks))
            ]
        )
    else:
        best_test_loss = float("inf")
        best_avg_test_accuracy = 0.0

    # Save results
    experiment_dir = (
        output_dir
        / f"seed_{seed}"
        / f"p{num_individual_params}_n{nonlinear_units}_h{hidden_size}_s{sample_size}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = experiment_dir / "model.pt"
    torch.save(training_alg.model.state_dict(), model_path)

    # Extract latent states and compute bitcodes
    if latent_states:
        save_latent_states_and_bitcodes(
            training_alg.model,
            test_loader,
            args.device,
            selected_task_names,
            experiment_dir,
        )

    # Save training outputs
    results = {
        "num_tasks": num_tasks,
        "task_names": selected_task_names,
        "num_individual_params": num_individual_params,
        "nonlinear_units": nonlinear_units,
        "hidden_size": hidden_size,
        "seed": seed,
        "loss_history": loss_history,
        "test_loss_history": test_loss_history,
        "train_task_accuracies": train_task_accuracies,
        "test_task_accuracies": test_task_accuracies,
        "train_task_losses": train_task_losses,
        "test_task_losses": test_task_losses,
        "best_test_loss": best_test_loss,
        "best_avg_test_accuracy": best_avg_test_accuracy,
        "outputs": outputs,
        "best_epoch": best_epoch,
        "config": {
            "obs_size": args.obs_size,
            "hidden_size": args.hidden_size,
            "nonlinear_units": args.nonlinear_units,
            "M_reg": args.M_reg,
            "tau": args.tau,
            "num_epochs": args.num_epochs,
            "learning_rate": (
                args.learning_rate[0]
                if isinstance(args.learning_rate, tuple)
                else args.learning_rate
            ),
            "individual_learning_rate": (
                args.learning_rate[1]
                if isinstance(args.learning_rate, tuple)
                else args.learning_rate
            ),
            "early_stopping_patience": early_stopping_patience,
            "sample_size": sample_size,
            "per_task_sample_sizes": per_task_sample_sizes,
            "batch_size": args.batch_size,
        },
    }

    # Save as numpy
    np.savez(
        experiment_dir / "results.npz",
        loss_history=np.array(loss_history),
        test_loss_history=np.array(test_loss_history),
        train_task_accuracies=np.array(train_task_accuracies),
        test_task_accuracies=np.array(test_task_accuracies),
        train_task_losses=np.array(train_task_losses),
        test_task_losses=np.array(test_task_losses),
    )

    # Save metadata as JSON
    metadata = {
        "num_tasks": num_tasks,
        "task_names": selected_task_names,
        "num_individual_params": num_individual_params,
        "nonlinear_units": nonlinear_units,
        "hidden_size": hidden_size,
        "seed": seed,
        "best_test_loss": best_test_loss,
        "best_avg_test_accuracy": best_avg_test_accuracy,
        "config": results["config"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical PLRNN experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nonlinear-units", nargs="+", type=int, default=[2])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--individual-params", nargs="+", type=int, default=[4])
    parser.add_argument(
        "--hierarchisation",
        type=str,
        default="all",
        choices=["all", "AW", "CD"],
        help="Which params to hierarchise: all (default), AW (dynamics only), CD (I/O only)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Trials per task = num_tasks * sample_size",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-epoch training details"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop if no improvement for N epochs",
    )
    parser.add_argument(
        "--latent-states",
        action="store_true",
        help="Extract latent states, bitcodes, and save DataFrame after training",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to experiment dir to load model and compute latent states/bitcodes",
    )
    parser.add_argument(
        "--per-task-sample-sizes",
        nargs="+",
        default=None,
        help="Per-task sample sizes as TaskName:N pairs, e.g. NoiseCleaner:100 DelayPro:25",
    )
    parser.add_argument(
        "--per-task-sample-sizes-file",
        type=str,
        default=None,
        help='Path to JSON file with per-task sample sizes, e.g. {"NoiseCleaner": 100}',
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="Path to a pretrained model.pt to fine-tune from",
    )
    parser.add_argument(
        "--unfreeze",
        nargs="+",
        default=None,
        help="Parameter names to unfreeze during fine-tuning (e.g. p_vector p2W p2C). "
        "All other parameters will be frozen.",
    )
    parser.add_argument(
        "--finetune-new-task",
        action="store_true",
        help="Add new task(s) to a pretrained model. Extends p_vector with new rows "
        "and trains only the new row(s). Requires --finetune.",
    )
    parser.add_argument(
        "--exclude-task",
        type=str,
        default=None,
        help="Name of a task to exclude from training (leave-one-out). "
        "E.g. --exclude-task DelayPro",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the train/test DataLoaders",
    )
    parser.add_argument(
        "--m-reg",
        type=int,
        default=None,
        help="Regularization dimension M_reg (defaults to config value if unset)",
    )
    args = parser.parse_args()

    per_task_sample_sizes = None
    if args.per_task_sample_sizes_file is not None:
        with open(args.per_task_sample_sizes_file, "r") as f:
            per_task_sample_sizes = json.load(f)
    elif args.per_task_sample_sizes is not None:
        per_task_sample_sizes = {}
        for entry in args.per_task_sample_sizes:
            name, count = entry.split(":")
            per_task_sample_sizes[name] = int(count)

    if args.load_model is not None:
        load_and_analyze(
            args.load_model,
            sample_size=args.sample_size,
            hierarchisation=args.hierarchisation,
        )
        return

    # Output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results/experiments/20260327"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_tasks = len(task_names)

    # Per-task accuracy thresholds (task_id -> radians); None uses default THRESHOLD
    task_thresholds = {0: 10 * np.pi / 180}

    # Create list of all experiment configs
    experiment_configs = [
        (
            num_tasks,
            num_individual_params,
            nonlinear_units,
            hidden_size,
            seed,
            output_dir,
            task_thresholds,
            args.sample_size,
            args.verbose,
            args.num_epochs,
            args.early_stopping_patience,
            args.latent_states,
            per_task_sample_sizes,
            args.hierarchisation,
            args.finetune,
            args.unfreeze,
            args.finetune_new_task,
            args.exclude_task,
            args.batch_size,
            args.m_reg,
        )
        for seed in args.seeds
        for nonlinear_units in args.nonlinear_units
        for hidden_size in args.hidden_sizes
        for num_individual_params in args.individual_params
    ]

    print(f"Total experiments to run: {len(experiment_configs)}")

    num_processes = max(1, get_num_workers())

    # Run experiments in parallel
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(run_experiment_wrapper, experiment_configs)

    # Filter out None results (failed experiments)
    all_results = [r for r in results_list if r is not None]

    # Save summary of all experiments
    summary_path = output_dir / "experiment_summary_prova.json"
    summary = {
        "total_experiments": len(all_results),
        "timestamp": datetime.now().isoformat(),
        "experiments": [
            {
                "num_tasks": r["num_tasks"],
                "task_names": r["task_names"],
                "num_individual_params": r["num_individual_params"],
                "nonlinear_units": r["nonlinear_units"],
                "hidden_size": r["hidden_size"],
                "seed": r["seed"],
                "best_test_loss": float(r["best_test_loss"]),
                "best_avg_test_accuracy": float(r["best_avg_test_accuracy"]),
            }
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
