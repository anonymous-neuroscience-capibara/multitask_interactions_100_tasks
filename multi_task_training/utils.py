from tasks import (
    Arithmetics,
    CategoryDecision,
    ContextIntegration,
    CopyTask,
    DelayComparison,
    DelayedMatchToSample,
    DelayedResponse,
    DualDelayMatchSample,
    DurationEstimation,
    GoNogo,
    IntervalDiscrimination,
    MultiSensoryIntegration,
    NoiseCleaner,
    PerceptualDecisionMaking,
)
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
from .rnn_model import PLRNN
import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


@dataclass
class Params:
    duration_params: dict
    trials_per_task: int
    n_test_trials: int
    model_params: dict
    epochs: int
    num_ex: int | None = None
    task1: str | None = None
    other_tasks: List[str] | None = None
    task_names: List[str] | None = None


def build_tasks(task_names, duration_params):
    task_registry = {
        "NoiseCleaner": lambda: NoiseCleaner(duration_params, noise_level=0.5),
        "DelayPro": lambda: DelayedResponse(duration_params, mode="pro"),
        "DelayAnti": lambda: DelayedResponse(duration_params, mode="anti"),
        "CatPro": lambda: CategoryDecision(duration_params, mode="pro"),
        "CatAnti": lambda: CategoryDecision(duration_params, mode="anti"),
        "Match2Sample": lambda: DelayedMatchToSample(duration_params, mode="match"),
        "NonMatch2Sample": lambda: DelayedMatchToSample(
            duration_params, mode="nonmatch"
        ),
        "CtxIntMod1": lambda: ContextIntegration(duration_params, relevant_modality=1),
        "CtxIntMod2": lambda: ContextIntegration(duration_params, relevant_modality=2),
        "ArithMultiply": lambda: Arithmetics(duration_params, mode="multiply"),
        "ArithAdd": lambda: Arithmetics(duration_params, mode="avg"),
        "CopyTask": lambda: CopyTask(
            duration_params, seq_len=8, num_symbols=4, delay_range=0
        ),
        "GoNogo": lambda: GoNogo(duration_params),
        "PerceptualDM": lambda: PerceptualDecisionMaking(duration_params),
        "DelayedComparison": lambda: DelayComparison(duration_params),
        "DualDelayMatchSample": lambda: DualDelayMatchSample(duration_params),
        "DurationPro": lambda: DurationEstimation(duration_params, mode="pro"),
        "DurationAnti": lambda: DurationEstimation(duration_params, mode="anti"),
        "IntDisc": lambda: IntervalDiscrimination(duration_params),
        "MultiSens": lambda: MultiSensoryIntegration(duration_params),
    }

    return [task_registry[task_name]() for task_name in task_names]


def save_model(model, optimizer, filepath):  # loss_history, task_accuracies,
    """Save model checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            #  'loss_history': loss_history,
            # 'task_accuracies': task_accuracies,
            "model_config": {
                "M": model.M,
                "L": model.L,
                "N": model.N,
                "input_dim": model.input_dim,
            },
        },
        filepath,
    )


def load_model(filepath, model=None, optimizer=None, device="cpu"):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint["model_config"]
    model = PLRNN(cfg["M"], cfg["L"], cfg["N"], cfg["input_dim"]).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        model,
        optimizer,
    )  # , checkpoint['loss_history'], checkpoint['task_accuracies']


def save_params_info(params, test_accuracies, results_dir):
    # Convert Params to dict for YAML
    params_dict = {
        "task_names": params.task_names,
        "duration_params": params.duration_params,
        "model_params": params.model_params,
        "trials_per_task": params.trials_per_task,
        "n_test_trials": params.n_test_trials,
        "epochs": params.epochs,
        "test_accuracies": {
            params.task_names: f"{acc:.4f}" for _, acc in test_accuracies.items()
        },
    }

    # Save YAML
    with open(os.path.join(results_dir, "params.yaml"), "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False, sort_keys=False)


def save_pictures(model_anti, model_transfer, output_dir):
    A_anti = model_anti.A
    A_transfer = model_transfer.A

    print(A_transfer, A_anti)

    W_anti = model_anti.W
    W_transfer = model_transfer.W

    df = pd.DataFrame(
        W_transfer.detach().cpu().numpy() - model_anti.W.detach().cpu().numpy(),
        index=[str(i) for i in range(32)],
        columns=[str(i) for i in range(32)],
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.round(2), annot=True, fmt=".2f", cmap="Greys", cbar=False, linewidths=0.5
    )  # Minimal color, focus on numbers
    plt.title("Latent matrix, transfer minus anti")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "latent_diff.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    df.to_csv(os.path.join(output_dir, "latent_diff.csv"))

    df = pd.DataFrame(
        W_anti.detach().cpu().numpy(),
        index=["Task" + str(i) for i in range(32)],
        columns=["Task" + str(i) for i in range(32)],
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.round(2), annot=True, fmt=".2f", cmap="Greys", cbar=False, linewidths=0.5
    )  # Minimal color, focus on numbers
    plt.title("Latent matrix, anti")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "latent_anti.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    df.to_csv(os.path.join(output_dir, "latent_anti.csv"))


def epochs_to_threshold(acc_list, threshold=0.9):
    """
    Returns number of epochs (1-based) needed to reach acc > threshold.
    Returns np.inf if never reached.
    """
    for i, acc in enumerate(acc_list):
        if acc > threshold:
            return i + 1  # 1-based epoch count
    return np.inf


def save_latent_states(
    task_name: str,
    latent_states: np.ndarray,
    trial_info,
    P: int,
    results_dir: str | None = None,
):
    """Save or update latent states for a task in an NPZ file.

    Parameters
    ----------
    task_name : str
        Name of the task being saved.
    latent_states : np.ndarray
        Latent state activations.
    trial_info : any
        Trial information to store with the latent states.
    P : int
        Number of latent units (P parameter).
    """
    # Load existing data if file exists

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
    filepath = os.path.join(results_dir, f"P={P}_latent_states.npz")

    data = {}
    if os.path.exists(filepath):
        # Handle empty/corrupt files gracefully
        if os.path.getsize(filepath) > 0:
            data = dict(np.load(filepath, allow_pickle=True))
        else:
            data = {}

    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Add/update task data
    data[task_name] = {
        "latent_states": latent_states,
        "trial_info": trial_info,
        "model_L": P,
    }

    # Save updated data
    np.savez(filepath, **data)


def save_training_history(
    params: str,
    task1: str,
    task2: str,
    duration_params: dict,
    loss_history_first: np.ndarray,
    loss_history_second: np.ndarray,
    loss_history_transfer: np.ndarray,
    task_accuracies_first: dict,
    task_accuracies_second: dict,
    task_accuracies_transfer: dict,
    epochs_to_scratch: float,
    epochs_to_transfer: float,
    results_dir: str | None = None,
):
    """Save or update training history for a task pair in a consolidated NPZ file.

    Parameters
    ----------
    filepath : str
        Path to the consolidated training_history.npz file (e.g., results/training_history_P=5.npz).
    task1 : str
        Name of first task.
    task2 : str
        Name of second task.
    task_names : list
        Task names list.
    duration_params : dict
        Duration parameters dict.
    loss_history_first : np.ndarray
        Loss history for scratch training on task2.
    loss_history_second : np.ndarray
        Loss history for scratch training on task2.
    loss_history_transfer : np.ndarray
        Loss history for transfer learning.
    task_accuracies_first : dict
        Task accuracies for first task.
    task_accuracies_second : dict
        Task accuracies for scratch training on task2.
    task_accuracies_transfer : dict
        Task accuracies for transfer learning.
    epochs_to_scratch : float
        Epochs to reach 0.9 accuracy from scratch.
    epochs_to_transfer : float
        Epochs to reach 0.9 accuracy via transfer.
    """
    # Load existing data if file exists

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
    filepath = os.path.join(
        results_dir, f'P={params.model_params["L"]}_training_history.npz'
    )
    data = {}
    if os.path.exists(filepath):
        try:
            if os.path.getsize(filepath) > 0:
                data = dict(np.load(filepath, allow_pickle=True))
            else:
                data = {}
        except Exception:
            data = {}
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Create key for this task pair
    key = f"{task1}_{task2}"

    # Add/update task pair data
    data[key] = {
        "task_names": [task1, task2],
        "duration_params": duration_params,
        "loss_history_first": loss_history_first,
        "loss_history_second": loss_history_second,
        "loss_history_transfer": loss_history_transfer,
        "task_accuracies_first": task_accuracies_first,
        "task_accuracies_second": task_accuracies_second,
        "task_accuracies_transfer": task_accuracies_transfer,
        "epochs_to_scratch": epochs_to_scratch,
        "epochs_to_transfer": epochs_to_transfer,
    }

    # Save updated data
    np.savez(filepath, **data)


def plot_transfer_and_similarity(
    base_path: str,
    latent_path: str,
    training_history_path: str | None = None,
    tasks: List[str] | None = None,
    similarity_cmap: str = "Greens",
    show: bool = True,
    suptitle: str | None = None,
    suptitle_size: int = 16,
):
    """Plot transfer-efficiency and latent-state similarity side by side.

    Parameters
    ----------
    base_path : str
        Folder containing per-pair training_history.npz (task1_task2/...). Ignored if training_history_path is provided.
    latent_path : str
        Path to latent_states.npz with bitcode distributions.
    training_history_path : str | None
        Optional path to consolidated training_history file (e.g., P=1_training_history.npz).
    tasks : list[str] | None
        Optional task order; defaults to the 10 task set used in notebooks.
    similarity_cmap : str
        Matplotlib/Seaborn colormap name for the similarity heatmap.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    (fig, (ax1, ax2)), matrix, sim_matrix_3_2
        Matplotlib figure/axes plus the numeric matrices for further use.
    """

    # Lazy import to avoid circulars
    import importlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.spatial.distance import jensenshannon

    af = importlib.import_module("multi_task_training.analysis_functions")

    def compute_similarity_matrix(prob_matrix, metric: str = "js"):
        n = prob_matrix.shape[0]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if metric == "js":
                    dist = jensenshannon(prob_matrix[i], prob_matrix[j])
                elif metric == "hellinger":
                    dist = np.sqrt(1 - np.sum(np.sqrt(prob_matrix[i] * prob_matrix[j])))
                elif metric == "bhattacharyya":
                    dist = -np.log(np.sum(np.sqrt(prob_matrix[i] * prob_matrix[j])))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                sim_matrix[i, j] = sim_matrix[j, i] = dist
        return sim_matrix

    if tasks is None:
        tasks = [
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
        ]

    n = len(tasks)
    matrix = np.full((n, n), np.nan, dtype=float)
    inf_mask = np.full((n, n), False, dtype=bool)

    if training_history_path and os.path.exists(training_history_path):
        # Consolidated file mode
        th = dict(np.load(training_history_path, allow_pickle=True))
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                key = f"{task1}_{task2}"
                if key not in th:
                    continue
                entry = th[key].item() if hasattr(th[key], "item") else th[key]
                epochs_to_scratch = entry.get("epochs_to_scratch", np.nan)
                epochs_to_transfer = entry.get("epochs_to_transfer", np.nan)

                if epochs_to_scratch == np.inf:
                    # Inf scratch case - use scratch accuracy max as baseline
                    scratch_acc = (
                        entry.get("task_accuracies_second", {}).get(1, [])
                        if isinstance(entry.get("task_accuracies_second"), dict)
                        else entry.get("task_accuracies_second", [])
                    )
                    if len(scratch_acc) > 0:
                        idx_max = np.argmax(scratch_acc)
                        max_val = scratch_acc[idx_max]
                        transfer_acc = (
                            entry.get("task_accuracies_transfer", {}).get(1, [])
                            if isinstance(entry.get("task_accuracies_transfer"), dict)
                            else entry.get("task_accuracies_transfer", [])
                        )
                        n_epochs = epochs_to_threshold(transfer_acc, max_val)

                        matrix[i, j] = n_epochs / (idx_max + 1)
                        inf_mask[i, j] = True
                    else:
                        matrix[i, j] = np.nan
                        inf_mask[i, j] = True
                elif np.isfinite(epochs_to_scratch):
                    matrix[i, j] = epochs_to_transfer / epochs_to_scratch
                    inf_mask[i, j] = False
                else:
                    matrix[i, j] = np.nan
                    inf_mask[i, j] = True
    else:
        # Legacy per-pair directory mode
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                filename = f"{task1}_{task2}/training_history.npz"
                full_path = os.path.join(base_path, filename)
                if not os.path.exists(full_path):
                    continue
                data = np.load(full_path, allow_pickle=True)
                if "epochs_to_scratch" in data:
                    epochs_to_scratch = data["epochs_to_scratch"].item()
                    if epochs_to_scratch == np.inf:
                        anti_acc = data["task_accuracies_anti"].item()[1]
                        idx_max = np.argmax(anti_acc)
                        max_val = anti_acc[idx_max]
                        n_epochs = epochs_to_threshold(
                            data["task_accuracies_transfer"].item(),
                            max_val,
                        )
                        matrix[i, j] = n_epochs / (idx_max + 1)
                        inf_mask[i, j] = True
                    else:
                        epochs_to_transfer = data["epochs_to_transfer"].item()
                        matrix[i, j] = epochs_to_transfer / epochs_to_scratch
                        inf_mask[i, j] = False
                data.close()

    data = np.load(latent_path, allow_pickle=True)
    stored_keys = list(data.keys())
    # Determine model_L from any entry
    model_L = None
    for k in stored_keys:
        try:
            model_L = data[k].item().get("model_L", None)
            if model_L is not None:
                break
        except Exception:
            continue
    if model_L is None:
        raise ValueError("Could not determine model_L from latent NPZ file")

    # Prepare caches for distributions
    single_dist_cache = {}
    pair_dist_cache = {}

    def _extract_first(arr):
        try:
            return arr[0]
        except Exception:
            return arr

    def _extract_second(arr):
        try:
            return arr[1]
        except Exception:
            # Fallback: if only one is present, use it
            try:
                return arr[-1]
            except Exception:
                return arr

    # Build similarity matrix comparing row task vs row_col pair
    sim_matrix_3_2 = np.full((n, n), np.nan, dtype=float)

    for i, t_row in enumerate(tasks):
        # Load single-task distribution for row task
        if t_row in stored_keys and t_row not in single_dist_cache:
            entry = data[t_row].item()
            ls = entry["latent_states"]
            ls_use = _extract_first(ls)
            single_dist_cache[t_row] = af.bitcode_distribution(ls_use, model_L)
        for j, t_col in enumerate(tasks):
            pair_key = f"{t_row}_{t_col}"
            if pair_key in stored_keys:
                if pair_key not in pair_dist_cache:
                    entry = data[pair_key].item()
                    ls = entry["latent_states"]
                    ls_use = _extract_second(ls)
                    pair_dist_cache[pair_key] = af.bitcode_distribution(ls_use, model_L)
                # Ensure single distribution exists
                if t_row in single_dist_cache:
                    from scipy.spatial.distance import jensenshannon

                    # Align two distributions to the same bitcode basis
                    _dists = {0: single_dist_cache[t_row], 1: pair_dist_cache[pair_key]}
                    aligned, _labels = af.align_bitcode_distributions(_dists)
                    p = aligned[0]
                    q = aligned[1]
                    jsd = jensenshannon(p, q)
                    sim_matrix_3_2[i, j] = 1 - jsd
            else:
                sim_matrix_3_2[i, j] = np.nan

    # Compute mean transfer efficiency statistics
    valid_all = matrix[~np.isnan(matrix)]
    mean_with_red = np.mean(valid_all) if len(valid_all) > 0 else np.nan

    valid_no_red = matrix[~np.isnan(matrix) & ~inf_mask]
    mean_without_red = np.mean(valid_no_red) if len(valid_no_red) > 0 else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=tasks,
        yticklabels=tasks,
        cbar_kws={"label": "Transfer / Scratch Ratio"},
        mask=np.isnan(matrix),
        linewidths=0.5,
        linecolor="gray",
        ax=ax1,
    )

    sns.heatmap(
        matrix,
        mask=~inf_mask,
        cmap=sns.color_palette(["red"]),
        vmin=0,
        vmax=1,
        cbar=False,
        annot=True,
        fmt=".2f",
        xticklabels=tasks,
        yticklabels=tasks,
        linewidths=0.5,
        linecolor="gray",
        ax=ax1,
    )

    ax1.set_title(
        f"Transfer Efficiency: Epochs(Transfer)/ Epochs(Scratch)\n"
        f"Red = Infinite Scratch (anti-max baseline)\n"
        f"Mean (all): {mean_with_red:.3f} | Mean (excl. red): {mean_without_red:.3f}",
        fontsize=20,
        pad=20,
    )
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Task")
    ax1.tick_params(axis="x", rotation=45)
    ax1.tick_params(axis="y", rotation=0)

    annot_labels = np.empty_like(sim_matrix_3_2, dtype=object)
    for i in range(sim_matrix_3_2.shape[0]):
        for j in range(sim_matrix_3_2.shape[1]):
            annot_labels[i, j] = "" if i == j else f"{sim_matrix_3_2[i, j]:.2f}"

    sim_matrix_plot = sim_matrix_3_2.copy()

    sns.heatmap(
        sim_matrix_plot,
        annot=annot_labels,
        fmt="",
        cmap=similarity_cmap,
        vmin=0,
        vmax=1,
        xticklabels=tasks,
        yticklabels=False,
        cbar_kws={"label": "Similarity"},
        mask=np.isnan(sim_matrix_plot),
        linewidths=0.5,
        linecolor="gray",
        ax=ax2,
    )

    ax2.set_title("Similarity Matrix (Bitcode Distributions)", fontsize=20, pad=20)
    ax2.set_xlabel("Task")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", left=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_size)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()
    if show:
        plt.show()

    return (fig, (ax1, ax2)), matrix, sim_matrix_3_2
