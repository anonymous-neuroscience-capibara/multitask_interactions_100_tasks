"""
Generate per-trial latent states, bitcodes, and phase annotations for all tasks
in trained hierarchical PLRNN experiments.
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project packages importable
sys.path.insert(0, str(Path(__file__).parent))

from hierachical_model_task.model import HierarchicalPLRNN
from hierachical_model_task.utils import HiearchicalModelConfig
from hierachical_model_task.run_experiments import (
    TASKS_MAP,
    _get_device,
    _handle_defaults,
)
from tasks.dataset import CognitiveTask


BASE_INPUT_DIM = CognitiveTask.BASE_INPUT_DIM


def pad_inputs(inputs: torch.Tensor) -> torch.Tensor:
    """Pad inputs to BASE_INPUT_DIM if necessary (mirrors CognitiveTask.__call__)."""
    if inputs.shape[1] < BASE_INPUT_DIM:
        padding = torch.zeros(inputs.shape[0], BASE_INPUT_DIM - inputs.shape[1])
        inputs = torch.cat([inputs, padding], dim=1)
    return inputs


def generate_trials_for_task(task, n_trials):
    """Generate n_trials from a task using generate_trial_with_phases.

    Returns padded and stacked arrays:
        inputs:  (N, T_max, BASE_INPUT_DIM)
        targets: list of tensors (variable shape per task)
        masks:   (N, T_max)
        phases:  (N, T_max)  -- padded with -1
    """
    all_inputs = []
    all_masks = []
    all_phases = []

    for _ in range(n_trials):
        inputs, targets, mask, phases = task.generate_trial_with_phases()
        inputs = pad_inputs(inputs)
        all_inputs.append(inputs)
        all_masks.append(mask)
        all_phases.append(phases)

    # Find max T
    max_T = max(inp.shape[0] for inp in all_inputs)

    # Pad and stack
    inputs_padded = torch.zeros(n_trials, max_T, BASE_INPUT_DIM)
    masks_padded = torch.zeros(n_trials, max_T)
    phases_padded = np.full((n_trials, max_T), -1, dtype=np.int64)

    for i in range(n_trials):
        T = all_inputs[i].shape[0]
        inputs_padded[i, :T] = all_inputs[i]
        masks_padded[i, :T] = all_masks[i]
        phases_padded[i, :T] = all_phases[i]

    return inputs_padded, masks_padded, phases_padded


def extract_latent_states(model, inputs, task_idx, device):
    """Run forward pass manually to extract latent states z at every timestep.

    Args:
        model: HierarchicalPLRNN
        inputs: (N, T, input_dim) tensor
        task_idx: integer task index (same for all trials in this batch)
        device: torch device

    Returns:
        z_all: (N, T, M) numpy array of latent states
    """
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        N, T, _ = inputs.shape

        # Create task_ids tensor: all same task
        task_ids = torch.full((N,), task_idx, dtype=torch.long, device=device)

        A, W, h, C, D = model.hierarchisation_scheme.get_parameters(task_ids)
        z = torch.zeros(N, model.M, device=device)
        z_all = torch.empty(N, T, model.M, device=device)

        for t in range(T):
            if model.L > 0:
                z_non_latent = z[:, : -model.L]
                z_latent = z[:, -model.L :]
                z_latent_scaled = A * z_latent
                z_latent_act = F.relu(z_latent)
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                z_update = torch.cat(
                    [torch.zeros_like(z_non_latent), z_latent_scaled], dim=1
                )
            else:
                z_combined = z
                z_update = torch.zeros_like(z)

            C_output = torch.einsum("bij,bj->bi", C, inputs[:, t])
            W_output = torch.einsum("bij,bj->bi", W, z_combined)
            z = z_update + W_output + C_output + h
            z_all[:, t] = z

    return z_all.cpu().numpy()


def compute_bitcodes(z_all, L):
    """Compute integer bitcodes from last L dims of latent states.

    Args:
        z_all: (N, T, M) numpy array
        L: number of latent (bitcode) units

    Returns:
        bitcodes: (N, T) numpy array of integers
    """
    if L <= 0:
        return np.zeros(z_all.shape[:2], dtype=np.int64)

    # Last L dims -> binary -> integer
    latent_bits = (z_all[:, :, -L:] > 0).astype(np.int64)  # (N, T, L)
    powers = (1 << np.arange(L)).astype(np.int64)  # (L,)
    bitcodes = (latent_bits * powers[None, None, :]).sum(axis=-1)  # (N, T)
    return bitcodes


def process_experiment(exp_dir, args, n_trials):
    """Process a single experiment directory."""
    exp_dir = Path(exp_dir)

    # Load metadata
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  Skipping {exp_dir}: no metadata.json")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    selected_task_names = metadata["task_names"]
    tasks = [TASKS_MAP[name] for name in selected_task_names]

    # Build model
    model_args = HiearchicalModelConfig(tasks=tasks)
    model_args.num_individual_params = metadata["num_individual_params"]
    model_args.nonlinear_units = metadata["nonlinear_units"]
    model_args.hidden_size = metadata["hidden_size"]
    model_args.hierarchisation = args.hierarchisation
    model_args = _get_device(model_args)
    model_args = _handle_defaults(model_args)

    from tasks.dataset import HierarchicalTasksDataset

    model = HierarchicalPLRNN(model_args, HierarchicalTasksDataset)

    model_path = exp_dir / "model.pt"
    if not model_path.exists():
        print(f"  Skipping {exp_dir}: no model.pt")
        return

    checkpoint = torch.load(model_path, map_location=model_args.device)

    # Legacy compatibility: older CD/AW checkpoints hierarchised `h` via a
    # `p2h` projection. The current schemes use a shared `h_shared` parameter
    # instead. When we detect a legacy checkpoint, register `p2h` on the model,
    # fill in a placeholder `h_shared`, and patch `get_parameters` to compute
    # `h = p_vector @ p2h` at runtime.
    legacy_p2h = checkpoint.pop("p2h", None)
    if legacy_p2h is not None and "h_shared" not in checkpoint:
        model.p2h = nn.Parameter(torch.zeros_like(legacy_p2h), requires_grad=False)
        checkpoint["p2h"] = legacy_p2h  # load into the newly registered param
        # Placeholder h_shared (will be ignored by the patched get_parameters).
        if hasattr(model, "h_shared"):
            checkpoint["h_shared"] = torch.zeros_like(model.h_shared.data)

        scheme = model.hierarchisation_scheme
        original_get_parameters = scheme.get_parameters

        def get_parameters_legacy(subject):
            A, W, _h, C, D = original_get_parameters(subject)
            p = model.p_vector[subject]
            h = p @ model.p2h
            return A, W, h, C, D

        scheme.get_parameters = get_parameters_legacy
        print("  [legacy] checkpoint uses p2h; computing h = p_vector @ p2h")

    model.load_state_dict(checkpoint)
    model.to(model_args.device)
    model.eval()

    L = model.L
    M = model.M

    # Output directory
    out_dir = exp_dir / "latent_and_bitcodes_multitrial"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_idx, task_name in enumerate(selected_task_names):
        task = TASKS_MAP[task_name]

        # Check that the task has generate_trial_with_phases implemented
        try:
            # Generate trials
            inputs_padded, masks_padded, phases_padded = generate_trials_for_task(
                task, n_trials
            )
        except NotImplementedError:
            print(
                f"    Skipping {task_name}: generate_trial_with_phases not implemented"
            )
            continue

        # Extract latent states
        z_all = extract_latent_states(
            model, inputs_padded, task_idx, model_args.device
        )  # (N, T_max, M)

        # Compute bitcodes
        bitcodes = compute_bitcodes(z_all, L)  # (N, T_max)

        # Save
        np.savez(
            out_dir / f"{task_name}.npz",
            latent_states=z_all,
            bitcodes=bitcodes,
            phases=phases_padded,
            masks=masks_padded.numpy(),
            inputs=inputs_padded.numpy(),
            phase_names=np.array(task.PHASE_NAMES),
        )
        print(
            f"    Saved {task_name}: {n_trials} trials, T_max={inputs_padded.shape[1]}, M={M}, L={L}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-trial latent states, bitcodes, and phase annotations"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials per task"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\garcias\Downloads\20260327\20260327",
        help="Root data directory containing seed_* subdirectories",
    )
    parser.add_argument(
        "--hierarchisation",
        type=str,
        default="all",
        help="Hierarchisation mode (all, AW, CD)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="List of seed values to process",
    )
    parser.add_argument(
        "--exp-filter",
        type=str,
        default="p*_n*_h*_s*",
        help="Glob pattern for experiment dirs under each seed",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for seed in args.seeds:
        seed_dir = data_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Seed dir not found: {seed_dir}")
            continue

        # Find matching experiment directories
        exp_pattern = str(seed_dir / args.exp_filter)
        exp_dirs = sorted(glob(exp_pattern))

        if not exp_dirs:
            print(
                f"No experiments found for seed {seed} with pattern {args.exp_filter}"
            )
            continue

        for exp_dir in exp_dirs:
            print(f"Processing {exp_dir} ...")
            process_experiment(exp_dir, args, args.n_trials)

    print("Done.")


if __name__ == "__main__":
    main()
