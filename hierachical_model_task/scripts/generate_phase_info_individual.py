"""
Generate latent states with phase annotations for individually-trained PLRNN models.

For each (task, seed, M, P, N) combination found under
  multi_task_training/scripts/results/individual_tasks/
this script:
  1. Loads the saved model
  2. Generates fresh trials via generate_trial_with_phases()
  3. Runs the forward pass to extract latent states
  4. Saves a per-task .npz with latent_states, phases, phase_names, masks, inputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from multi_task_training.utils import build_tasks, load_model


def pad_inputs(inputs: torch.Tensor, target_dim: int) -> torch.Tensor:
    if inputs.shape[1] < target_dim:
        padding = torch.zeros(inputs.shape[0], target_dim - inputs.shape[1])
        inputs = torch.cat([inputs, padding], dim=1)
    return inputs


def generate_trials_for_task(task, n_trials, input_dim):
    all_inputs, all_masks, all_phases = [], [], []
    for _ in range(n_trials):
        inputs, targets, mask, phases = task.generate_trial_with_phases()
        inputs = pad_inputs(inputs, input_dim)
        all_inputs.append(inputs)
        all_masks.append(mask)
        all_phases.append(phases)

    max_T = max(inp.shape[0] for inp in all_inputs)
    inputs_padded = torch.zeros(n_trials, max_T, input_dim)
    masks_padded = torch.zeros(n_trials, max_T)
    phases_padded = np.full((n_trials, max_T), -1, dtype=np.int64)

    for i in range(n_trials):
        T = all_inputs[i].shape[0]
        inputs_padded[i, :T] = all_inputs[i]
        masks_padded[i, :T] = all_masks[i]
        phases_padded[i, :T] = all_phases[i]

    return inputs_padded, masks_padded, phases_padded


def extract_latent_states(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        N, T, _ = inputs.shape
        z = model.init_hidden(N).to(device)
        z_all = torch.empty(N, T, model.M, device=device)

        for t in range(T):
            if model.L > 0:
                z_non_latent = z[:, : -model.L]
                z_latent = z[:, -model.L :]
                z_latent_scaled = model.A * z_latent
                z_latent_act = F.relu(z_latent)
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                z_update = torch.cat(
                    [torch.zeros_like(z_non_latent), z_latent_scaled], dim=1
                )
            else:
                z_combined = z
                z_update = torch.zeros_like(z)

            z = (
                z_update
                + z_combined @ model.W.t()
                + inputs[:, t] @ model.C.t()
                + model.h
            )
            z_all[:, t] = z

    return z_all.cpu().numpy()


duration_params = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}


def process_task_dir(task_dir, task_name, n_trials, output_dir, device):
    model_path = task_dir / "model.pt"
    if not model_path.exists():
        print(f"  Skipping {task_dir}: no model.pt")
        return

    # Build the task object
    tasks = build_tasks([task_name], duration_params)
    task = tasks[0]

    if not hasattr(task, "generate_trial_with_phases"):
        print(f"  Skipping {task_name}: no generate_trial_with_phases")
        return

    # Load model
    model, _ = load_model(str(model_path), device=device)
    model.to(device)
    model.eval()
    input_dim = model.input_dim  # actual dim the model was trained with

    # Generate trials with phase info
    try:
        inputs_padded, masks_padded, phases_padded = generate_trials_for_task(
            task, n_trials, input_dim
        )
    except NotImplementedError:
        print(f"  Skipping {task_name}: generate_trial_with_phases not implemented")
        return

    # Extract latent states
    z_all = extract_latent_states(model, inputs_padded, device)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_name}.npz"
    np.savez(
        out_path,
        latent_states=z_all,
        phases=phases_padded,
        masks=masks_padded.numpy(),
        inputs=inputs_padded.numpy(),
        phase_names=np.array(task.PHASE_NAMES),
    )
    print(
        f"  Saved {out_path.name}: {n_trials} trials, T_max={inputs_padded.shape[1]}, M={model.M}, L={model.L}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase-annotated latent states for individual task models"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials per task",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\garcias\Documents\Projects\Linear_Nonlinear_Memory\multi_task_training\scripts\results\individual_tasks",
        help="Root directory containing seed_*/TaskName_M*_P*_N* folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\garcias\Documents\Projects\Linear_Nonlinear_Memory\phase_info_individual_tasks",
        help="Output directory for phase-annotated .npz files",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Seeds to process",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Filter to specific model tag, e.g. M64_P2_N200. If None, process all.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        seed_dir = data_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Seed dir not found: {seed_dir}")
            continue

        # Find all task directories
        if args.model_tag:
            pattern = f"*_{args.model_tag}"
        else:
            pattern = "*_M*_P*_N*"

        task_dirs = sorted(seed_dir.glob(pattern))
        if not task_dirs:
            print(f"No task dirs found in {seed_dir} with pattern {pattern}")
            continue

        for task_dir in task_dirs:
            dirname = task_dir.name
            # Parse task name: everything before _M\d+
            import re

            m = re.match(r"^(.+?)_(M\d+_P\d+_N\d+)$", dirname)
            if not m:
                print(f"  Skipping {dirname}: doesn't match expected pattern")
                continue

            task_name = m.group(1)
            model_tag = m.group(2)

            out_dir = output_base / f"seed_{seed}" / model_tag
            print(f"Processing {task_name} [{model_tag}] seed={seed}...")
            process_task_dir(task_dir, task_name, args.n_trials, out_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
