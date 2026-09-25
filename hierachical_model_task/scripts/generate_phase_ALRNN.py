"""
Generate per-trial latent states, bitcodes, and phase annotations for trained
multi-task PLRNN experiments (run_task_indv.py output).
"""

import argparse
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make project packages importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_task_training.rnn_model import PLRNN
from multi_task_training.utils import build_tasks
from tasks.dataset import CognitiveTask

BASE_INPUT_DIM = CognitiveTask.BASE_INPUT_DIM
DURATION_PARAMS = {  # match run_task_indv.py
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}


def pad_inputs(inputs):
    if inputs.shape[1] < BASE_INPUT_DIM:
        pad = torch.zeros(inputs.shape[0], BASE_INPUT_DIM - inputs.shape[1])
        inputs = torch.cat([inputs, pad], dim=1)
    return inputs


def generate_trials_for_task(task, task_idx, n_tasks, n_trials):
    all_inputs, all_masks, all_phases = [], [], []
    for _ in range(n_trials):
        inputs, _targets, mask, phases = task.generate_trial_with_phases()
        inputs = pad_inputs(inputs)  # (T, BASE_INPUT_DIM)
        # Append one-hot task identity (matches MultiTaskDataset)
        task_id = torch.zeros(inputs.shape[0], n_tasks)
        task_id[:, task_idx] = 1
        inputs = torch.cat([inputs, task_id], dim=1)  # (T, BASE_INPUT_DIM + n_tasks)
        all_inputs.append(inputs)
        all_masks.append(mask)
        all_phases.append(phases)

    full_input_dim = all_inputs[0].shape[1]
    max_T = max(inp.shape[0] for inp in all_inputs)
    inputs_padded = torch.zeros(n_trials, max_T, full_input_dim)
    masks_padded = torch.zeros(n_trials, max_T)
    phases_padded = np.full((n_trials, max_T), -1, dtype=np.int64)
    for i in range(n_trials):
        T = all_inputs[i].shape[0]
        inputs_padded[i, :T] = all_inputs[i]
        masks_padded[i, :T] = all_masks[i]
        phases_padded[i, :T] = all_phases[i]
    return inputs_padded, masks_padded, phases_padded


def extract_latent_states(model, inputs, device):
    """PLRNN forward pass, returning all latent states (no task-specific params)."""
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        N, T, _ = inputs.shape
        z = torch.zeros(N, model.M, device=device)
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


def compute_bitcodes(z_all, L):
    if L <= 0:
        return np.zeros(z_all.shape[:2], dtype=np.int64)
    bits = (z_all[:, :, -L:] > 0).astype(np.int64)
    powers = (1 << np.arange(L)).astype(np.int64)
    return (bits * powers[None, None, :]).sum(axis=-1)


def process_experiment(exp_dir, n_trials, device):
    exp_dir = Path(exp_dir)
    m = re.match(r"M(\d+)_P(\d+)_N(\d+)$", exp_dir.name)
    if not m:
        print(f"  Skipping {exp_dir}: name doesn't match M*_P*_N*")
        return

    results_path = exp_dir / "results.npz"
    model_path = exp_dir / "model.pt"
    if not (results_path.exists() and model_path.exists()):
        print(f"  Skipping {exp_dir}: missing results.npz or model.pt")
        return

    data = np.load(results_path, allow_pickle=True)
    task_names = list(data["task_names"])
    tasks = build_tasks(task_names, DURATION_PARAMS)

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["model_config"]
    state = ckpt["model_state_dict"]
    model = PLRNN(M=cfg["M"], L=cfg["L"], N=cfg["N"], input_dim=cfg["input_dim"]).to(
        device
    )
    model.load_state_dict(state)
    model.eval()

    out_dir = exp_dir / "latent_and_bitcodes_multitrial"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = len(task_names)
    for task_idx, (task_name, task) in enumerate(zip(task_names, tasks)):
        try:
            inputs_padded, masks_padded, phases_padded = generate_trials_for_task(
                task, task_idx, n_tasks, n_trials
            )
        except NotImplementedError:
            print(
                f"    Skipping {task_name}: generate_trial_with_phases not implemented"
            )
            continue

        z_all = extract_latent_states(model, inputs_padded, device)
        bitcodes = compute_bitcodes(z_all, cfg["L"])

        np.savez(
            out_dir / f"{task_name}.npz",
            latent_states=z_all,
            bitcodes=bitcodes,
            phases=phases_padded,
            masks=masks_padded.numpy(),
            inputs=inputs_padded.numpy(),
            phase_names=np.array(getattr(task, "PHASE_NAMES", [])),
        )
        print(
            f"    Saved {task_name}: {n_trials} trials, "
            f"T_max={inputs_padded.shape[1]}, M={cfg['M']}, L={cfg['L']}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument(
        "--data-dir",
        default=r"C:\Users\garcias\Documents\Projects\Linear_Nonlinear_Memory\multi_task_training\scripts\results\individual_training\all\20260409_prova_early_stopping_acc_based",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[2])
    p.add_argument(
        "--exp-filter",
        default="M64_P2_N*",
        help="Glob under each seed (default targets the seed_2/M64/P2 runs)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    for seed in args.seeds:
        seed_dir = data_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Seed dir not found: {seed_dir}")
            continue
        for exp_dir in sorted(glob(str(seed_dir / args.exp_filter))):
            print(f"Processing {exp_dir} ...")
            process_experiment(exp_dir, args.n_trials, device)
    print("Done.")


if __name__ == "__main__":
    main()
