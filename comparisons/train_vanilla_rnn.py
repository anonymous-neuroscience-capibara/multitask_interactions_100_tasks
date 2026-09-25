#!/usr/bin/env python
"""Individual (per-task) training of a VANILLA (Elman) RNN on the 100-task foundation
battery, as an architectural baseline for the PLRNN runs
(multi_task_training/scripts/train_indiv_models.py and the joint M64_lr5e4_NPsweep).

Model: the textbook vanilla RNN
    z_t = phi(W z_{t-1} + C x_t + h),   y_t = D z_t
with phi = tanh by default (--nonlinearity relu isolates the PLRNN's structural
decomposition instead, since the nonlinearity is then shared). One model per task,
no task one-hot, no per-task parameters -- exactly like the flat PLRNN runs.

WHAT IS MATCHED TO train_indiv_models.py (and through it to the joint runs):
  fixed datasets        HierarchicalTasksDataset(..., fixed=True) for BOTH train and
                        test, so --sample-size really is the training-set size
  learning rate         5e-4 constant, no warmup
  weight decay          1e-4 on Adam
  loss                  hierachical_model_task.rnn_model.normalized_mse_loss only
                        (no fix_loss term)
  accuracy              the joint pipeline's compute_accuracies (36 deg threshold,
                        fixation gate, tau*sigma tolerance for the 7 continuous
                        tasks unless --legacy-accuracy)
  grad clip 10, non-finite batch guard with adaptive lr halving, eval_interval 25,
  early stopping on TEST LOSS with es_start=300 and patience=500,
  checkpoint = best-by-test-loss state.

DELIBERATE DEVIATIONS (the architecture comparison itself):
  no MAR term           the manifold-attractor regularizer is a PLRNN-specific
                        inductive bias (it shapes the A/W split). By default the
                        vanilla RNN trains with weight decay only. --mar applies the
                        analogous term ((W_ii-1)^2 + offdiag(W_i)^2 + h_i^2 over the
                        first M_reg units, tau=0.01, M_reg=M//2) if you want the
                        training objectives matched instead.
  initialisation        --init orthogonal (default): W orthogonal at gain 1, the
                        standard vanilla-RNN init; without the PLRNN's near-identity
                        A, the PLRNN-style W ~ randn*0.1/sqrt(M) (std 0.0125) would
                        start the network with almost no memory and handicap the
                        baseline. --init matched uses those PLRNN-native scales
                        anyway, for an init-controlled ablation.
  no L sweep            the vanilla RNN has no linear/nonlinear split; every unit is
                        nonlinear. The sweep axis here is --hidden-sizes (M).

Output layout (mirrors the flat-PLRNN runs, with M in place of N{L}):
    <results>/M{M}_s{s}/<task>/seed_{seed}/{metadata.json, results.npz, model.pt}
"""

import argparse
import json
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def _find_root(start):
    """Repo root = the dir holding both `tasks/` and `multi_task_training/`."""
    for cand in [start, *start.parents]:
        if (cand / "tasks").is_dir() and (cand / "multi_task_training").is_dir():
            return cand
    return None


ROOT = _find_root(Path(__file__).resolve().parent) or _find_root(Path.cwd().resolve())
if ROOT is None:
    raise SystemExit("could not locate repo root (needs tasks/ + multi_task_training/)")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader  # noqa: E402

# The JOINT pipeline's scoring + loss, so accuracy is the same function the
# PLRNN runs were scored with (36 deg threshold + fixation gate + tau*sigma).
from hierachical_model_task.rnn_model import (  # noqa: E402
    THRESHOLD,
    TAU_SIGMA,
    compute_accuracies,
    compute_loss,
    compute_per_task_loss,
    compute_scalar_vector_tol,
    normalized_mse_loss,
)
from tasks.dataset import HierarchicalTasksDataset, collate_fn  # noqa: E402
import foundation_tasks as ft  # noqa: E402


class VanillaRNN(nn.Module):
    """Elman RNN with the joint pipeline's call signature: forward(inputs, subject=None).

    z_t = phi(W z_{t-1} + C x_t + h), y_t = D z_t. Parameter names (W, C, h, D) and
    the M attribute match the PLRNN class so analysis code that touches them ports
    over directly. L is kept as an attribute (== M: every unit is nonlinear) for
    code that reads model.L.
    """

    def __init__(self, M, N, input_dim, nonlinearity="tanh", init="orthogonal"):
        super().__init__()
        self.M = M
        self.L = M
        self.N = N
        self.input_dim = input_dim
        self.nonlinearity = nonlinearity

        if init == "orthogonal":
            W = torch.empty(M, M)
            nn.init.orthogonal_(W)
        elif init == "matched":
            W = torch.randn(M, M) * 0.1 / np.sqrt(M)
        else:
            raise ValueError(f"unknown init: {init!r}")
        self.W = nn.Parameter(W)

        self.h = nn.Parameter(torch.zeros(M))
        self.C = nn.Parameter(torch.randn(M, input_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(N, M) * 0.1)

    def phi(self, x):
        return torch.tanh(x) if self.nonlinearity == "tanh" else F.relu(x)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.M)

    def forward(self, inputs, subject=None, return_latent=False):
        """
        Args:
            inputs: (batch, T, input_dim)
            subject: accepted and ignored (joint scoring passes task_ids here)
            return_latent: also return the hidden trajectory (batch, T, M)
        Returns:
            outputs: (batch, T, N)  [, latents: (batch, T, M)]
        """
        batch_size, T, _ = inputs.shape
        z = self.init_hidden(batch_size).to(inputs.device)

        traj = []
        latents = [] if return_latent else None
        for t in range(T):
            z = self.phi(z @ self.W.t() + inputs[:, t] @ self.C.t() + self.h)
            traj.append(z @ self.D.t())
            if return_latent:
                latents.append(z)

        outputs = torch.stack(traj, dim=1)
        if return_latent:
            return outputs, torch.stack(latents, dim=1)
        return outputs


def vanilla_regularization_loss(model, tau, M_reg):
    """MAR analogue for the vanilla RNN (there is no A: diag_eff = W_ii).

    Same math as multi_task_training.MAR.regularization_loss restricted to a full
    recurrent matrix: over the first M_reg units, (W_ii - 1)^2 + off-diagonal row
    power + bias^2, scaled by tau.
    """
    loss = 0.0
    for i in range(M_reg):
        loss += (model.W[i, i] - 1) ** 2
        loss += torch.sum(model.W[i, :] ** 2) - model.W[i, i] ** 2
        loss += model.h[i] ** 2
    return tau * loss


def train_one(job):
    """Train a single (task, M, sample_size, seed) combination. Pool-friendly."""
    task_name, M, sample_size, seed, cfg = job

    out = Path(cfg["results_dir"]) / f"M{M}_s{sample_size}" / task_name / f"seed_{seed}"
    if cfg["skip_existing"] and (out / "metadata.json").exists():
        return f"skip  {task_name} M={M} s={sample_size} seed={seed} (already done)"

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg["threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = ft.FOUNDATION_TASKS_MAP[task_name]
    tasks = [task]
    n_out = task.output_dim
    M_reg, tau = M // 2, cfg["tau"]

    # fixed=True on BOTH: the trials are drawn once and reused every epoch.
    train_ds = HierarchicalTasksDataset(
        tasks, n_trials=sample_size, task_indices=[0] * sample_size, fixed=True)
    test_ds = HierarchicalTasksDataset(
        tasks, n_trials=cfg["test_size"], task_indices=[0] * cfg["test_size"], fixed=True)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn, generator=g)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=collate_fn)

    input_dim = next(iter(train_loader))[0].shape[-1]
    model = VanillaRNN(M, n_out, input_dim, nonlinearity=cfg["nonlinearity"],
                       init=cfg["init"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"],
                     weight_decay=cfg["weight_decay"])

    task_loss_channels = ({0: task.loss_channels}
                          if getattr(task, "loss_channels", None) is not None else {})

    # tau*sigma tolerance for the 7 continuous-readout tasks; {} for the rest, which
    # fall back to the fixed NON_ANGULAR_THRESHOLD (same as the PLRNN runs).
    scalar_vector_tol = None if cfg["legacy_accuracy"] else (
        compute_scalar_vector_tol(tasks) or None)

    base_lr = cfg["learning_rate"]
    warm = cfg["lr_warmup_epochs"]
    lr_scale = 1.0
    grad_clip = cfg["grad_clip"]
    eval_interval = cfg["eval_interval"]

    loss_history = []
    eval_epochs, test_loss_history, test_acc_hist, test_tl_hist = [], [], [], []
    train_acc_hist, train_eval_epochs = [], []
    best_test_loss, best_state = float("inf"), None
    n_nonfinite = 0
    stopped_early = False

    for epoch in range(cfg["epochs"]):
        # effective LR = base * warmup ramp * adaptive scale (BPTT._apply_lr)
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for grp in optimizer.param_groups:
            grp["lr"] = base_lr * factor * lr_scale

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for inputs, targets, masks, task_ids in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            masks, task_ids = masks.to(device), task_ids.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, task_ids)

            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)
            masked_outputs = outputs * masks_expanded
            if targets.dim() == 2:
                targets_exp = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                masked_targets = targets_exp * masks_expanded
            else:
                masked_targets = targets * masks_expanded

            loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels)
            if cfg["mar"]:
                loss = loss + vanilla_regularization_loss(model, tau, M_reg)

            # non-finite guard: skip the batch, cool the LR down (BPTT behaviour)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                lr_scale = max(lr_scale * 0.5, 1e-4)
                optimizer.zero_grad()
                continue

            loss.backward()
            if grad_clip and grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    n_nonfinite += 1
                    lr_scale = max(lr_scale * 0.5, 1e-4)
                    optimizer.zero_grad()
                    continue
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

        # ---- test-side metrics EVERY epoch (best_epoch = exact argmin of test loss) --
        tl = compute_loss(model, test_loader, device, tau, M_reg, task_loss_channels)
        per_task_tl = compute_per_task_loss(
            model, test_loader, device, 1, task_loss_channels)
        test_acc = float(compute_accuracies(
            model, test_loader, device, 1, task_thresholds=None, tasks=tasks,
            scalar_vector_tol=scalar_vector_tol)[0])

        eval_epochs.append(epoch + 1)
        test_loss_history.append(float(tl))
        test_tl_hist.append(float(per_task_tl[0]))
        test_acc_hist.append(test_acc)

        if tl < best_test_loss:
            best_test_loss = float(tl)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # ---- train accuracy only on the interval (expensive, nothing selects on it) --
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == cfg["epochs"]:
            tr_accs = compute_accuracies(
                model, train_loader, device, 1, task_thresholds=None, tasks=tasks,
                scalar_vector_tol=scalar_vector_tol)
            train_eval_epochs.append(epoch + 1)
            train_acc_hist.append(float(tr_accs[0]))

        # early stopping in EPOCHS, only after es_start (BPTT semantics)
        if test_loss_history and epoch + 1 > cfg["es_start"]:
            epochs_since_best = (epoch + 1) - eval_epochs[
                int(np.argmin(test_loss_history))]
            if cfg["patience"] and epochs_since_best >= cfg["patience"]:
                stopped_early = True
                break

    # Saved weights = the best-by-test-loss state (train_foundation.py's convention).
    if best_state is not None:
        model.load_state_dict(best_state)
        selection = "best_test_loss"
    else:
        selection = "final"

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

    test_lh = np.asarray(test_loss_history, dtype=float)
    best_idx = int(np.argmin(test_lh)) if test_lh.size else -1

    meta = {
        "task": task_name,
        "seed": seed,
        "model": "vanilla_rnn",
        "nonlinearity": cfg["nonlinearity"],
        "init": cfg["init"],
        "mar": cfg["mar"],
        "parameterization": "flat_shared",
        "num_individual_params": 0,
        "task_onehot": False,
        "sample_size": sample_size,
        "test_size": cfg["test_size"],
        "batch_size": cfg["batch_size"],
        "hidden_size": M,
        "output_size": n_out,
        "input_dim": input_dim,
        "num_epochs": cfg["epochs"],
        "epochs_run": len(loss_history),
        "stopped_early": stopped_early,
        "checkpoint_selection": selection,
        # first epoch at which test accuracy reached 1.0 (None if never). Recorded
        # only; it does NOT stop the run.
        "epochs_to_first_perfect": next(
            (int(eval_epochs[i]) for i, v in enumerate(test_acc_hist) if v >= 1.0), None),
        "early_stopping_patience": cfg["patience"],
        "early_stopping_start": cfg["es_start"],
        "eval_interval": eval_interval,
        "learning_rate": base_lr,
        "lr_warmup_epochs": warm,
        "weight_decay": cfg["weight_decay"],
        "grad_clip": grad_clip,
        "tau": tau if cfg["mar"] else None,
        "M_reg": M_reg if cfg["mar"] else None,
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies (fixation gate on)",
        "tau_sigma_metric": not cfg["legacy_accuracy"],
        "tau_sigma": None if cfg["legacy_accuracy"] else TAU_SIGMA,
        "scalar_vector_tol": (None if not scalar_vector_tol
                              else float(scalar_vector_tol[0])
                              if 0 in scalar_vector_tol else None),
        "n_nonfinite_batches": n_nonfinite,
        "final_lr_scale": lr_scale,
        "n_shared_params": int(sum(q.numel() for q in model.parameters())),
        "best_epoch": int(eval_epochs[best_idx]) if best_idx >= 0 else None,
        "best_test_acc": float(test_acc_hist[best_idx]) if best_idx >= 0 else None,
        "best_test_loss": float(test_tl_hist[best_idx]) if best_idx >= 0 else None,
        "best_mean_test_loss": float(test_lh[best_idx]) if best_idx >= 0 else None,
        "final_train_acc": float(train_acc_hist[-1]) if train_acc_hist else None,
        "final_test_acc": float(test_acc_hist[-1]) if test_acc_hist else None,
        "timestamp": datetime.now().isoformat(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    np.savez_compressed(
        out / "results.npz",
        loss_history=np.asarray(loss_history, dtype=float),
        # test-side traces: ONE ENTRY PER EPOCH, so eval_epochs == 1..epochs_run
        eval_epochs=np.asarray(eval_epochs, dtype=int),
        test_loss_history=test_lh,
        test_task_accuracies=np.asarray(test_acc_hist, dtype=float),
        test_task_losses=np.asarray(test_tl_hist, dtype=float),
        # train accuracy is on the --eval-interval cadence, so it is SHORTER than the
        # arrays above; index it with train_eval_epochs, not with eval_epochs.
        train_task_accuracies=np.asarray(train_acc_hist, dtype=float),
        train_eval_epochs=np.asarray(train_eval_epochs, dtype=int),
    )

    return (f"done  {task_name} M={M} s={sample_size} seed={seed} | "
            f"epochs={len(loss_history)}/{cfg['epochs']}"
            f"{' (early stop)' if stopped_early else ''} "
            f"best_epoch={meta['best_epoch']} best_test_acc={meta['best_test_acc']}")


def main():
    all_names = list(ft.FOUNDATION_TASKS_MAP)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=all_names,
                   help=f"tasks, each in its OWN model (default: all {len(all_names)})")
    p.add_argument("--hidden-sizes", "-M", nargs="+", type=int, default=[64],
                   help="M sweep (default 64, the PLRNN runs' M)")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[200])
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--nonlinearity", choices=["tanh", "relu"], default="tanh",
                   help="tanh = textbook vanilla RNN (default); relu shares the "
                        "PLRNN's nonlinearity and isolates its structural split")
    p.add_argument("--init", choices=["orthogonal", "matched"], default="orthogonal",
                   help="W init: orthogonal gain 1 (standard, default) or 'matched' "
                        "PLRNN-native scales (randn*0.1/sqrt(M))")
    p.add_argument("--mar", action="store_true",
                   help="add the MAR-analogue regularizer ((W_ii-1)^2 + offdiag + "
                        "bias^2 over the first M//2 units). OFF by default: MAR is a "
                        "PLRNN-specific inductive bias.")
    p.add_argument("--test-size", type=int, default=50, help="joint runs used TEST=50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0,
                   help="LR ramp over this many epochs. 0 = OFF (matches the "
                        "individual PLRNN runs)")
    p.add_argument("--patience", type=int, default=500, help="on test loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--tau", type=float, default=0.01, help="MAR strength (with --mar)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads PER worker (num_processes*threads <= cores)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip combos whose metadata.json exists (idempotent resubmits)")
    p.add_argument("--legacy-accuracy", action="store_true",
                   help="score the 7 continuous tasks with the fixed 0.1 "
                        "NON_ANGULAR_THRESHOLD instead of tau*sigma")
    a = p.parse_args()

    unknown = [t for t in a.tasks if t not in ft.FOUNDATION_TASKS_MAP]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")

    cfg = {k: getattr(a, k) for k in (
        "test_size", "batch_size", "epochs", "learning_rate", "weight_decay",
        "grad_clip", "lr_warmup_epochs", "patience", "eval_interval", "tau",
        "threads", "skip_existing", "legacy_accuracy", "nonlinearity", "init", "mar")}
    cfg["results_dir"] = a.results_dir
    cfg["es_start"] = a.es_start

    jobs = [(t, M, s, seed, cfg)
            for t in a.tasks for M in a.hidden_sizes
            for s in a.sample_sizes for seed in a.seeds]

    print(f"{len(jobs)} runs | {len(a.tasks)} tasks x M={a.hidden_sizes} "
          f"x s={a.sample_sizes} x seeds={a.seeds}")
    print(f"vanilla RNN ({a.nonlinearity}, {a.init} init, MAR {'ON' if a.mar else 'off'}) "
          f"batch={a.batch_size} lr={a.learning_rate} wd={a.weight_decay} "
          f"clip={a.grad_clip} warmup={a.lr_warmup_epochs} eval={a.eval_interval} "
          f"patience={a.patience}@{a.es_start} test={a.test_size} | "
          f"threshold={THRESHOLD * 180 / np.pi:.0f}deg + fixation gate")
    print(f"-> {a.results_dir}")

    if a.num_processes > 1:
        with Pool(processes=a.num_processes) as pool:
            for msg in pool.imap_unordered(train_one, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(train_one(job), flush=True)
    print("all done")


if __name__ == "__main__":
    main()
