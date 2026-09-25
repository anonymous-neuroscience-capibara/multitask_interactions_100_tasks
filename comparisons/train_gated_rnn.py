#!/usr/bin/env python
"""Individual (per-task) training of a GATED RNN (LSTM or GRU, --cell) on the
100-task foundation battery -- the gated baselines next to comparisons/
train_vanilla_rnn.py and the flat-PLRNN runs.

Model: single-layer torch.nn.LSTM / torch.nn.GRU (PyTorch-default uniform
+-1/sqrt(M) init, the standard recipe for gated cells) with the same linear
readout D (N, M) as the vanilla/PLRNN models. One model per task, no task
one-hot, no per-task parameters.

WHAT IS MATCHED TO train_vanilla_rnn.py / train_indiv_models.py:
  fixed datasets, lr 5e-4 constant (no warmup), weight decay 1e-4, the joint
  pipeline's normalized_mse_loss (no fix_loss), compute_accuracies scoring
  (36 deg + fixation gate + tau*sigma), grad clip 10, non-finite guard with
  adaptive lr halving, eval_interval 25, early stopping on TEST LOSS
  (patience 500, start 300), checkpoint = best-by-test-loss state,
  output layout <results>/M{M}_s{s}/<task>/seed_{seed}/.

DELIBERATE DEVIATIONS:
  no MAR             the manifold-attractor term has no analogue for gated
                     cells (there is no single recurrent diagonal to pin to 1);
                     weight decay only.
  parameter count    at equal M an LSTM has ~4x and a GRU ~3x the recurrent
                     parameters of the vanilla RNN / PLRNN. n_shared_params is
                     recorded in metadata.json; sweep --hidden-sizes (e.g. 32
                     for LSTM vs 64 vanilla) if you want a params-matched arm
                     in addition to the state-dimension-matched one.

Output layout:
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


class GatedRNN(nn.Module):
    """LSTM or GRU with the joint pipeline's call signature: forward(inputs, subject=None).

    Recurrent core = torch.nn.LSTM / torch.nn.GRU (single layer, batch_first,
    PyTorch default init); readout D (N, M) matches the vanilla/PLRNN classes.
    L is kept as an attribute (== M) for code that reads model.L.
    """

    def __init__(self, M, N, input_dim, cell="lstm"):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError(f"unknown cell: {cell!r}")
        self.M = M
        self.L = M
        self.N = N
        self.input_dim = input_dim
        self.cell_type = cell

        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, M, num_layers=1, batch_first=True)
        self.D = nn.Parameter(torch.randn(N, M) * 0.1)

    def forward(self, inputs, subject=None, return_latent=False):
        """
        Args:
            inputs: (batch, T, input_dim)
            subject: accepted and ignored (joint scoring passes task_ids here)
            return_latent: also return the hidden trajectory (batch, T, M).
                           For the LSTM this is h_t (the output state, the part
                           the readout sees), not the cell state c_t.
        Returns:
            outputs: (batch, T, N)  [, latents: (batch, T, M)]
        """
        h_all, _ = self.rnn(inputs)  # (batch, T, M)
        outputs = h_all @ self.D.t()
        if return_latent:
            return outputs, h_all
        return outputs


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
    model = GatedRNN(M, n_out, input_dim, cell=cfg["cell"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"],
                     weight_decay=cfg["weight_decay"])

    task_loss_channels = ({0: task.loss_channels}
                          if getattr(task, "loss_channels", None) is not None else {})

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
        tl = compute_loss(model, test_loader, device, 0.0, 0, task_loss_channels)
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
        "model": cfg["cell"],
        "init": "pytorch_default",
        "mar": False,
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
        "epochs_to_first_perfect": next(
            (int(eval_epochs[i]) for i, v in enumerate(test_acc_hist) if v >= 1.0), None),
        "early_stopping_patience": cfg["patience"],
        "early_stopping_start": cfg["es_start"],
        "eval_interval": eval_interval,
        "learning_rate": base_lr,
        "lr_warmup_epochs": warm,
        "weight_decay": cfg["weight_decay"],
        "grad_clip": grad_clip,
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
    p.add_argument("--cell", choices=["lstm", "gru"], required=True,
                   help="gated cell type (explicit on purpose: keep lstm and gru "
                        "runs in separate --results-dir trees)")
    p.add_argument("--tasks", nargs="+", default=all_names,
                   help=f"tasks, each in its OWN model (default: all {len(all_names)})")
    p.add_argument("--hidden-sizes", "-M", nargs="+", type=int, default=[64],
                   help="M sweep (default 64; note the 4x/3x recurrent-parameter "
                        "count vs the vanilla RNN at equal M)")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[200])
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--test-size", type=int, default=50, help="joint runs used TEST=50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=500, help="on test loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=25)
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
        "cell", "test_size", "batch_size", "epochs", "learning_rate", "weight_decay",
        "grad_clip", "lr_warmup_epochs", "patience", "eval_interval",
        "threads", "skip_existing", "legacy_accuracy")}
    cfg["results_dir"] = a.results_dir
    cfg["es_start"] = a.es_start

    jobs = [(t, M, s, seed, cfg)
            for t in a.tasks for M in a.hidden_sizes
            for s in a.sample_sizes for seed in a.seeds]

    print(f"{len(jobs)} runs | {len(a.tasks)} tasks x M={a.hidden_sizes} "
          f"x s={a.sample_sizes} x seeds={a.seeds}")
    print(f"{a.cell.upper()} (pytorch-default init, no MAR) "
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
