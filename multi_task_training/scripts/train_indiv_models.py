#!/usr/bin/env python
"""Individual (per-task) training of the FLAT, P-free PLRNN on the 100-task foundation
battery, configured to match the joint M64_lr5e4_NPsweep runs as closely as the two
parameterizations allow.

One model per task, trained completely independently: no hierarchisation, no p_vector,
no individual parameters, and no task one-hot in the input (a single-task model needs
no task cue). Model = `multi_task_training.rnn_model.PLRNN`: one set of
A (L,), W (M, M), h (M), C (M, 7), D (5, M), left at its native initialisation.

WHAT IS MATCHED TO THE JOINT RUNS (train_2026-07-19_M64_lr5e4_NPsweep_seeds234.sh):
  fixed datasets        HierarchicalTasksDataset(..., fixed=True) for BOTH train and test,
                        so --sample-size really is the training-set size
  learning rate         5e-4, held constant. (In bptt the 0.999^epoch LambdaLR is
                        overwritten by _apply_lr every epoch, so its exponential decay
                        never takes effect either.)
  regularization        weight_decay=1e-4 on Adam, plus the MAR term
                        tau*sum_i[(diag_eff-1)^2 + offdiag(W_i)^2 + h_i^2] over the first
                        M_reg units, tau=0.01, M_reg=M//2=32 -- identical math to
                        BPTT.regularization_loss
  accuracy thresholds   the joint pipeline's OWN scoring code is imported and called
                        (hierachical_model_task.rnn_model.compute_accuracies), so the
                        angular THRESHOLD is 36 deg (NOT multi_task_training's 20 deg),
                        the Yang-2019 fixation gate is applied, and every per-task branch
                        (binary / scalar / vector / argmax / flipflop / reaction / copytask)
                        is the same code path. scalar_vector_tol is left None because bptt
                        VERIFIED on the cluster (HEAD a46f6f3, same as local): no caller
                        passes scalar_vector_tol, so every M64_lr5e4_NPsweep number for
                        the 7 continuous tasks used the fixed 0.1 NON_ANGULAR_THRESHOLD.
  tau*sigma metric      ON here by default (metric_proposal/tau_sigma_metric.md, tau=0.35):
                        the 7 continuous-readout tasks are scored against tau*sigma_t, the
                        other 93 fall back to 0.1 and are unaffected. --legacy-accuracy
                        restores the fixed 0.1 everywhere. NOTE: the joint checkpoints must
                        be re-scored with the same setting before the arms are comparable
                        on those 7 tasks.
  loss                  normalized_mse_loss + reg only. multi_task_training's
                        train_multitask adds an extra `fix_loss` term that bptt does not
                        have, which is why this script runs its own loop instead.
  grad clip 10, non-finite batch guard with adaptive lr halving, eval_interval 25,
  early stopping on TEST LOSS with early_stopping_start=300 and patience=500.

DELIBERATE DEVIATION: no LR warmup (--lr-warmup-epochs 0). The joint runs ramped the LR
over 100 epochs; here the LR is constant from epoch 0.

NOT MATCHED, and unmatchable: the weight initialisation. The joint model has no W to
initialise -- its W is einsum(p_vector, p2W) with p2W xavier-uniform at gain 0.05, whose
effective std grows as ~sqrt(dp/3) (measured: 0.0005 at P=1 -> 0.0029 at P=32), and whose
effective A starts near 0. The flat PLRNN uses W ~ randn*0.1/sqrt(M) (std 0.0125) and
A ~ 0.9 (near-integrator). There is no dp-independent flat equivalent, so the native
init is used as defined.

Output layout (readable by the existing analysis notebooks):
    <results>/N{L}_s{s}/<task>/seed_{seed}/{metadata.json, results.npz, model.pt}
"""

import argparse
import json
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
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

# The flat model. Imported as a package: rnn_model.py does `from .MAR import ...`.
from multi_task_training.rnn_model import PLRNN  # noqa: E402
from multi_task_training.MAR import regularization_loss  # noqa: E402

# The JOINT pipeline's scoring + loss, so accuracy is the same function the
# M64_lr5e4_NPsweep runs were scored with (36 deg threshold + fixation gate).
from hierachical_model_task.rnn_model import (  # noqa: E402
    THRESHOLD,
    TAU_SIGMA,
    compute_accuracies,
    compute_loss,
    compute_per_task_loss,
    compute_scalar_vector_tol,
    normalized_mse_loss,
)

# HierarchicalTasksDataset is just a trial generator -- despite the name it does no
# hierarchisation. It yields the raw 7 input channels; MultiTaskDataset would append
# a one-hot task-identity vector, which independent training does not need.
from tasks.dataset import HierarchicalTasksDataset, collate_fn  # noqa: E402
import foundation_tasks as ft  # noqa: E402


class FlatPLRNN(PLRNN):
    """PLRNN with the joint pipeline's call signature: forward(inputs, subject=None).

    The joint scoring helpers call `model(inputs, task_ids)`. The flat model has no
    per-task parameters, so the second argument is accepted and ignored. Native
    PLRNN initialisation is untouched.
    """

    def forward(self, inputs, subject=None):  # noqa: D102
        return super().forward(inputs)


def train_one(job):
    """Train a single (task, L, sample_size, seed) combination. Pool-friendly."""
    task_name, L, sample_size, seed, cfg = job

    out = Path(cfg["results_dir"]) / f"N{L}_s{sample_size}" / task_name / f"seed_{seed}"
    if cfg["skip_existing"] and (out / "metadata.json").exists():
        return f"skip  {task_name} L={L} s={sample_size} seed={seed} (already done)"

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg["threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = ft.FOUNDATION_TASKS_MAP[task_name]
    tasks = [task]
    n_out = task.output_dim
    M, M_reg, tau = cfg["hidden_size"], cfg["hidden_size"] // 2, cfg["tau"]

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
    model = FlatPLRNN(M, L, n_out, input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"],
                     weight_decay=cfg["weight_decay"])

    task_loss_channels = ({0: task.loss_channels}
                          if getattr(task, "loss_channels", None) is not None else {})

    # tau*sigma tolerance for the 7 continuous-readout tasks (metric_proposal/
    # tau_sigma_metric.md). Returns {} for the other 93, which then fall back to the
    # fixed NON_ANGULAR_THRESHOLD, so only the intended tasks are affected. The
    # helper snapshots/restores the RNG, so it does not perturb reproducibility.
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
    best_test_loss, best_state, epochs_since_best = float("inf"), None, 0
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

            task_loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels)
            loss = task_loss + regularization_loss(model, tau, M_reg)

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

        # ---- test-side metrics EVERY epoch ----------------------------------------
        # These are cheap (test_size trials per pass), and computing the test LOSS every
        # epoch is what makes best_epoch exact: it is argmin(test_loss), so evaluating it
        # on a 25-epoch grid would quantise the selected epoch to that grid.
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

        # ---- train accuracy only on the interval ----------------------------------
        # This is the expensive pass (sample_size trials, 4x the test set at s=200), and
        # nothing selects on it, so it stays on the --eval-interval cadence.
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == cfg["epochs"]:
            tr_accs = compute_accuracies(
                model, train_loader, device, 1, task_thresholds=None, tasks=tasks,
                scalar_vector_tol=scalar_vector_tol)
            train_eval_epochs.append(epoch + 1)
            train_acc_hist.append(float(tr_accs[0]))

        # early stopping in EPOCHS, only after early_stopping_start (BPTT semantics)
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
        "parameterization": "flat_shared",
        "num_individual_params": 0,
        "task_onehot": False,
        "init": "plrnn_native",
        "sample_size": sample_size,
        "test_size": cfg["test_size"],
        "batch_size": cfg["batch_size"],
        "nonlinear_units": L,
        "hidden_size": M,
        "output_size": n_out,
        "input_dim": input_dim,
        "num_epochs": cfg["epochs"],
        "epochs_run": len(loss_history),
        "stopped_early": stopped_early,
        "checkpoint_selection": selection,
        # first epoch at which test accuracy reached 1.0 (None if never). Recorded only;
        # it does NOT stop the run.
        "epochs_to_first_perfect": next(
            (i + 1 for i, v in enumerate(test_acc_hist) if v >= 1.0), None),
        "early_stopping_patience": cfg["patience"],
        "early_stopping_start": cfg["es_start"],
        "eval_interval": eval_interval,
        "learning_rate": base_lr,
        "lr_warmup_epochs": warm,
        "weight_decay": cfg["weight_decay"],
        "grad_clip": grad_clip,
        "tau": tau,
        "M_reg": M_reg,
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies (fixation gate on)",
        "tau_sigma_metric": not cfg["legacy_accuracy"],
        "tau_sigma": None if cfg["legacy_accuracy"] else TAU_SIGMA,
        # the actual tolerance used for this task ({} / None => fell back to
        # NON_ANGULAR_THRESHOLD, i.e. this is not one of the 7 continuous tasks)
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

    return (f"done  {task_name} L={L} s={sample_size} seed={seed} | "
            f"epochs={len(loss_history)}/{cfg['epochs']}"
            f"{' (early stop)' if stopped_early else ''} "
            f"best_epoch={meta['best_epoch']} best_test_acc={meta['best_test_acc']}")


def main():
    all_names = list(ft.FOUNDATION_TASKS_MAP)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=all_names,
                   help=f"tasks, each in its OWN model (default: all {len(all_names)})")
    p.add_argument("--nonlinear-units", "-L", nargs="+", type=int,
                   default=[0, 1, 2, 4, 8, 16], help="L sweep (default 0 1 2 4 8 16)")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[200])
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--hidden-size", type=int, default=64, help="M")
    p.add_argument("--test-size", type=int, default=50, help="joint runs used TEST=50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0,
                   help="LR ramp over this many epochs. 0 = OFF (the default here); the "
                        "joint runs used 100. Set it only if you want that ramp back.")
    p.add_argument("--patience", type=int, default=500, help="on test loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads PER worker (num_processes*threads <= cores)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip combos whose metadata.json exists (idempotent resubmits)")
    p.add_argument("--legacy-accuracy", action="store_true",
                   help="score the 7 continuous tasks with the fixed 0.1 "
                        "NON_ANGULAR_THRESHOLD instead of tau*sigma. Use this only to "
                        "reproduce the un-re-scored M64_lr5e4_NPsweep numbers.")
    a = p.parse_args()

    unknown = [t for t in a.tasks if t not in ft.FOUNDATION_TASKS_MAP]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")

    cfg = {k: getattr(a, k) for k in (
        "hidden_size", "test_size", "batch_size", "epochs", "learning_rate",
        "weight_decay", "grad_clip", "lr_warmup_epochs", "patience", "eval_interval",
        "tau", "threads", "skip_existing", "legacy_accuracy")}
    cfg["results_dir"] = a.results_dir
    cfg["es_start"] = a.es_start

    jobs = [(t, L, s, seed, cfg)
            for t in a.tasks for L in a.nonlinear_units
            for s in a.sample_sizes for seed in a.seeds]

    print(f"{len(jobs)} runs | {len(a.tasks)} tasks x L={a.nonlinear_units} "
          f"x s={a.sample_sizes} x seeds={a.seeds}")
    print(f"M={a.hidden_size} batch={a.batch_size} lr={a.learning_rate} "
          f"wd={a.weight_decay} clip={a.grad_clip} warmup={a.lr_warmup_epochs} "
          f"eval={a.eval_interval} patience={a.patience}@{a.es_start} "
          f"test={a.test_size} | P=0, native init, "
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
