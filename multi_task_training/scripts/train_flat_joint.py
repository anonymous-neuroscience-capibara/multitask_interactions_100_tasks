#!/usr/bin/env python
"""JOINT training of ONE flat, P-free PLRNN on the whole 100-task foundation battery.

The missing cell in the 2x2. The other three exist:

    parameterization   individual (one model per task)      joint (one model, all tasks)
    hierarchical       (n/a -- a p_vector needs >1 task)    foundation_model/train_foundation.py
    flat (P-free)      scripts/train_indiv_models.py        >>> THIS SCRIPT <<<

This is the no-P joint baseline for the M64_lr5e4_NPsweep_varnorm / final_run arms: one
shared set of A (L,), W (M, M), h (M), C (M, 7+n_tasks), D (5, M) trained on every task at
once, with NO p_vector and NO per-task parameters of any kind.

HOW THE MODEL KNOWS WHICH TASK IT IS DOING
    The hierarchical model gets task identity from its p_vector, so train_foundation.py
    feeds it the raw 7 input channels (HierarchicalTasksDataset) and passes task_ids to
    model(inputs, task_ids) to index that vector. A flat model has no such vector, so
    identity must enter through the INPUT: MultiTaskDataset appends a one-hot task-identity
    vector, widening the input from 7 to 7+n_tasks and C from (M, 7) to (M, 7+n_tasks).
    FlatPLRNN still accepts and ignores the `subject` argument -- task_ids is used by the
    surrounding code (loss channel routing, per-task bookkeeping), never by the network.

    Cost of that choice, for the parameter-count comparison: at M=64 the one-hot adds
    64*100 = 6400 parameters in C (11332 total vs 4932 for the individual runs at
    input_dim=7). The hierarchical arm spends ~65k on p2W alone at P=16. Recorded in
    metadata as `n_task_identity_params` so the comparison can be stated honestly.

    NOT a learned task embedding: that is halfway back to a p_vector and would muddy the
    comparison. One-hot is the Yang-2019 choice and what MultiTaskDataset already does.

WHAT IS MATCHED TO THE JOINT HIERARCHICAL RUNS (train_foundation.py + bptt.py)
  balanced fixed datasets   balanced_indices(n_tasks, per_task) with fixed=True on BOTH,
                            so --sample-size is trials PER TASK (as in train_foundation.py),
                            not a total. The train set is n_tasks * sample_size trials.
  variance-normalized loss  --varnorm (default: the LOSS_VARNORM env var, exactly as
                            bptt.py reads it). ON for every current reference run
                            (M64_lr5e4_NPsweep_varnorm, NPsweep_varnorm_extended,
                            final_run, and all three comparisons/ architectures), so it is
                            required for a like-for-like number. The frozen per-task
                            reference L_i(0) is computed the same way as bptt.py:160-188 --
                            one no-grad pass over the TRAIN loader at init, per-task mean
                            of per_task_mse_loss. NOTE: foundation_model/loss_normalization.md
                            still says layer 2 is "off for every run so far"; that is stale.
  regularization            weight_decay on Adam + the MAR term over the first M_reg=M//2
                            units (tau=0.01), identical math to BPTT.regularization_loss.
  scoring                   the joint pipeline's OWN compute_accuracies is imported and
                            called, so the angular threshold is 36 deg (NOT
                            multi_task_training's 20 deg), the Yang-2019 fixation gate is
                            applied, and every per-task branch (binary / scalar / vector /
                            argmax / flipflop / reaction / perstep-binary / copytask) is
                            the same code path. multi_task_training's own scorer handles
                            only 4 of the 8 flags the battery uses and would silently score
                            the vector/flipflop/reaction tasks through the angular branch.
  tau*sigma metric          ON by default (tau=0.35) for the 7 continuous-readout tasks;
                            --legacy-accuracy restores the fixed 0.1 everywhere.
  dual selection            model.pt = min mean test loss (what early stopping tracks),
                            model_bestacc.pt = max mean test accuracy, both reported.
  grad clip, non-finite batch guard with adaptive lr halving, LR warmup, early stopping on
  TEST LOSS with --es-start / --patience.

DELIBERATE DEVIATIONS
  no fix_loss     multi_task_training.rnn_model.train_multitask adds an extra fix_loss
                  term that bptt does not have, which is why this script runs its own
                  loop rather than calling it. (train_multitask is otherwise generic over
                  n_tasks -- it is the fix_loss and the 20-deg scorer that rule it out.)
  init            the flat PLRNN uses its native W ~ randn*0.1/sqrt(M), A ~ 0.9. The joint
                  model has no W to initialise (it is einsum(p_vector, p2W)), so this is
                  unmatchable in either direction. Same situation as train_indiv_models.py.

COST -- READ BEFORE SUBMITTING
  Per epoch this is n_tasks x the individual runs: at --sample-size 200 the train set is
  20 000 trials (625 batches at batch 32), not 200. Joint batches also mix trial lengths,
  so collate_fn pads to the batch max and the RNN runs T_max steps for every trial in it.
  Budget this like train_foundation.py (E880/GPU, thousands of epochs, best epoch ~1300),
  NOT like the individual runs. --eval-interval governs the expensive test AND train
  accuracy passes; 1 is affordable at 50 test trials but not at 5000. --skip-train-acc
  drops the most expensive pass (the train set is 4x the test set at s200/test50).

Output layout mirrors train_foundation.py so rank_results.py and the notebooks read it:
    <output-dir>/{model.pt, model_bestacc.pt, metadata.json, results.npz}
"""

import argparse
import json
import os
import sys
from datetime import datetime
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

# The flat model, in its one canonical home.
from multi_task_training.flat_model import FlatPLRNN  # noqa: E402
from multi_task_training.MAR import regularization_loss  # noqa: E402

# The JOINT pipeline's scoring + loss, so accuracy and the training objective are the same
# functions the M64_lr5e4_NPsweep_varnorm runs used.
from hierachical_model_task.rnn_model import (  # noqa: E402
    THRESHOLD,
    TAU_SIGMA,
    compute_accuracies,
    compute_loss,
    compute_per_task_loss,
    compute_scalar_vector_tol,
    normalized_mse_loss,
    per_task_mse_loss,
    variance_normalized_mse_loss,
)

# MultiTaskDataset (NOT HierarchicalTasksDataset): it appends the one-hot task-identity
# vector that a P-free joint model needs. Beware -- multi_task_training/Model.py defines a
# DIFFERENT, broken class of the same name (its __getitem__ returns None); this is the one.
from tasks.dataset import MultiTaskDataset, collate_fn  # noqa: E402
import foundation_tasks as ft  # noqa: E402


def balanced_indices(n_tasks, per_task):
    """Exactly `per_task` trials for every task (train_foundation.py:30-34)."""
    return [t for t in range(n_tasks) for _ in range(per_task)]


def compute_task_ref(model, loader, device, n_tasks, task_loss_channels):
    """Frozen per-task reference scale L_i(0), mirroring bptt.py:160-188.

    One no-grad pass over the TRAIN loader at initialisation; each task's reference is the
    mean of its per-sample layer-1 losses, which is ~= its target variance. Tasks that
    somehow contribute nothing fall back to 1.0, as in bptt.
    """
    model.eval()
    acc = {i: [] for i in range(n_tasks)}
    with torch.no_grad():
        for inputs, targets, masks, task_ids in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            masks, task_ids = masks.to(device), task_ids.to(device)
            outputs = model(inputs, task_ids)
            me = masks.unsqueeze(-1).expand_as(outputs)
            mo = outputs * me
            if targets.dim() == 2:
                mt = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1) * me
            else:
                mt = targets * me
            for tid, v in per_task_mse_loss(
                mo, mt, masks, task_ids, n_tasks, task_loss_channels
            ).items():
                if v > 0:
                    acc[tid].append(v)
    model.train()
    return {i: (float(np.mean(acc[i])) if acc[i] else 1.0) for i in range(n_tasks)}


def main():
    all_names = list(ft.FOUNDATION_TASKS_MAP)
    env_varnorm = os.environ.get("LOSS_VARNORM", "").strip().lower() not in (
        "", "0", "false", "no",
    )

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--sample-size", type=int, default=200,
                   help="train trials PER TASK (train_foundation.py semantics)")
    p.add_argument("--test-size", type=int, default=50, help="test trials PER TASK")
    p.add_argument("--nonlinear-units", "-N", type=int, default=4,
                   help="L: number of ReLU units (the rest are linear). NOTE: this is the "
                        "N of the N4/N16 results paths; PLRNN's own third ctor arg is the "
                        "OUTPUT dim, not this.")
    p.add_argument("--hidden-size", "-M", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64, help="joint runs used 64")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=100,
                   help="joint runs ramped the LR over 100 epochs")
    p.add_argument("--patience", type=int, default=300, help="on test loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=10,
                   help="epochs between test/train metric passes. best_epoch is quantised "
                        "to this grid (as in train_foundation.py)")
    p.add_argument("--tau", type=float, default=0.01, help="MAR strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated task names: train on just this SUBSET of the "
                        "battery (battery order is kept; overrides --leave-out). For the "
                        "nonlinearity-as-routing battery-size sweep.")
    p.add_argument("--leave-out", type=str, default="",
                   help="hold this task out of the battery (LOO), by name")
    p.add_argument("--leave-out-index", type=int, default=None,
                   help="same, by index into the full battery")
    p.add_argument("--varnorm", dest="varnorm", action="store_true", default=env_varnorm,
                   help="variance-normalized loss. DEFAULT: the LOSS_VARNORM env var, "
                        "which every current reference run sets to 1")
    p.add_argument("--no-varnorm", dest="varnorm", action="store_false")
    p.add_argument("--skip-train-acc", action="store_true",
                   help="skip the train-accuracy pass (the most expensive one)")
    p.add_argument("--legacy-accuracy", action="store_true",
                   help="score the 7 continuous tasks with the fixed 0.1 "
                        "NON_ANGULAR_THRESHOLD instead of tau*sigma")
    a = p.parse_args()

    # ---- battery ------------------------------------------------------------------
    left_out = None
    if a.leave_out_index is not None and a.leave_out_index >= 0:
        if a.leave_out_index >= len(all_names):
            raise SystemExit(
                f"--leave-out-index {a.leave_out_index} out of range "
                f"(battery has {len(all_names)} tasks: valid 0..{len(all_names) - 1})"
            )
        left_out = all_names[a.leave_out_index]
    elif a.leave_out:
        if a.leave_out not in ft.FOUNDATION_TASKS_MAP:
            raise SystemExit(f"--leave-out '{a.leave_out}' is not in the battery")
        left_out = a.leave_out

    if a.tasks:
        if left_out is not None:
            raise SystemExit("--tasks and --leave-out are mutually exclusive")
        subset = [t.strip() for t in a.tasks.split(",") if t.strip()]
        bad = [t for t in subset if t not in ft.FOUNDATION_TASKS_MAP]
        if bad:
            raise SystemExit(f"--tasks names not in the battery: {bad}")
        if len(set(subset)) != len(subset):
            raise SystemExit("--tasks contains duplicates")
        names = [n for n in all_names if n in set(subset)]  # battery order, like the LOO path
        print(f"SUBSET: {len(names)} tasks: {names}")
    else:
        names = [n for n in all_names if n != left_out]  # left_out=None -> full battery
    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    n_tasks = len(tasks)

    out_dims = {t.output_dim for t in tasks}
    if len(out_dims) != 1:
        raise SystemExit(f"tasks disagree on output_dim: {sorted(out_dims)}")
    n_out = out_dims.pop()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M, tau = a.hidden_size, a.tau
    M_reg = M // 2
    L = a.nonlinear_units

    if left_out is not None:
        print(f"LEAVE-ONE-OUT: holding out '{left_out}' "
              f"(index {all_names.index(left_out)} of {len(all_names)})")
    print(f"FLAT JOINT (P=0, one-hot task input): {n_tasks} tasks | M={M} N={L} "
          f"s={a.sample_size}/task test={a.test_size}/task seed={a.seed}")
    print(f"device: {device} | varnorm={'ON' if a.varnorm else 'OFF'} | "
          f"threshold={THRESHOLD * 180 / np.pi:.0f}deg + fixation gate")

    # ---- data ---------------------------------------------------------------------
    # fixed=True on BOTH: trials are drawn once and reused every epoch, so --sample-size
    # really is the training-set size. task_indices is passed explicitly -- left as None,
    # MultiTaskDataset draws np.random.randint, i.e. an UNBALANCED multinomial.
    train_idx = balanced_indices(n_tasks, a.sample_size)
    test_idx = balanced_indices(n_tasks, a.test_size)
    train_ds = MultiTaskDataset(
        tasks, n_trials=len(train_idx), task_indices=train_idx, fixed=True)
    test_ds = MultiTaskDataset(
        tasks, n_trials=len(test_idx), task_indices=test_idx, fixed=True)

    # Dedicated generator so batch ORDER is independent of the global RNG.
    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=g)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    input_dim = next(iter(train_loader))[0].shape[-1]
    print(f"train {len(train_ds)} trials / {len(train_loader)} batches | "
          f"test {len(test_ds)} trials | input_dim={input_dim} "
          f"(7 base + {n_tasks} one-hot)")

    model = FlatPLRNN(M, L, n_out, input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=a.learning_rate,
                     weight_decay=a.weight_decay)

    # Keyed by task_id: this dict is the ONE place task_ids changes the objective, routing
    # each task to its own loss channels. A missing key silently falls back to "all
    # channels except 0", which would change the loss for every channel-restricted task.
    task_loss_channels = {
        i: t.loss_channels for i, t in enumerate(tasks)
        if getattr(t, "loss_channels", None) is not None
    }

    # tau*sigma tolerance for the continuous-readout tasks; {} for the rest, which fall
    # back to NON_ANGULAR_THRESHOLD. Snapshots/restores the RNG. ~500 trials per task.
    scalar_vector_tol = None if a.legacy_accuracy else (
        compute_scalar_vector_tol(tasks) or None)

    task_ref = None
    if a.varnorm:
        task_ref = compute_task_ref(
            model, train_loader, device, n_tasks, task_loss_channels)
        vals = list(task_ref.values())
        print(f"[LOSS_VARNORM] fixed variance-normalized loss ON | per-task reference "
              f"L_i(0) for {len(task_ref)} tasks: min={min(vals):.4f} "
              f"max={max(vals):.4f} (max/min={max(vals) / max(min(vals), 1e-9):.0f}x)",
              flush=True)

    # ---- train --------------------------------------------------------------------
    base_lr, warm = a.learning_rate, a.lr_warmup_epochs
    lr_scale, n_nonfinite = 1.0, 0
    grad_clip, eval_interval = a.grad_clip, a.eval_interval

    loss_history = []
    train_task_loss_hist = []            # (n_evals-ish, n_tasks), per EPOCH actually
    eval_epochs, test_loss_history = [], []
    test_acc_hist, test_tl_hist = [], []  # each entry: list of n_tasks floats
    train_acc_hist, train_eval_epochs = [], []
    best_test_loss, best_state = float("inf"), None
    best_acc_mean, best_acc_state, best_acc_epoch = -float("inf"), None, None
    stopped_early = False

    for epoch in range(a.epochs):
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for grp in optimizer.param_groups:
            grp["lr"] = base_lr * factor * lr_scale

        model.train()
        epoch_loss, n_batches, epoch_nonfinite = 0.0, 0, 0
        ep_task_loss = {i: [] for i in range(n_tasks)}

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

            if task_ref is not None:
                task_loss = variance_normalized_mse_loss(
                    masked_outputs, masked_targets, masks, task_ids,
                    task_loss_channels, task_ref)
            else:
                task_loss = normalized_mse_loss(
                    masked_outputs, masked_targets, masks, task_ids, task_loss_channels)
            loss = task_loss + regularization_loss(model, tau, M_reg)

            # skip the batch so NaN can't reach the weights; LR cools once per epoch below
            if not torch.isfinite(loss):
                epoch_nonfinite += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            if grad_clip and grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    epoch_nonfinite += 1
                    optimizer.zero_grad()
                    continue
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

            # per-task TRAIN loss, from the outputs we already have (no extra forward)
            with torch.no_grad():
                for tid, v in per_task_mse_loss(
                    masked_outputs.detach(), masked_targets.detach(), masks, task_ids,
                    n_tasks, task_loss_channels,
                ).items():
                    if v > 0:
                        ep_task_loss[tid].append(v)

        # halve the LR once per epoch that exploded, floor 1e-3 (bptt.py:349-353)
        if epoch_nonfinite > 0:
            n_nonfinite += epoch_nonfinite
            lr_scale = max(lr_scale * 0.5, 1e-3)
            print(f"[stability] epoch {epoch + 1}: {epoch_nonfinite} non-finite "
                  f"batch(es) skipped; lr_scale -> {lr_scale:.4g}", flush=True)

        loss_history.append(epoch_loss / max(n_batches, 1))
        train_task_loss_hist.append([
            float(np.mean(ep_task_loss[i])) if ep_task_loss[i] else 0.0
            for i in range(n_tasks)
        ])

        # ---- metrics on the --eval-interval cadence -------------------------------
        # Unlike the individual runs, test metrics are NOT per-epoch: the test set is
        # n_tasks * --test-size trials (5000 at defaults), so a per-epoch pass is not
        # affordable. best_epoch is therefore quantised to this grid, as in
        # train_foundation.py.
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == a.epochs:
            tl = compute_loss(model, test_loader, device, tau, M_reg,
                              task_loss_channels, task_ref)
            per_task_tl = compute_per_task_loss(
                model, test_loader, device, n_tasks, task_loss_channels)
            accs = compute_accuracies(
                model, test_loader, device, n_tasks, task_thresholds=None, tasks=tasks,
                scalar_vector_tol=scalar_vector_tol)

            acc_vec = [float(accs[i]) for i in range(n_tasks)]
            eval_epochs.append(epoch + 1)
            test_loss_history.append(float(tl))
            test_tl_hist.append([float(per_task_tl[i]) for i in range(n_tasks)])
            test_acc_hist.append(acc_vec)
            mean_acc = float(np.mean(acc_vec))

            if tl < best_test_loss:
                best_test_loss = float(tl)
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if mean_acc > best_acc_mean:
                best_acc_mean = mean_acc
                best_acc_epoch = epoch + 1
                best_acc_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}

            msg = (f"epoch {epoch + 1}/{a.epochs} | train {loss_history[-1]:.4f} | "
                   f"test {tl:.4f} | mean acc {mean_acc:.2%}")

            if not a.skip_train_acc:
                tr = compute_accuracies(
                    model, train_loader, device, n_tasks, task_thresholds=None,
                    tasks=tasks, scalar_vector_tol=scalar_vector_tol)
                train_eval_epochs.append(epoch + 1)
                train_acc_hist.append([float(tr[i]) for i in range(n_tasks)])
                msg += f" | train acc {np.mean(train_acc_hist[-1]):.2%}"
            print(msg, flush=True)

            # early stopping in EPOCHS, only after --es-start (BPTT semantics)
            if epoch + 1 > a.es_start:
                since_best = (epoch + 1) - eval_epochs[
                    int(np.argmin(test_loss_history))]
                if a.patience and since_best >= a.patience:
                    stopped_early = True
                    print(f"early stop: {since_best} epochs since best test loss")
                    break

    # ---- save ---------------------------------------------------------------------
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if best_acc_state is not None:
        torch.save(best_acc_state, out / "model_bestacc.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
        selection = "best_test_loss"
    else:
        selection = "final"
    torch.save(model.state_dict(), out / "model.pt")

    test_lh = np.asarray(test_loss_history, dtype=float)
    best_idx = int(np.argmin(test_lh)) if test_lh.size else -1

    if best_idx >= 0:
        best_test = {names[i]: float(test_acc_hist[best_idx][i]) for i in range(n_tasks)}
        best_test_loss_pt = {
            names[i]: float(test_tl_hist[best_idx][i]) for i in range(n_tasks)}
        best_epoch = int(eval_epochs[best_idx])
    else:
        best_test, best_test_loss_pt, best_epoch = {}, {}, None

    if test_acc_hist:
        acc_traj = np.asarray(test_acc_hist, dtype=float)     # (n_evals, n_tasks)
        bai = int(np.argmax(acc_traj.mean(axis=1)))
        best_test_byacc = {names[i]: float(acc_traj[bai, i]) for i in range(n_tasks)}
        mean_best_acc_selected = float(acc_traj[bai].mean())
    else:
        best_test_byacc, mean_best_acc_selected = {}, None

    _acc_msg = "n/a" if mean_best_acc_selected is None else f"{mean_best_acc_selected:.2%}"
    _loss_msg = "n/a" if not best_test else f"{np.mean(list(best_test.values())):.2%}"
    print(f"[SELECT] min-loss epoch {best_epoch} -> mean acc {_loss_msg}   |   "
          f"max-acc epoch {best_acc_epoch} -> mean acc {_acc_msg}", flush=True)

    # (n_tasks, n_evals) to match train_foundation.py's results.npz orientation.
    def _T(hist):
        return (np.asarray(hist, dtype=float).T if hist
                else np.zeros((n_tasks, 0), dtype=float))

    np.savez(
        out / "results.npz",
        loss_history=np.asarray(loss_history, dtype=float),
        test_loss_history=test_lh,
        eval_epochs=np.asarray(eval_epochs, dtype=int),
        test_task_accuracies=_T(test_acc_hist),
        test_task_losses=_T(test_tl_hist),
        # train accuracy is on the eval cadence and EMPTY under --skip-train-acc; index it
        # with train_eval_epochs. train_task_losses is per EPOCH, so it is longer.
        train_task_accuracies=_T(train_acc_hist),
        train_eval_epochs=np.asarray(train_eval_epochs, dtype=int),
        train_task_losses=_T(train_task_loss_hist),
    )

    meta = {
        "n_tasks": n_tasks,
        "task_names": names,
        "seed": a.seed,
        "left_out_task": left_out,
        "left_out_index": all_names.index(left_out) if left_out is not None else None,
        # what makes this run the flat arm
        "parameterization": "flat_shared_joint",
        "num_individual_params": 0,
        "task_onehot": True,
        "init": "plrnn_native",
        "input_dim": input_dim,
        "n_task_identity_params": int(M * n_tasks),
        "n_shared_params": int(sum(q.numel() for q in model.parameters())),
        # config
        "nonlinear_units": L,
        "hidden_size": M,
        "output_size": n_out,
        "sample_size": a.sample_size,
        "test_size": a.test_size,
        "batch_size": a.batch_size,
        "num_epochs": a.epochs,
        "epochs_run": len(loss_history),
        "stopped_early": stopped_early,
        "learning_rate": base_lr,
        "lr_warmup_epochs": warm,
        "weight_decay": a.weight_decay,
        "grad_clip": grad_clip,
        "tau": tau,
        "M_reg": M_reg,
        "eval_interval": eval_interval,
        "early_stopping_patience": a.patience,
        "early_stopping_start": a.es_start,
        "loss_varnorm": bool(a.varnorm),
        "n_nonfinite_batches": n_nonfinite,
        "final_lr_scale": lr_scale,
        # scoring
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies "
                   "(fixation gate on)",
        "tau_sigma_metric": not a.legacy_accuracy,
        "tau_sigma": None if a.legacy_accuracy else TAU_SIGMA,
        # results, both selections
        "checkpoint_selection": selection,
        "best_epoch": best_epoch,
        "best_test_accuracy_per_task": best_test,
        "mean_best_test_accuracy": (float(np.mean(list(best_test.values())))
                                    if best_test else None),
        "best_test_loss_per_task": best_test_loss_pt,
        "mean_best_test_loss": (float(np.mean(list(best_test_loss_pt.values())))
                                if best_test_loss_pt else None),
        "best_acc_epoch": best_acc_epoch,
        "mean_best_acc_selected_test_accuracy": mean_best_acc_selected,
        "best_acc_selected_test_accuracy_per_task": best_test_byacc,
        "timestamp": datetime.now().isoformat(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nmean best-epoch test acc {_loss_msg}  ->  {out}")


if __name__ == "__main__":
    main()
