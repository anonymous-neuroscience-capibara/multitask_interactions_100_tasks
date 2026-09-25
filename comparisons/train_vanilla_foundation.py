#!/usr/bin/env python
"""Joint training of the hierarchical VANILLA RNN on the full foundation battery.

The vanilla-RNN counterpart of foundation_model/train_foundation.py: one model over
all FOUNDATION_TASKS_MAP tasks (task = subject), per-task parameters produced from a
P-dim p_vector through shared projections, trained with the SAME BPTT trainer,
loss, scoring (36 deg + fixation gate + tau*sigma), early stopping, and output
layout (metadata.json / results.npz readable by rank_results.py and the notebooks).

WHAT DIFFERS FROM train_foundation.py (the architecture comparison itself):
  recurrence      z_t = phi(W z_{t-1} + C x_t + h) with phi = tanh by default
                  (--nonlinearity relu shares the PLRNN's nonlinearity and isolates
                  its structural A/ReLU split). No A, no L: the --nonlinear-units
                  axis of the NP sweep does not exist here; the sweep axes are
                  P (--num-individual-params) and M (--hidden-size).
  MAR             OFF by default (tau=0): the manifold-attractor term is a
                  PLRNN-specific inductive bias. --mar turns it back on (tau 0.01),
                  where BPTT.regularization_loss with model.L=0 computes exactly the
                  vanilla analogue ((W_ii-1)^2 + offdiag + h^2, first M//2 units).
  no --a-clamp    there is no A to clamp.

Everything else (lr 5e-4 via --learning-rate, warmup, grad clip, init modes on p2W,
balanced fixed datasets, leave-one-out support, checkpointing) matches.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tasks.dataset import HierarchicalTasksDataset, collate_fn  # noqa: E402
from hierachical_model_task.utils import HiearchicalModelConfig  # noqa: E402
from hierachical_model_task.bptt import BPTT  # noqa: E402
from comparisons.hierarchical_vanilla_rnn import HierarchicalVanillaRNN  # noqa: E402
import foundation_tasks as ft  # noqa: E402


def balanced_indices(n_tasks, per_task):
    idx = []
    for t in range(n_tasks):
        idx.extend([t] * per_task)
    return idx


def apply_init(model, mode, gain):
    """Re-initialize the recurrent projection p2W (in place).

    mode='gaussian'   -> xavier-normal (Gaussian) on the whole (dp, M, M) tensor.
    mode='orthogonal' -> each (M, M) slice orthonormal, scaled by `gain`.
    mode='default'    -> leave the model's own xavier-uniform init untouched.
    """
    with torch.no_grad():
        p2W = model.p2W  # (dp, M, M)
        if mode == "gaussian":
            torch.nn.init.xavier_normal_(p2W, gain=gain)
        elif mode == "orthogonal":
            for d in range(p2W.shape[0]):
                torch.nn.init.orthogonal_(p2W[d], gain=gain)


def main():
    p = argparse.ArgumentParser(
        description="Joint-train the hierarchical vanilla RNN on the foundation battery."
    )
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--early-stopping-patience", type=int, default=50)
    p.add_argument("--early-stopping-start", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--w-init-gain", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--nonlinearity",
        choices=["tanh", "relu"],
        default="tanh",
        help="tanh = textbook vanilla RNN (default); relu shares the PLRNN's "
        "nonlinearity and isolates its structural split",
    )
    p.add_argument(
        "--mar",
        action="store_true",
        help="add the MAR-analogue regularizer ((W_ii-1)^2 + offdiag + bias^2 over "
        "the first M//2 units, tau=--tau). OFF by default: MAR is a PLRNN-specific "
        "inductive bias.",
    )
    p.add_argument("--tau", type=float, default=0.01, help="MAR strength (with --mar)")
    p.add_argument(
        "--init-mode",
        choices=["default", "gaussian", "orthogonal"],
        default="default",
        help="p2W init: default=xavier-uniform (model default), "
        "gaussian=xavier-normal, orthogonal=orthonormal slices",
    )
    p.add_argument(
        "--init-gain",
        type=float,
        default=0.9,
        help="gain for gaussian/orthogonal init (default 0.9)",
    )
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-individual-params", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "foundation_joint_vanilla"
        ),
    )
    p.add_argument("--resume", type=str, default="")
    p.add_argument(
        "--leave-out",
        type=str,
        default="",
        help="task NAME to hold out; joint-train on the other tasks. "
        "'' = train on the full battery (default).",
    )
    p.add_argument(
        "--leave-out-index",
        type=int,
        default=-1,
        help="task INDEX (0-based, into FOUNDATION_TASKS_MAP order) to hold "
        "out; overrides --leave-out. -1 = off.",
    )
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    all_names = list(ft.FOUNDATION_TASKS_MAP)
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

    names = [n for n in all_names if n != left_out]  # left_out=None -> full battery
    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    n_tasks = len(tasks)
    if left_out is not None:
        print(
            f"LEAVE-ONE-OUT: holding out '{left_out}' "
            f"(index {all_names.index(left_out)} of {len(all_names)})"
        )
    print(
        f"Foundation battery: {n_tasks} tasks | VANILLA RNN ({a.nonlinearity}, "
        f"MAR {'ON' if a.mar else 'off'}) "
        f"| M={a.hidden_size} P={a.num_individual_params} "
        f"s={a.sample_size} seed={a.seed}"
    )

    train_idx = balanced_indices(n_tasks, a.sample_size)
    test_idx = balanced_indices(n_tasks, a.test_size)
    train_ds = HierarchicalTasksDataset(
        tasks, n_trials=len(train_idx), task_indices=train_idx, fixed=True
    )
    test_ds = HierarchicalTasksDataset(
        tasks, n_trials=len(test_idx), task_indices=test_idx, fixed=True
    )
    # Dedicated generator seeds the shuffle ORDER independently of the global RNG,
    # so batch order is identical regardless of init (orthogonal re-init consumes
    # extra RNG) -> a clean init A/B (only p2W differs, not the data order).
    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate_fn
    )

    args = HiearchicalModelConfig(tasks=tasks)
    args.nonlinear_units = 0  # vanilla RNN: no linear/nonlinear split
    args.hidden_size = a.hidden_size
    args.num_individual_params = a.num_individual_params
    args.num_epochs = a.num_epochs
    args.early_stopping_patience = a.early_stopping_patience
    args.early_stopping_start = a.early_stopping_start
    args.grad_clip = a.grad_clip
    args.weight_decay = a.weight_decay
    args.lr_warmup_epochs = a.lr_warmup_epochs
    args.w_init_gain = a.w_init_gain
    # tau=0 turns BPTT's MAR term into an exact zero -> weight decay only
    args.tau = a.tau if a.mar else 0.0
    args.nonlinearity = a.nonlinearity
    if a.learning_rate is not None:
        args.learning_rate = a.learning_rate
        args.individual_learning_rate = a.learning_rate
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")
    args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = HierarchicalVanillaRNN(args, HierarchicalTasksDataset)
    if a.resume:
        print(f"resuming from checkpoint: {a.resume}")
        model.load_state_dict(torch.load(a.resume, map_location=args.device))
    alg = BPTT(model, args)

    # <<< optional init-mode (OFF by default) >>>
    if a.init_mode != "default" and not a.resume:
        apply_init(alg.model, a.init_mode, a.init_gain)
        print(f"init-mode: {a.init_mode} (gain={a.init_gain}) on p2W")

    (
        loss_hist,
        train_acc,
        test_loss_hist,
        test_acc,
        train_tl,
        test_tl,
        best_epoch,
        _,
    ) = alg.train(
        train_loader,
        test_loader,
        verbose=True,
        task_names=names,
        eval_interval=a.eval_interval,
        checkpoint_path=str(out / "model.pt"),
        checkpoint_acc_path=str(out / "model_bestacc.pt"),
    )

    torch.save(alg.model.state_dict(), out / "model.pt")
    best_idx = int(np.argmin(test_loss_hist)) if test_loss_hist else -1
    best_test = {names[i]: float(test_acc[i][best_idx]) for i in range(n_tasks)}
    best_test_loss = {names[i]: float(test_tl[i][best_idx]) for i in range(n_tasks)}

    # Accuracy-optimal selection, in parallel to the loss-optimal one above.
    if test_acc and n_tasks:
        mean_acc_traj = np.mean([test_acc[i] for i in range(n_tasks)], axis=0)
        best_acc_idx = int(np.argmax(mean_acc_traj))
        best_test_byacc = {
            names[i]: float(test_acc[i][best_acc_idx]) for i in range(n_tasks)
        }
        mean_best_acc_selected = float(mean_acc_traj[best_acc_idx])
    else:
        best_test_byacc = {}
        mean_best_acc_selected = None
    best_acc_epoch = getattr(alg, "best_acc_epoch", None)
    _acc_msg = (
        "n/a" if mean_best_acc_selected is None else f"{mean_best_acc_selected:.2%}"
    )
    print(
        f"[SELECT] min-loss epoch {best_epoch} -> mean acc "
        f"{np.mean(list(best_test.values())):.2%}   |   max-acc epoch {best_acc_epoch} "
        f"-> mean acc {_acc_msg}",
        flush=True,
    )
    np.savez(
        out / "results.npz",
        loss_history=np.array(loss_hist),
        test_loss_history=np.array(test_loss_hist),
        train_task_accuracies=np.array([train_acc[i] for i in range(n_tasks)]),
        test_task_accuracies=np.array([test_acc[i] for i in range(n_tasks)]),
        train_task_losses=np.array([train_tl[i] for i in range(n_tasks)]),
        test_task_losses=np.array([test_tl[i] for i in range(n_tasks)]),
    )
    with open(out / "metadata.json", "w") as f:
        json.dump(
            {
                "model": "vanilla_rnn_hierarchical",
                "nonlinearity": a.nonlinearity,
                "mar": a.mar,
                "tau": a.tau if a.mar else 0.0,
                "n_tasks": n_tasks,
                "task_names": names,
                "seed": a.seed,
                "left_out_task": left_out,
                "left_out_index": (
                    all_names.index(left_out) if left_out is not None else None
                ),
                "resumed_from": a.resume or None,
                "num_epochs": a.num_epochs,
                "best_epoch": best_epoch,
                "nonlinear_units": None,
                "hidden_size": a.hidden_size,
                "num_individual_params": a.num_individual_params,
                "batch_size": a.batch_size,
                "learning_rate": a.learning_rate,
                "weight_decay": a.weight_decay,
                "init_mode": a.init_mode,
                "init_gain": a.init_gain,
                "sample_size": a.sample_size,
                "best_test_accuracy_per_task": best_test,
                "mean_best_test_accuracy": float(np.mean(list(best_test.values()))),
                "best_test_loss_per_task": best_test_loss,
                "mean_best_test_loss": float(np.mean(list(best_test_loss.values()))),
                "best_acc_epoch": best_acc_epoch,
                "mean_best_acc_selected_test_accuracy": mean_best_acc_selected,
                "best_acc_selected_test_accuracy_per_task": best_test_byacc,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(
        f"\nmean best-epoch test acc {np.mean(list(best_test.values())):.1%}  ->  {out}"
    )


if __name__ == "__main__":
    main()
