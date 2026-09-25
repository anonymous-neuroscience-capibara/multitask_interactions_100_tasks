#!/usr/bin/env python
"""Foundation training that ALSO snapshots the per-task recurrent weights over
learning, so the weight tensor W in R^{n_tasks x M x M} can be tracked across
training and analysed for low-tensor-rank learning (Pellegrino et al., NeurIPS
2023, "Low Tensor Rank Learning of Neural Dynamics").

This is a thin fork of train_foundation.py -- it reproduces exactly the
run the M64 N x P sweep launches (same script lineage => same DataLoader
generator => same RNG), and leaves bptt.py / model.py untouched. The ONLY
addition is a per-epoch snapshot hook installed by monkey-patching alg._apply_lr
(which bptt.BPTT.train calls once at the top of every epoch, receiving the epoch
index). At each snapshot epoch we pull the effective per-task parameters via the
(possibly spectrally-capped) hierarchisation scheme and stash them.

Defaults are pinned to the requested cell:
    foundation_joint_M64_N2_P24_s200_b64_lr0.0005_seed3
so `python "foundation model/train_foundation_snapshots.py"` reproduces it and
writes W_snapshots.npz alongside model.pt / results.npz / metadata.json.

What gets saved (W_snapshots.npz, compressed):
    W        (K, n_tasks, M, M)  effective coupling matrix per task, per snapshot
    A        (K, n_tasks, L)     diagonal self-recurrence of the L nonlinear units
    h        (K, n_tasks, M)     per-task bias
    epochs   (K,)                epoch index entering which the snapshot was taken
                                 (epoch 0 == initialization, i.e. W^(0))
    W_best   (n_tasks, M, M)     effective W of the restored best-by-test model
    best_epoch  scalar
    task_names, M, L, P, seed    metadata for the analysis script

Note on "the weight matrix": this AL-RNN keeps the L nonlinear units' linear
self-loop in a SEPARATE diagonal A (see model.forward), so the full linear-regime
recurrent Jacobian is W with A ADDED onto the last-L diagonal entries. We save W
and A separately; fold A into W's latent diagonal downstream if you want the
paper's single recurrent operator. Snapshots reflect the weights ENTERING each
epoch (pre-update), evenly spaced by --snapshot-interval, which is the natural
discrete "trial" index k of the low-tensor-rank framework.

Storage: K * n_tasks * M * M * 4 bytes for W (dominant term). E.g. 98 tasks,
M=64, interval=25 over ~6000 epochs => ~240 snapshots => ~385 MB before
compression. Increase --snapshot-interval if that is too much.
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

from tasks.dataset import HierarchicalTasksDataset, collate_fn
from hierachical_model_task.utils import HiearchicalModelConfig
from hierachical_model_task.bptt import BPTT
from hierachical_model_task.model import HierarchicalPLRNN
import foundation_tasks as ft


def balanced_indices(n_tasks, per_task):
    idx = []
    for t in range(n_tasks):
        idx.extend([t] * per_task)
    return idx


def spectral_cap_W(W, rho, n_iter=3):
    """Bound the largest singular value of each (M, M) matrix in a batch to <= rho.

    Identical to train_foundation.py. W: (B, M, M). Returns W scaled by
    min(1, rho / sigma_max) per matrix (a CAP, not a hard normalization).
    """
    B, M, _ = W.shape
    with torch.no_grad():
        u = torch.randn(B, M, device=W.device, dtype=W.dtype)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        v = None
        for _ in range(n_iter):
            v = torch.einsum("bij,bi->bj", W, u)  # W^T u
            v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
            u = torch.einsum("bij,bj->bi", W, v)  # W v
            u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
    sigma = torch.einsum("bi,bij,bj->b", u, W, v).abs()  # u^T W v (diff. wrt W)
    scale = (rho / (sigma + 1e-9)).clamp(max=1.0)
    return W * scale.view(B, 1, 1)


def install_spectral_cap(model, rho, n_iter):
    """Monkey-patch get_parameters so W is spectrally capped every forward."""
    scheme = model.hierarchisation_scheme
    orig = scheme.get_parameters

    def patched(subject):
        A, W, h, C, D = orig(subject)
        W = spectral_cap_W(W, rho, n_iter)
        return A, W, h, C, D

    scheme.get_parameters = patched


def apply_init(model, mode, gain):
    """Re-initialize the recurrent projection p2W (in place). Same as spectral."""
    with torch.no_grad():
        p2W = model.p2W  # (dp, M, M)
        if mode == "gaussian":
            torch.nn.init.xavier_normal_(p2W, gain=gain)
        elif mode == "orthogonal":
            for d in range(p2W.shape[0]):
                torch.nn.init.orthogonal_(p2W[d], gain=gain)


def install_snapshot_hook(alg, n_tasks, interval, store):
    """Record the per-task (A, W, h) tensor every `interval` epochs.

    Wraps alg._apply_lr, which bptt.BPTT.train() calls once at the top of each
    epoch with the epoch index -- so this fires exactly once per epoch, BEFORE
    that epoch's weight updates. epoch 0 therefore captures the initialization.
    The (possibly spectral-capped) scheme.get_parameters is used, so W is the
    EFFECTIVE recurrent matrix the model actually runs with.
    """
    orig_apply = alg._apply_lr
    scheme = alg.model.hierarchisation_scheme
    device = alg.device
    subjects = torch.arange(n_tasks, device=device)

    def snapshot(epoch):
        with torch.no_grad():
            A, W, h, _, _ = scheme.get_parameters(subjects)
            store["epochs"].append(int(epoch))
            store["W"].append(W.detach().to("cpu", torch.float32).numpy())
            store["A"].append(A.detach().to("cpu", torch.float32).numpy())
            store["h"].append(h.detach().to("cpu", torch.float32).numpy())

    def patched(epoch):
        orig_apply(epoch)  # keep LR warmup / adaptive scaling intact
        if epoch % interval == 0:
            snapshot(epoch)

    alg._apply_lr = patched
    return subjects  # reused for the final best-model snapshot


def main():
    p = argparse.ArgumentParser(
        description="Joint-train the AL-RNN on the foundation battery, snapshotting "
        "the per-task weight tensor over learning for low-tensor-rank analysis."
    )
    # ---- defaults PINNED to foundation_joint_M64_N2_P24_s200_b64_lr0.0005_seed3
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--num-epochs", type=int, default=6000)
    p.add_argument("--early-stopping-patience", type=int, default=500)
    p.add_argument("--early-stopping-start", type=int, default=300)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--a-clamp", type=float, default=0.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=100)
    p.add_argument("--w-init-gain", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    p.add_argument(
        "--spectral-rho",
        type=float,
        default=0.0,
        help="cap on sigma_max(W) (0 = off; matches the sweep cell)",
    )
    p.add_argument("--spectral-iters", type=int, default=3)
    p.add_argument(
        "--init-mode", choices=["default", "gaussian", "orthogonal"], default="default"
    )
    p.add_argument("--init-gain", type=float, default=0.9)
    p.add_argument("--nonlinear-units", type=int, default=2)  # N
    p.add_argument("--hidden-size", type=int, default=64)  # M
    p.add_argument("--num-individual-params", type=int, default=24)  # P
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=3)
    # ---- snapshot-specific
    p.add_argument(
        "--snapshot-interval",
        type=int,
        default=25,
        help="record the weight tensor every N epochs (epoch 0 always "
        "captured = W^(0)); larger = fewer snapshots / less disk",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="default: results/foundation_snapshots/<combo folder name>",
    )
    p.add_argument("--resume", type=str, default="")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tasks = list(ft.FOUNDATION_TASKS_MAP.values())
    names = list(ft.FOUNDATION_TASKS_MAP)
    n_tasks = len(tasks)

    # Self-labelling run folder in the sweep's naming scheme, so rank_results.py
    # and the analysis tooling parse it unchanged.
    combo = (
        f"foundation_joint_M{a.hidden_size}_N{a.nonlinear_units}"
        f"_P{a.num_individual_params}_s{a.sample_size}_b{a.batch_size}"
        f"_lr{a.learning_rate}_seed{a.seed}"
    )
    out = (
        Path(a.output_dir)
        if a.output_dir
        else (
            Path(__file__).resolve().parent / "results" / "foundation_snapshots" / combo
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"Foundation battery: {n_tasks} tasks | {combo}")
    print(
        f"snapshot every {a.snapshot_interval} epochs (spectral_rho={a.spectral_rho})"
    )

    train_idx = balanced_indices(n_tasks, a.sample_size)
    test_idx = balanced_indices(n_tasks, a.test_size)
    train_ds = HierarchicalTasksDataset(
        tasks, n_trials=len(train_idx), task_indices=train_idx, fixed=True
    )
    test_ds = HierarchicalTasksDataset(
        tasks, n_trials=len(test_idx), task_indices=test_idx, fixed=True
    )
    # Dedicated generator seeds shuffle ORDER independently of the global RNG
    # (matches train_foundation.py so this reproduces the sweep cell).
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
    args.nonlinear_units = a.nonlinear_units
    args.hidden_size = a.hidden_size
    args.num_individual_params = a.num_individual_params
    args.num_epochs = a.num_epochs
    args.early_stopping_patience = a.early_stopping_patience
    args.early_stopping_start = a.early_stopping_start
    args.grad_clip = a.grad_clip
    args.weight_decay = a.weight_decay
    args.a_clamp = a.a_clamp
    args.lr_warmup_epochs = a.lr_warmup_epochs
    args.w_init_gain = a.w_init_gain
    if a.learning_rate is not None:
        args.learning_rate = a.learning_rate
        args.individual_learning_rate = a.learning_rate
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")
    args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start

    if a.resume:
        print(f"resuming from checkpoint: {a.resume}")
        model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
        model.load_state_dict(torch.load(a.resume, map_location=args.device))
        alg = BPTT(model, args)
    else:
        alg = BPTT(None, args)

    if a.init_mode != "default" and not a.resume:
        apply_init(alg.model, a.init_mode, a.init_gain)
        print(f"init-mode: {a.init_mode} (gain={a.init_gain}) on p2W")
    if a.spectral_rho and a.spectral_rho > 0:
        install_spectral_cap(alg.model, a.spectral_rho, a.spectral_iters)
        print(f"spectral cap ON: sigma_max(W) <= {a.spectral_rho}")

    # <<< the only real addition over train_foundation.py >>>
    store = {"epochs": [], "W": [], "A": [], "h": []}
    subjects = install_snapshot_hook(alg, n_tasks, a.snapshot_interval, store)
    print(f"snapshot hook installed on _apply_lr (interval={a.snapshot_interval})")

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
    )

    # alg.train restored the best-by-test-loss model; capture its effective W too.
    with torch.no_grad():
        _, W_best, _, _, _ = alg.model.hierarchisation_scheme.get_parameters(subjects)
        W_best = W_best.detach().to("cpu", torch.float32).numpy()

    # ---- save the weight-tensor snapshots ----
    epochs = np.array(store["epochs"], dtype=np.int32)
    W_stack = np.stack(store["W"]).astype(np.float32)  # (K, n_tasks, M, M)
    A_stack = np.stack(store["A"]).astype(np.float32)  # (K, n_tasks, L)
    h_stack = np.stack(store["h"]).astype(np.float32)  # (K, n_tasks, M)
    np.savez_compressed(
        out / "W_snapshots.npz",
        W=W_stack,
        A=A_stack,
        h=h_stack,
        epochs=epochs,
        W_best=W_best,
        best_epoch=np.int32(best_epoch),
        task_names=np.array(names),
        M=np.int32(a.hidden_size),
        L=np.int32(a.nonlinear_units),
        P=np.int32(a.num_individual_params),
        seed=np.int32(a.seed),
    )
    print(
        f"saved {W_stack.shape[0]} snapshots -> {out / 'W_snapshots.npz'} "
        f"(W {W_stack.shape}, {W_stack.nbytes / 1e6:.0f} MB uncompressed)"
    )

    # ---- standard artefacts (same as the base scripts) ----
    torch.save(alg.model.state_dict(), out / "model.pt")
    best_idx = int(np.argmin(test_loss_hist)) if test_loss_hist else -1
    best_test = {names[i]: float(test_acc[i][best_idx]) for i in range(n_tasks)}
    best_test_loss = {names[i]: float(test_tl[i][best_idx]) for i in range(n_tasks)}
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
                "n_tasks": n_tasks,
                "task_names": names,
                "seed": a.seed,
                "resumed_from": a.resume or None,
                "num_epochs": a.num_epochs,
                "best_epoch": best_epoch,
                "nonlinear_units": a.nonlinear_units,
                "hidden_size": a.hidden_size,
                "num_individual_params": a.num_individual_params,
                "batch_size": a.batch_size,
                "learning_rate": a.learning_rate,
                "weight_decay": a.weight_decay,
                "a_clamp": a.a_clamp,
                "spectral_rho": a.spectral_rho,
                "init_mode": a.init_mode,
                "init_gain": a.init_gain,
                "sample_size": a.sample_size,
                "snapshot_interval": a.snapshot_interval,
                "num_snapshots": int(W_stack.shape[0]),
                "best_test_accuracy_per_task": best_test,
                "mean_best_test_accuracy": float(np.mean(list(best_test.values()))),
                "best_test_loss_per_task": best_test_loss,
                "mean_best_test_loss": float(np.mean(list(best_test_loss.values()))),
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
