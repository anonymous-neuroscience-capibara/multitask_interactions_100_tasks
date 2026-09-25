#!/usr/bin/env python
"""Fine-tune ONLY a new p_vector row for a brand-new held-out task.

Loads a trained FOUNDATION checkpoint (all shared projections p2A/p2W/p2h/p2C/
p2D + the 98 trained p_vector rows), appends ONE new p_vector row for a novel
held-out task, FREEZES everything except that single new row, and fits it on a
handful of trials of the new task.

This asks: are the shared "dynamical primitives" general enough that a genuinely
new task can be solved just by finding the right point in the 16-D individual-
parameter (p_vector) space -- learning nothing else?

Because only the new p_vector row is updated (old rows masked, all shared params
frozen), the 98 pretrained tasks are BIT-FOR-BIT unchanged: forgetting is zero
by construction, so no forgetting check is needed.

Held-out tasks (--task; defined in foundation_tasks/heldout_tasks.py):
    DelayRotate90/45/135 : report the remembered direction rotated by a fixed
                           novel angle (Pro = 0, Anti = 180)
    AngleReflect         : report the MIRRORED direction (theta -> -theta)
    AngleAverage         : two sequential cues -> report their circular mean
    ScalarInvert         : analog level s -> respond 1 - s after a delay
    FreqDouble           : SineGen's cue, oscillate at DOUBLE the cued omega

p_vector init for the new row (--init-source):
    random        : torch.empty(1, dp).uniform_(-1, 1)
    <task name>   : copy an existing trained task's row (e.g. DelayPro)

Example (defaults to this repo's ortho_run M64/N4/P16 checkpoint):
    python "foundation model/finetune_new_task.py" \\
        --task AngleReflect --init-source DelayPro --sample-size 200
"""

import argparse
import copy
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
from hierachical_model_task.model import HierarchicalPLRNN
from hierachical_model_task.utils import HiearchicalModelConfig
from hierachical_model_task.rnn_model import compute_accuracies, normalized_mse_loss
import foundation_tasks as ft
from foundation_tasks.heldout_tasks import HELDOUT_TASKS


DEFAULT_CKPT = (
    ROOT
    / "data"
    / "ortho_run"
    / "foundation_joint_M64_N4_P16_s200_initorthogonal_spec0.0_seed0"
)


def _test_loss(model, loader, device, task_loss_channels):
    """Mean normalized_mse_loss over a loader (no grad)."""
    model.eval()
    tot, nb = 0.0, 0
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
            tot += normalized_mse_loss(
                mo, mt, masks, task_ids, task_loss_channels
            ).item()
            nb += 1
    return tot / max(nb, 1)


def finetune(
    ckpt_dir,
    task_name,
    init_source,
    sample_size,
    n_test_trials,
    num_epochs,
    lr,
    batch_size,
    finetune_seed,
    eval_every,
    use_gpu,
    output_dir,
):
    if task_name not in HELDOUT_TASKS:
        raise ValueError(
            f"Unknown held-out task '{task_name}'. Choices: {list(HELDOUT_TASKS)}"
        )
    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    meta = json.loads((ckpt_dir / "metadata.json").read_text())

    # --- assemble tasks in the SAME order as the checkpoint's p_vector rows ---
    trained_names = meta["task_names"]
    missing = [n for n in trained_names if n not in ft.FOUNDATION_TASKS_MAP]
    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoint tasks are not in the current battery "
            f"(e.g. {missing[:3]}). This checkpoint predates the current "
            f"FOUNDATION_TASKS_MAP; row<->task alignment would be wrong."
        )
    new_task = HELDOUT_TASKS[task_name](ft.duration_params)
    all_tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in trained_names] + [new_task]
    new_idx = len(all_tasks) - 1

    # --- config identical to how the checkpoint was trained -------------------
    args = HiearchicalModelConfig(tasks=all_tasks)
    args.num_individual_params = meta["num_individual_params"]
    args.nonlinear_units = meta["nonlinear_units"]
    args.hidden_size = meta["hidden_size"]
    args.a_clamp = meta.get("a_clamp", 0.0)
    args.hierarchisation = "all"
    args.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    # this run was spec0.0 (no spectral cap), so no cap is installed; the
    # orthogonal init was baked into p2W at train time and is restored on load.

    old_p = state["p_vector"]
    old_n, dp = old_p.shape
    assert old_n == len(
        trained_names
    ), f"checkpoint p_vector rows ({old_n}) != trained tasks ({len(trained_names)})"

    # --- initialize the new task's p_vector row -------------------------------
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    if init_source == "random":
        new_row = torch.empty(1, dp).uniform_(-1, 1)
        resolved = "random"
    else:
        if init_source not in trained_names:
            raise ValueError(
                f"--init-source '{init_source}' is not a trained task. "
                f"Use 'random' or one of: {trained_names[:6]}..."
            )
        src = trained_names.index(init_source)
        new_row = old_p[src : src + 1].clone()
        resolved = init_source
    print(f"held-out task: {task_name} | new p_vector row initialized from: {resolved}")

    state["p_vector"] = torch.cat([old_p, new_row], dim=0)
    old_noise = state["noise_cov"]
    state["noise_cov"] = torch.cat(
        [old_noise, torch.zeros(1, *old_noise.shape[1:])], dim=0
    )

    model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
    model.load_state_dict(state)
    model.to(args.device)

    # --- freeze everything, unfreeze p_vector, mask all but the new row -------
    for p in model.parameters():
        p.requires_grad = False
    model.p_vector.requires_grad = True

    def _mask_old(grad, n_old=old_n):
        grad[:n_old] = 0  # only the new (last) row keeps its gradient
        return grad

    model.p_vector.register_hook(_mask_old)
    optimizer = torch.optim.Adam([model.p_vector], lr=lr)

    # --- data: only the new task; fixed trials so train/test are reproducible -
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    train_ds = HierarchicalTasksDataset(
        all_tasks,
        n_trials=sample_size,
        task_indices=[new_idx] * sample_size,
        fixed=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_ds = HierarchicalTasksDataset(
        all_tasks,
        n_trials=n_test_trials,
        task_indices=[new_idx] * n_test_trials,
        fixed=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    task_loss_channels = {
        i: t.loss_channels
        for i, t in enumerate(all_tasks)
        if getattr(t, "loss_channels", None) is not None
    }

    # sanity: verify only the new row will change
    p_before = model.p_vector.detach().clone()

    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    hist = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    best_acc, best_epoch, best_state = -1.0, None, None

    for epoch in range(num_epochs):
        model.train()
        ep_loss = 0.0
        for inputs, targets, masks, task_ids in train_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            masks, task_ids = masks.to(args.device), task_ids.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs, task_ids)
            me = masks.unsqueeze(-1).expand_as(outputs)
            mo = outputs * me
            if targets.dim() == 2:
                mt = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1) * me
            else:
                mt = targets * me
            loss = normalized_mse_loss(mo, mt, masks, task_ids, task_loss_channels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        train_loss = ep_loss / len(train_loader)

        if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            tr_acc = compute_accuracies(
                model, train_loader, args.device, len(all_tasks), tasks=all_tasks
            ).get(new_idx, 0.0)
            te_acc = compute_accuracies(
                model, test_loader, args.device, len(all_tasks), tasks=all_tasks
            ).get(new_idx, 0.0)
            te_loss = _test_loss(model, test_loader, args.device, task_loss_channels)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(train_loss)
            hist["test_loss"].append(te_loss)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)
            if te_acc > best_acc:
                best_acc, best_epoch = te_acc, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch + 1:4d} | train_loss {train_loss:.4f} "
                    f"| test_loss {te_loss:.4f} | train_acc {tr_acc:.3f} "
                    f"| test_acc {te_acc:.3f} | best {best_acc:.3f}"
                )

    # --- sanity check: only the new row moved --------------------------------
    p_after = model.p_vector.detach()
    moved_old = (p_after[:old_n] - p_before[:old_n]).abs().max().item()
    moved_new = (p_after[old_n:] - p_before[old_n:]).abs().max().item()
    print(
        f"\nmax |dp| old rows = {moved_old:.2e} (should be ~0), "
        f"new row = {moved_new:.2e}"
    )
    assert moved_old < 1e-6, "old p_vector rows changed -- gradient mask failed!"

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- save -----------------------------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    np.savez(out / "history.npz", **{k: np.array(v) for k, v in hist.items()})
    torch.save(
        {
            "p_init_row": new_row.cpu(),
            "init_source": resolved,
            "new_task_idx": new_idx,
            "final_p_row": p_after[old_n:].cpu(),
            "best_p_row": (
                best_state["p_vector"][old_n:].cpu()
                if best_state is not None
                else p_after[old_n:].cpu()
            ),
        },
        out / "p_vector.pt",
    )
    with open(out / "results.json", "w") as f:
        json.dump(
            {
                "new_task": task_name,
                "init_source": resolved,
                "checkpoint": str(ckpt_dir),
                "sample_size": sample_size,
                "n_test_trials": n_test_trials,
                "num_epochs": num_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "finetune_seed": finetune_seed,
                "best_test_acc": best_acc,
                "best_epoch": best_epoch,
                "final_test_acc": hist["test_acc"][-1] if hist["test_acc"] else 0.0,
                "final_train_acc": hist["train_acc"][-1] if hist["train_acc"] else 0.0,
                "moved_old_rows_max": moved_old,
                "moved_new_row_max": moved_new,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(
        f"\n{task_name} <- {resolved}: best test acc {best_acc:.1%} "
        f"@ epoch {best_epoch}  ->  {out}"
    )
    return hist, best_acc


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune only a new p_vector row on a novel held-out task."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT),
        help="Dir with model.pt + metadata.json (foundation run).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="DelayRotate90",
        choices=list(HELDOUT_TASKS),
        help="Which held-out task to learn.",
    )
    p.add_argument(
        "--init-source",
        type=str,
        default="DelayPro",
        help="'random' or a trained task name to copy the p_vector from.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of training trials of the new task.",
    )
    p.add_argument("--n-test-trials", type=int, default=200)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--finetune-seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    a = p.parse_args()

    out = a.output_dir or str(
        Path(a.checkpoint) / f"finetune_{a.task}_init{a.init_source}"
    )
    print(f"checkpoint: {a.checkpoint}")
    finetune(
        ckpt_dir=a.checkpoint,
        task_name=a.task,
        init_source=a.init_source,
        sample_size=a.sample_size,
        n_test_trials=a.n_test_trials,
        num_epochs=a.num_epochs,
        lr=a.lr,
        batch_size=a.batch_size,
        finetune_seed=a.finetune_seed,
        eval_every=a.eval_every,
        use_gpu=a.use_gpu,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
