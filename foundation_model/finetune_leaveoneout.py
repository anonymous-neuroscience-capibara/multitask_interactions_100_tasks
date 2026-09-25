#!/usr/bin/env python
"""Fine-tune ONLY a new p_vector row for a leave-one-out (LOO) held-out task.

Each LOO checkpoint in data/LeaveOneOutData84/ was trained on the foundation
battery with ONE real task left out (metadata['left_out_task']). This script:

  1. loads the checkpoint (shared p2A/p2W/p2h/p2C/p2D + the trained p_vector rows),
  2. appends ONE new p_vector row for the held-out task, initialized by copying
     a trained task that shares the same OUTPUT TYPE (accuracy_type in ft.TAGS),
  3. FREEZES every weight, unfreezes only p_vector, and masks the gradient so
     ONLY the new (last) row is learned,
  4. trains on --sample-size trials, tests on --n-test-trials trials,
  5. saves the loss + accuracy across ALL epochs to history.npz and history.csv.

Because only the new row moves, the pretrained tasks are bit-for-bit unchanged
(forgetting is zero by construction).

Examples:
    # single checkpoint (loo_0 -> held out NoiseCleaner), auto init source
    python "foundation model/finetune_leaveoneout.py" --loo-index 0

    # sweep every loo_* checkpoint, write one summary row per held-out task
    python "foundation model/finetune_leaveoneout.py" --all
"""

import argparse
import copy
import csv
import json
import re
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

DEFAULT_LOO_DIR = ROOT / "data" / "LeaveOneOutData84"

# Per-epoch history columns (shared by each run's history.csv and the combined
# --all history so the latter is just every run's rows stacked).
HISTORY_FIELDS = (
    "loo_index",
    "new_task",
    "output_type",
    "init_source",
    "epoch",
    "train_loss",
    "test_loss",
    "train_acc",
    "test_acc",
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


def _resolve_init_source(trained_names, held_out_name, init_source, rng=None):
    """Return the trained-task name whose p_vector row seeds the new row."""
    held_type = ft.TAGS[held_out_name]["accuracy_type"]
    same_type = [n for n in trained_names if ft.TAGS[n]["accuracy_type"] == held_type]
    if init_source == "auto":
        if not same_type:
            raise ValueError(
                f"no trained task shares output type '{held_type}' with "
                f"{held_out_name}; pass --init-source explicitly."
            )
        resolved = same_type[0]
    elif init_source == "random-sametype":
        # pick a RANDOM trained task that shares the held-out task's output type
        if not same_type:
            raise ValueError(
                f"no trained task shares output type '{held_type}' with "
                f"{held_out_name}; pass --init-source explicitly."
            )
        r = rng if rng is not None else np.random
        resolved = same_type[int(r.randint(len(same_type)))]
    elif init_source == "random":
        resolved = "random"
    else:
        if init_source not in trained_names:
            raise ValueError(
                f"--init-source '{init_source}' is not 'auto'/'random' or a trained task."
            )
        resolved = init_source
    return resolved, held_type, same_type


def finetune(
    ckpt_dir,
    sample_size,
    n_test_trials,
    num_epochs,
    lr,
    batch_size,
    finetune_seed,
    eval_every,
    init_source,
    use_gpu,
    output_dir=None,
    patience=200,
    verbose=True,
):
    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    meta = json.loads((ckpt_dir / "metadata.json").read_text())

    trained_names = meta["task_names"]
    held_out_name = meta["left_out_task"]

    missing = [n for n in trained_names if n not in ft.FOUNDATION_TASKS_MAP]
    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoint tasks are not in the current battery "
            f"(e.g. {missing[:3]}); row<->task alignment would be wrong."
        )
    if held_out_name not in ft.FOUNDATION_TASKS_MAP:
        raise RuntimeError(f"held-out task '{held_out_name}' not in current battery.")
    if held_out_name in trained_names:
        raise RuntimeError(f"held-out task '{held_out_name}' is in the trained set?!")

    all_tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in trained_names] + [
        ft.FOUNDATION_TASKS_MAP[held_out_name]
    ]
    new_idx = len(all_tasks) - 1

    resolved, held_type, same_type = _resolve_init_source(
        trained_names,
        held_out_name,
        init_source,
        rng=np.random.RandomState(finetune_seed),
    )
    if verbose:
        print(
            f"held-out task: {held_out_name} (output type '{held_type}') | "
            f"trained on {len(trained_names)} tasks"
        )
        print(
            f"new p_vector row initialized from: {resolved} "
            f"({len(same_type)} same-type candidates)"
        )

    # --- config identical to how the checkpoint was trained -------------------
    args = HiearchicalModelConfig(tasks=all_tasks)
    args.num_individual_params = meta["num_individual_params"]
    args.nonlinear_units = meta["nonlinear_units"]
    args.hidden_size = meta["hidden_size"]
    args.a_clamp = meta.get("a_clamp", 0.0)
    args.hierarchisation = "all"
    args.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    old_p = state["p_vector"]
    old_n, dp = old_p.shape
    assert old_n == len(
        trained_names
    ), f"checkpoint p_vector rows ({old_n}) != trained tasks ({len(trained_names)})"

    # --- initialize the new task's p_vector row -------------------------------
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    if resolved == "random":
        new_row = torch.empty(1, dp).uniform_(-1, 1)
    else:
        src = trained_names.index(resolved)
        new_row = old_p[src : src + 1].clone()

    state["p_vector"] = torch.cat([old_p, new_row], dim=0)
    if "noise_cov" in state:
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

    p_before = model.p_vector.detach().clone()

    # --- train ----------------------------------------------------------------
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
    stopped_epoch = num_epochs  # actual epochs run (may be < num_epochs w/ patience)

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
            if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
                print(
                    f"  epoch {epoch + 1:4d} | train_loss {train_loss:.4f} "
                    f"| test_loss {te_loss:.4f} | train_acc {tr_acc:.3f} "
                    f"| test_acc {te_acc:.3f} | best {best_acc:.3f}"
                )
            # early stop: perfect test accuracy, or no improvement for `patience` epochs
            if te_acc >= 1.0:
                stopped_epoch = epoch + 1
                if verbose:
                    print(f"  early stop at epoch {stopped_epoch} (100% test accuracy)")
                break
            if (
                patience
                and best_epoch is not None
                and (epoch + 1) - best_epoch >= patience
            ):
                stopped_epoch = epoch + 1
                if verbose:
                    print(
                        f"  early stop at epoch {stopped_epoch} "
                        f"(no improvement for {patience} epochs; best @ {best_epoch})"
                    )
                break

    # --- sanity check: only the new row moved --------------------------------
    p_after = model.p_vector.detach()
    moved_old = (p_after[:old_n] - p_before[:old_n]).abs().max().item()
    moved_new = (p_after[old_n:] - p_before[old_n:]).abs().max().item()
    if verbose:
        print(
            f"  max |dp| old rows = {moved_old:.2e} (should be ~0), "
            f"new row = {moved_new:.2e}"
        )
    assert moved_old < 1e-6, "old p_vector rows changed -- gradient mask failed!"

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- save -----------------------------------------------------------------
    out = (
        Path(output_dir)
        if output_dir
        else (ckpt_dir / f"finetune_{held_out_name}_init{resolved}")
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")

    # loss + accuracy across epochs -> npz AND csv (easy to plot either way).
    # Each CSV row carries the task-identifying columns so per-task histories
    # can be stacked into one file (see the --all combined history below).
    np.savez(out / "history.npz", **{k: np.array(v) for k, v in hist.items()})
    loo_index = meta.get("left_out_index")
    hist_rows = [
        {
            "loo_index": loo_index,
            "new_task": held_out_name,
            "output_type": held_type,
            "init_source": resolved,
            "epoch": e,
            "train_loss": trl,
            "test_loss": tel,
            "train_acc": tra,
            "test_acc": tea,
        }
        for e, trl, tel, tra, tea in zip(
            hist["epoch"],
            hist["train_loss"],
            hist["test_loss"],
            hist["train_acc"],
            hist["test_acc"],
        )
    ]
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        w.writeheader()
        w.writerows(hist_rows)

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
    result = {
        "new_task": held_out_name,
        "init_source": resolved,
        "output_type": held_type,
        "checkpoint": str(ckpt_dir),
        "loo_index": meta.get("left_out_index"),
        "sample_size": sample_size,
        "n_test_trials": n_test_trials,
        "num_epochs": num_epochs,
        "patience": patience,
        "stopped_epoch": stopped_epoch,
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
        "history_rows": hist_rows,  # per-epoch rows, for the combined --all csv
    }
    with open(out / "results.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history_rows"}, f, indent=2)
    if verbose:
        print(
            f"{held_out_name} <- {resolved}: best test acc {best_acc:.1%} "
            f"@ epoch {best_epoch}  ->  {out}\n"
        )
    return result


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune only a new p_vector row on a LOO held-out task."
    )
    p.add_argument(
        "--loo-dir",
        type=str,
        default=str(DEFAULT_LOO_DIR),
        help="Directory containing the loo_* checkpoints.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--loo-index",
        type=int,
        default=0,
        help="Fine-tune the single loo_<index>_* checkpoint.",
    )
    g.add_argument(
        "--ckpt-name",
        type=str,
        default=None,
        help="Fine-tune this exact checkpoint folder name.",
    )
    g.add_argument(
        "--all", action="store_true", help="Sweep every loo_* checkpoint in --loo-dir."
    )
    p.add_argument(
        "--init-source",
        type=str,
        default="auto",
        help="'auto' (same output type), 'random', or a trained task name.",
    )
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--n-test-trials", type=int, default=256)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early-stop after this many epochs without a test-acc "
        "improvement (0 = disabled).",
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--finetune-seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="With --all: re-run checkpoints that already have a "
        "finetune_*/ result (default is to skip/resume).",
    )
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (single-checkpoint mode only).",
    )
    p.add_argument(
        "--init-map",
        type=str,
        default=None,
        help="CSV with columns 'task,init_source' giving a per-held-out-task "
        "init source (looked up by each checkpoint's left_out_task; overrides "
        "--init-source per checkpoint, falls back to it if a task is missing).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="With --all: write each checkpoint's result to a NEW folder "
        "<output-root>/<checkpoint_name>/ instead of inside the checkpoint, "
        "so the (downloaded) models stay read-only.",
    )
    a = p.parse_args()

    loo_dir = Path(a.loo_dir)

    # optional per-task init map (task -> init_source)
    init_map = {}
    if a.init_map:
        with open(a.init_map, newline="") as f:
            init_map = {row["task"]: row["init_source"] for row in csv.DictReader(f)}
        print(f"loaded init map for {len(init_map)} tasks from {a.init_map}")
    output_root = Path(a.output_root) if a.output_root else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    def _init_for(ckpt_dir):
        """Per-checkpoint init source: --init-map by left_out_task, else --init-source."""
        if not init_map:
            return a.init_source
        held = json.loads((ckpt_dir / "metadata.json").read_text())["left_out_task"]
        return init_map.get(held, a.init_source)

    def _run(ckpt_dir, output_dir=None):
        return finetune(
            ckpt_dir=ckpt_dir,
            sample_size=a.sample_size,
            n_test_trials=a.n_test_trials,
            num_epochs=a.num_epochs,
            lr=a.lr,
            batch_size=a.batch_size,
            finetune_seed=a.finetune_seed,
            eval_every=a.eval_every,
            init_source=_init_for(ckpt_dir),
            use_gpu=a.use_gpu,
            output_dir=output_dir,
            patience=a.patience,
        )

    if a.all:
        # accept both the old naming ("loo_5_M64_...") and the new joint-run
        # naming ("foundation_joint_M64_..._loo5_seed0"); a checkpoint is any
        # subdir that carries both a model.pt and a metadata.json.
        def _loo_key(d):
            m = re.search(r"loo_?(\d+)", d.name)
            s = re.search(r"seed(\d+)", d.name)
            return (
                int(m.group(1)) if m else 0,
                int(s.group(1)) if s else 0,
                d.name,
            )

        folders = sorted(
            (
                d
                for d in loo_dir.iterdir()
                if d.is_dir()
                and (d / "model.pt").exists()
                and (d / "metadata.json").exists()
            ),
            key=_loo_key,
        )

        def _existing_dir(ckpt_dir, odir):
            """Return an already-finetuned output dir for this checkpoint, or None."""
            if odir is not None:  # output-root mode: result lands directly in odir
                return odir if (odir / "results.json").exists() else None
            hits = sorted(ckpt_dir.glob("finetune_*_init*/results.json"))
            return hits[0].parent if hits else None

        def _load_existing(rdir):
            """Reload a finished run so the combined CSVs stay complete on resume."""
            res = json.loads((rdir / "results.json").read_text())
            hist_rows = []
            hp = rdir / "history.csv"
            if hp.exists():
                with open(hp, newline="") as f:
                    hist_rows = list(csv.DictReader(f))
            res["history_rows"] = hist_rows
            return res

        summary, skipped = [], []
        for i, d in enumerate(folders):
            odir = (output_root / d.name) if output_root else None
            done = None if a.no_skip_existing else _existing_dir(d, odir)
            if done is not None:
                res = _load_existing(done)
                summary.append(res)
                print(
                    f"[{i + 1}/{len(folders)}] {d.name} -- already done "
                    f"({res['new_task']} <- {res['init_source']}, "
                    f"best {res['best_test_acc']:.1%}); skipping"
                )
                continue
            print(f"[{i + 1}/{len(folders)}] {d.name} <- init {_init_for(d)}")
            try:
                summary.append(_run(d, output_dir=odir))
            except ValueError as e:
                # e.g. held-out task is the only one of its output type -> no init source
                skipped.append((d.name, str(e)))
                print(f"    SKIP: {e}")
        # one summary row per held-out task
        keys = [
            "loo_index",
            "new_task",
            "output_type",
            "init_source",
            "best_test_acc",
            "best_epoch",
            "final_test_acc",
        ]
        out_base = output_root or loo_dir
        with open(out_base / "finetune_loo_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary:
                w.writerow({k: r[k] for k in keys})
        # combined per-epoch history: every task's full curve, stacked
        with open(out_base / "finetune_loo_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
            w.writeheader()
            for r in summary:
                w.writerows(r["history_rows"])
        mean_acc = (
            np.mean([float(r["best_test_acc"]) for r in summary]) if summary else 0.0
        )
        print(
            f"\nmean best_test_acc over {len(summary)} held-out tasks: {mean_acc:.1%}"
        )
        print(f"summary -> {out_base / 'finetune_loo_summary.csv'}")
        print(f"stacked per-epoch history -> {out_base / 'finetune_loo_history.csv'}")
        if skipped:
            print(f"\nskipped {len(skipped)} checkpoint(s) with no valid init source:")
            for name, msg in skipped:
                print(f"  {name}: {msg}")
    else:
        if a.ckpt_name is not None:
            ckpt_dir = loo_dir / a.ckpt_name
        else:
            matches = sorted(loo_dir.glob(f"loo_{a.loo_index}_*"))
            if not matches:
                raise SystemExit(f"no loo_{a.loo_index}_* under {loo_dir}")
            ckpt_dir = matches[0]
        print(f"checkpoint: {ckpt_dir}")
        _run(ckpt_dir, output_dir=a.output_dir)


if __name__ == "__main__":
    main()
