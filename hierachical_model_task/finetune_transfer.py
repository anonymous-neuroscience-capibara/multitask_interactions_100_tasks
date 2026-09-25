"""Fine-tune leave-one-out (LOO) PLRNN models on their held-out task.

For each excluded task / LOO model seed, this script loads the pretrained LOO
checkpoint, adds a ``p_vector`` row for the excluded task, and fine-tunes a
chosen set of parameters ("conditions") to learn that new task.

NEW-TASK p_vector INITIALIZATION
--------------------------------
The new task's individual parameter (``p_vector`` row) can be initialized in
two ways:
    - "new"        : a random row, torch.empty(1, dp).uniform_(-1, 1)
    - <task_name>  : copied from an already-trained task's p_vector row
                     (transfer the source task's "identity" as a starting point)
We can sweep over ALL trained tasks (+ "new") to study which source helps the
new task converge fastest.

For every (condition, init-source) it records, *for the new task*:
    - training loss per epoch          - training accuracy per epoch
    - test  loss per epoch             - test  accuracy per epoch
and computes ``epochs_to_threshold`` (epochs to reach >90% test/train accuracy),
using the same convention as multi_task_training/utils.py.

SAVING LAYOUT
-------------
    <output>/seed_<s>/<trained_task>/<condition>/<init_source>/
        model.pt       best model weights (by test accuracy during training)
        history.npz    per-epoch train/test loss + accuracy curves (new task)
        results.json   scalars: epochs_to_threshold, best acc/epoch, config, init
        datasets.pt    the exact train & test trials used (reproducibility)
        p_init.pt      the initial p_vector row used + its provenance
    <output>/summary_seed<s>_<tasks>.json   runs from one invocation aggregated

    <condition> folder names:
        p only        -> p_vector
        p + p2C       -> p_p2C
        p + p2D       -> p_p2D
        p + p2C + p2D -> p_p2C_p2D
        p + p2A + p2W -> p_p2A_p2W
    <init_source> is a trained task name (p copied from it) or "new" (random).

By default the init-source sweep applies only to the ``p only`` (p_vector)
condition; other conditions use a single "new" init. Use --sweep-conditions to
change which conditions get the full source sweep.

The finetuning logic mirrors ``finetune_with_best_model`` in
notebooks/transfer_learning.ipynb; the saving style mirrors run_experiments.py.

Example:
    python -m hierachical_model_task.finetune_transfer \\
        --loo-dir "C:/Users/garcias/Downloads/leave_one_out/leave_one_out" \\
        --output-dir hierachical_model_task/results/finetune_transfer \\
        --excluded-tasks ArithAdd --seeds 0 --num-epochs 500
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent and current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from hierachical_model_task.model import HierarchicalPLRNN
from hierachical_model_task.rnn_model import compute_accuracies, normalized_mse_loss
from hierachical_model_task.utils import HiearchicalModelConfig
from hierachical_model_task.run_experiments import (
    TASKS_MAP,
    _get_device,
    _handle_defaults,
)
from tasks.dataset import HierarchicalTasksDataset, collate_fn


# --- Fine-tuning conditions: which parameters to unfreeze ---------------------
# For "p_vector" only the *new* task row is trained (old rows are masked).
CONDITIONS = {
    "p only": ("p_vector",),
    "p + p2C": ("p_vector", "p2C"),
    "p + p2D": ("p_vector", "p2D"),
    "p + p2C + p2D": ("p_vector", "p2C", "p2D"),
    "p + p2A + p2W": ("p_vector", "p2A", "p2W"),
}

# Filesystem-friendly folder name per condition (level 2 of the layout).
COND_SLUG = {
    "p only": "p_vector",
    "p + p2C": "p_p2C",
    "p + p2D": "p_p2D",
    "p + p2C + p2D": "p_p2C_p2D",
    "p + p2A + p2W": "p_p2A_p2W",
}


def epochs_to_threshold(acc_list, threshold=0.9):
    """Number of epochs (1-based) needed to reach acc > threshold.

    Returns np.inf if never reached. Matches the convention in
    multi_task_training/utils.py:epochs_to_threshold.
    """
    for i, acc in enumerate(acc_list):
        if acc > threshold:
            return i + 1  # 1-based epoch count
    return np.inf


def _compute_test_loss(model, loader, device, task_loss_channels):
    """Mean normalized_mse_loss over a dataloader (no grad)."""
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for inputs, targets, masks, task_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            task_ids = task_ids.to(device)

            outputs = model(inputs, task_ids)
            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)
            masked_outputs = outputs * masks_expanded
            if targets.dim() == 2:
                targets_exp = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                masked_targets = targets_exp * masks_expanded
            else:
                masked_targets = targets * masks_expanded

            loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels
            )
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


def _dump_cache(cache):
    """Serialize a fixed dataset's _cache into plain dicts (for torch.save)."""
    return [
        {"inputs": inputs, "targets": targets, "mask": mask, "task_idx": int(task_idx)}
        for inputs, targets, mask, task_idx in cache
    ]


def finetune_condition(
    excluded_task,
    model_seed,
    trainable_params,
    init_source,
    loo_dir,
    hierarchisation="all",
    num_epochs=500,
    lr=1e-3,
    sample_size=50,
    n_test_trials=200,
    finetune_seed=42,
    eval_every=1,
    use_gpu=False,
    verbose=True,
    exclude_init_sources=None,
    n_alltask_trials=100,
    batch_size=64,
):
    """Fine-tune one LOO model on its excluded task, for one condition and one
    p_vector init source.

    init_source :
        "new"          -> random row, uniform(-1, 1)
        "random_peer"  -> copy the p_vector of ONE peer task chosen at random
                          (from trained tasks minus `exclude_init_sources`);
                          the pick is seeded per (held-out task, condition), so
                          it's reproducible but independent across conditions.
        <task name>    -> copy that specific trained task's p_vector row.

    Returns a dict with per-epoch curves, the best-by-test-accuracy model
    state_dict, the exact train/test caches, the init p-row, and scalars.
    """
    loo_dir = Path(loo_dir)
    seed_dir = loo_dir / excluded_task / f"seed_{model_seed}"
    exp_dirs = list(seed_dir.glob("p*_n*_h*_s*"))
    if not exp_dirs:
        raise FileNotFoundError(f"No LOO experiment found in {seed_dir}")
    exp_dir = exp_dirs[0]

    with open(exp_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    pretrained_state = torch.load(exp_dir / "model.pt", map_location="cpu")

    # Build config for the FULL task set (trained tasks + the excluded one last)
    trained_names = metadata["task_names"]
    all_names = trained_names + [excluded_task]
    all_tasks = [TASKS_MAP[name] for name in all_names]
    new_task_idx = len(all_tasks) - 1

    args = HiearchicalModelConfig(tasks=all_tasks)
    args.num_individual_params = metadata["num_individual_params"]
    args.nonlinear_units = metadata["nonlinear_units"]
    args.hidden_size = metadata["hidden_size"]
    args.hierarchisation = hierarchisation
    args.num_epochs = num_epochs
    args.individual_learning_rate = lr
    args.learning_rate = lr
    args.use_gpu = use_gpu
    args = _get_device(args)
    args = _handle_defaults(args)

    old_p = pretrained_state["p_vector"]
    old_n, dp = old_p.shape[0], old_p.shape[1]
    assert old_n == len(
        trained_names
    ), f"p_vector rows ({old_n}) != trained tasks ({len(trained_names)})"

    # --- Initialize the NEW task's p_vector row ---
    # Seed first so the random "new" draw is reproducible.
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    resolved_source = (
        init_source  # which task's p was actually used (for "random_peer")
    )
    if init_source == "new":
        new_p_row = torch.empty(1, dp).uniform_(-1, 1)
    elif init_source == "random_peer":
        # Randomly pick one eligible peer task's p_vector as the init.
        exclude = set(exclude_init_sources or [])
        candidates = [t for t in trained_names if t not in exclude]
        if not candidates:
            raise ValueError(
                f"No eligible peer p_vectors for '{excluded_task}' after "
                f"excluding {sorted(exclude)}."
            )
        # Dedicated RNG seeded per (held-out task, condition) so each condition
        # draws its OWN reproducible random peer, independent across conditions.
        # Uses hashlib (not hash()) so it's stable across processes/runs.
        cond_key = f"{excluded_task}|{'+'.join(trainable_params)}"
        offset = int(hashlib.md5(cond_key.encode()).hexdigest(), 16)
        rs = np.random.RandomState((finetune_seed + offset) % (2**32))
        resolved_source = candidates[int(rs.randint(len(candidates)))]
        src_idx = trained_names.index(resolved_source)
        new_p_row = old_p[src_idx : src_idx + 1].clone()
    else:
        if init_source not in trained_names:
            raise ValueError(
                f"init_source '{init_source}' is not a trained task of "
                f"'{excluded_task}'. Available: {trained_names} "
                f"(or 'new' / 'random_peer')."
            )
        src_idx = trained_names.index(init_source)
        new_p_row = old_p[src_idx : src_idx + 1].clone()

    pretrained_state["p_vector"] = torch.cat([old_p, new_p_row], dim=0)
    old_noise = pretrained_state["noise_cov"]
    pretrained_state["noise_cov"] = torch.cat(
        [old_noise, torch.zeros(1, *old_noise.shape[1:])], dim=0
    )

    model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
    model.load_state_dict(pretrained_state)
    model.to(args.device)

    # Freeze everything, then unfreeze requested params
    for param in model.parameters():
        param.requires_grad = False
    unfrozen = []
    for name, param in model.named_parameters():
        if name in trainable_params:
            param.requires_grad = True
            unfrozen.append(name)
    if not unfrozen:
        available = [n for n, _ in model.named_parameters()]
        raise ValueError(f"None of {trainable_params} found. Available: {available}")

    # For p_vector: mask gradients so only the NEW task row is updated
    if "p_vector" in trainable_params:

        def _mask_old_rows(grad, n_old=old_n):
            grad[:n_old] = 0
            return grad

        model.p_vector.register_hook(_mask_old_rows)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: torch.tensor(0.999) ** epoch
    )

    # --- Data loaders: only the NEW task. Re-seed so the train/test trials are
    # identical across every init source / condition (fair comparison). Both
    # datasets are fixed=True so the exact trials can be saved to disk. ---
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    train_indices = [new_task_idx] * sample_size
    train_dataset = HierarchicalTasksDataset(
        all_tasks, n_trials=sample_size, task_indices=train_indices, fixed=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_indices = [new_task_idx] * n_test_trials
    test_dataset = HierarchicalTasksDataset(
        all_tasks, n_trials=n_test_trials, task_indices=test_indices, fixed=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    task_loss_channels = {}
    for i, task in enumerate(all_tasks):
        if getattr(task, "loss_channels", None) is not None:
            task_loss_channels[i] = task.loss_channels

    # --- All-task test set (forgetting check): n_alltask_trials per task. ---
    # Built once and reused for the before/after snapshots. Set n_alltask_trials
    # to 0 to skip (e.g. to save time).
    all_task_loader = None
    if n_alltask_trials and n_alltask_trials > 0:
        torch.manual_seed(finetune_seed)
        np.random.seed(finetune_seed)
        all_idx = []
        for tid in range(len(all_tasks)):
            all_idx.extend([tid] * n_alltask_trials)
        all_task_ds = HierarchicalTasksDataset(
            all_tasks, n_trials=len(all_idx), task_indices=all_idx, fixed=True
        )
        all_task_loader = DataLoader(
            all_task_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    def _eval_all_tasks():
        if all_task_loader is None:
            return None
        accs = compute_accuracies(
            model, all_task_loader, args.device, len(all_tasks), tasks=all_tasks
        )
        return {all_names[t]: float(accs.get(t, 0.0)) for t in range(len(all_names))}

    # Snapshot all-task accuracy BEFORE fine-tuning (new p row + frozen rest).
    all_task_acc_before = _eval_all_tasks()

    # Re-seed so the training data shuffle is deterministic regardless of the
    # all-task eval above (keeps runs reproducible and conditions comparable).
    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)

    # Per-epoch histories (new task). epoch_index[k] = 0-based epoch of entry k.
    train_losses, test_losses, train_accs, test_accs, epoch_index = [], [], [], [], []
    best_test_acc, best_epoch, best_state_dict = -1.0, None, None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets, masks, task_ids in train_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            masks = masks.to(args.device)
            task_ids = task_ids.to(args.device)

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
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)

        if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            train_acc = compute_accuracies(
                model, train_loader, args.device, len(all_tasks), tasks=all_tasks
            ).get(new_task_idx, 0.0)
            test_acc = compute_accuracies(
                model, test_loader, args.device, len(all_tasks), tasks=all_tasks
            ).get(new_task_idx, 0.0)
            test_loss = _compute_test_loss(
                model, test_loader, args.device, task_loss_channels
            )

            epoch_index.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1  # 1-based
                best_state_dict = copy.deepcopy(model.state_dict())

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(
                    f"        epoch {epoch + 1:3d} | train_loss {train_loss:.4f} "
                    f"| test_loss {test_loss:.4f} | train_acc {train_acc:.3f} "
                    f"| test_acc {test_acc:.3f} | best {best_test_acc:.3f}"
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Snapshot all-task accuracy AFTER fine-tuning (best model loaded).
    all_task_acc_after = _eval_all_tasks()

    return {
        "excluded_task": excluded_task,
        "model_seed": model_seed,
        "trainable_params": list(trainable_params),
        "unfrozen": unfrozen,
        "init_source": init_source,
        "resolved_source": resolved_source,
        "all_names": all_names,
        "new_task_idx": new_task_idx,
        "num_epochs": num_epochs,
        "lr": lr,
        "sample_size": sample_size,
        "n_test_trials": n_test_trials,
        "finetune_seed": finetune_seed,
        "eval_every": eval_every,
        "hierarchisation": hierarchisation,
        # per-epoch curves (new task)
        "epoch_index": epoch_index,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        # summary scalars
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_test_acc": test_accs[-1] if test_accs else 0.0,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        # all-task accuracy snapshots (forgetting check); None if disabled
        "all_task_acc_before": all_task_acc_before,
        "all_task_acc_after": all_task_acc_after,
        # artifacts to persist
        "best_state_dict": best_state_dict,
        "p_init_row": new_p_row.detach().cpu(),
        "train_cache": train_dataset._cache,
        "test_cache": test_dataset._cache,
        "train_task_indices": train_indices,
        "test_task_indices": test_indices,
    }


def _resolve_init_sources(requested, trained_names, exclude=None):
    """Expand an --init-sources request into concrete sources for one task.

    `exclude` task names are dropped from the result (e.g. tasks whose
    p_vector you don't want to seed from). "new" is only removed if it is
    explicitly listed in `exclude`.
    """
    exclude = set(exclude or [])
    if requested == ["all"]:
        resolved = list(trained_names) + ["new"]
    else:
        resolved = []
        for s in requested:
            if s == "all":
                resolved.extend(list(trained_names))
            else:
                resolved.append(s)  # "new" or a task name; validated downstream
    # drop excluded sources, de-duplicate, preserve order
    seen, out = set(), []
    for s in resolved:
        if s in exclude or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _finetune_and_save(job):
    """Worker: fine-tune one (task, seed, condition, init_source) job and save
    its artifacts. Returns a summary-row dict, or None on failure.

    Defined at module level and takes a single picklable dict so it can be
    dispatched to a multiprocessing.Pool (spawn-safe on Windows).
    """
    # Limit intra-op threads so parallel workers don't oversubscribe cores.
    try:
        torch.set_num_threads(max(1, int(job.get("torch_threads", 1))))
    except Exception:
        pass

    excluded = job["excluded"]
    seed = job["seed"]
    cond_name = job["cond_name"]
    slug = job["slug"]
    source = job["source"]
    threshold = job["threshold"]
    output_dir = Path(job["output_dir"])

    tag = f"{excluded}/{slug}/{source} (seed {seed})"
    try:
        res = finetune_condition(
            excluded_task=excluded,
            model_seed=seed,
            trainable_params=job["params"],
            init_source=source,
            loo_dir=job["loo_dir"],
            hierarchisation=job["hierarchisation"],
            num_epochs=job["num_epochs"],
            lr=job["lr"],
            sample_size=job["sample_size"],
            n_test_trials=job["n_test_trials"],
            finetune_seed=job["finetune_seed"],
            eval_every=job["eval_every"],
            use_gpu=job["use_gpu"],
            verbose=job["verbose"],
            exclude_init_sources=job.get("exclude_init_sources"),
            n_alltask_trials=job.get("n_alltask_trials", 100),
            batch_size=job.get("batch_size", 64),
        )
    except Exception as e:
        print(f"  SKIP {tag} ({e})")
        return None

    resolved_source = res["resolved_source"]

    # Mean accuracy over the OTHER (already-trained) tasks = forgetting check.
    def _trained_mean(d):
        if not d:
            return None
        others = [v for k, v in d.items() if k != excluded]
        return float(np.mean(others)) if others else None

    all_acc_before = res["all_task_acc_before"]
    all_acc_after = res["all_task_acc_after"]
    trained_mean_before = _trained_mean(all_acc_before)
    trained_mean_after = _trained_mean(all_acc_after)

    ei = res["epoch_index"]
    ett_test = epochs_to_threshold(res["test_accs"], threshold)
    ett_train = epochs_to_threshold(res["train_accs"], threshold)
    ett_test_epoch = np.inf if ett_test == np.inf else ei[int(ett_test) - 1] + 1
    ett_train_epoch = np.inf if ett_train == np.inf else ei[int(ett_train) - 1] + 1

    # Layout: seed_<s>/<task>/<condition>/<init_source>/  (seed = least granular)
    run_dir = output_dir / f"seed_{seed}" / excluded / slug / source
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) fine-tuned weights (best by test accuracy)
    torch.save(res["best_state_dict"], run_dir / "model.pt")

    # 2) per-epoch curves
    np.savez(
        run_dir / "history.npz",
        epoch_index=np.array(res["epoch_index"]),
        train_losses=np.array(res["train_losses"]),
        test_losses=np.array(res["test_losses"]),
        train_accs=np.array(res["train_accs"]),
        test_accs=np.array(res["test_accs"]),
    )

    # 3) the exact data used (train + test trials). Biggest file on disk and
    #    fully reproducible from finetune_seed, so it can be skipped to save
    #    space/inodes via --no-datasets.
    if job.get("save_datasets", True):
        torch.save(
            {
                "train_set": _dump_cache(res["train_cache"]),
                "test_set": _dump_cache(res["test_cache"]),
                "train_task_indices": list(res["train_task_indices"]),
                "test_task_indices": list(res["test_task_indices"]),
                "task_names": res["all_names"],
                "new_task_idx": res["new_task_idx"],
                "finetune_seed": job["finetune_seed"],
            },
            run_dir / "datasets.pt",
        )

    # 4) the initial p_vector row + provenance
    #    init_source is the requested mode ("new"/"random_peer"/<task>);
    #    resolved_source is the task actually copied (== the random pick).
    torch.save(
        {
            "init_source": source,
            "resolved_source": resolved_source,
            "p_init_row": res["p_init_row"],
            "new_task_idx": res["new_task_idx"],
        },
        run_dir / "p_init.pt",
    )

    # 5) scalars + config (JSON; inf -> null)
    def _j(x):
        return None if x == np.inf else int(x)

    results_json = {
        "trained_task": excluded,
        "condition": cond_name,
        "condition_slug": slug,
        "init_source": source,
        "resolved_init_source": resolved_source,
        "model_seed": seed,
        "trainable_params": res["trainable_params"],
        "unfrozen": res["unfrozen"],
        "threshold": threshold,
        "epochs_to_threshold_test": _j(ett_test_epoch),
        "epochs_to_threshold_train": _j(ett_train_epoch),
        "best_test_acc": res["best_test_acc"],
        "best_epoch": res["best_epoch"],
        "final_test_acc": res["final_test_acc"],
        "final_train_acc": res["final_train_acc"],
        # all-task (forgetting) snapshots: per-task dicts + mean over the
        # already-trained tasks before vs after fine-tuning.
        "all_task_acc_before": all_acc_before,
        "all_task_acc_after": all_acc_after,
        "trained_mean_acc_before": trained_mean_before,
        "trained_mean_acc_after": trained_mean_after,
        "config": {
            "num_epochs": job["num_epochs"],
            "lr": job["lr"],
            "sample_size": job["sample_size"],
            "batch_size": job["batch_size"],
            "n_test_trials": job["n_test_trials"],
            "finetune_seed": job["finetune_seed"],
            "eval_every": job["eval_every"],
            "hierarchisation": job["hierarchisation"],
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(
        f"  done {tag} | best_test_acc {res['best_test_acc']:.3f} "
        f"@epoch {res['best_epoch']} | epochs_to_{int(threshold * 100)}%: "
        f"test={ett_test_epoch} train={ett_train_epoch}"
    )
    return {
        "trained_task": excluded,
        "condition": cond_name,
        "init_source": source,
        "resolved_init_source": resolved_source,
        "seed": seed,
        "epochs_to_threshold_test": _j(ett_test_epoch),
        "epochs_to_threshold_train": _j(ett_train_epoch),
        "best_test_acc": res["best_test_acc"],
        "best_epoch": res["best_epoch"],
        "final_test_acc": res["final_test_acc"],
        "trained_mean_acc_before": trained_mean_before,
        "trained_mean_acc_after": trained_mean_after,
    }


def _resolve_num_workers(requested, n_jobs):
    """Decide worker count. requested=0 -> auto-detect (cluster-aware).

    On SLURM we use the full allocation (cpus-1) with no cap; locally we cap
    at 8 to avoid hogging a workstation.
    """
    if requested and requested > 0:
        return min(requested, n_jobs)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        cpu = int(os.environ["SLURM_CPUS_PER_TASK"])
        return max(1, min(cpu - 1, n_jobs))  # no cap on the cluster
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 1, n_jobs, 8))  # cap local workstation


def run(
    loo_dir,
    output_dir,
    excluded_tasks,
    seeds,
    conditions,
    sweep_conditions,
    init_sources,
    exclude_init_sources,
    num_epochs,
    lr,
    sample_size,
    n_test_trials,
    threshold,
    finetune_seed,
    eval_every,
    hierarchisation,
    use_gpu,
    num_workers,
    n_alltask_trials,
    batch_size,
    save_datasets,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the full list of independent jobs.
    jobs = []
    for excluded in excluded_tasks:
        trained_names = [n for n in TASKS_MAP.keys() if n != excluded]
        for seed in seeds:
            for cond_name in conditions:
                if cond_name in sweep_conditions:
                    sources = _resolve_init_sources(
                        init_sources, trained_names, exclude_init_sources
                    )
                else:
                    sources = ["new"]
                for source in sources:
                    jobs.append(
                        {
                            "excluded": excluded,
                            "seed": seed,
                            "cond_name": cond_name,
                            "params": CONDITIONS[cond_name],
                            "slug": COND_SLUG[cond_name],
                            "source": source,
                            "exclude_init_sources": list(exclude_init_sources),
                            "loo_dir": str(loo_dir),
                            "output_dir": str(output_dir),
                            "hierarchisation": hierarchisation,
                            "num_epochs": num_epochs,
                            "lr": lr,
                            "sample_size": sample_size,
                            "batch_size": batch_size,
                            "n_test_trials": n_test_trials,
                            "finetune_seed": finetune_seed,
                            "eval_every": eval_every,
                            "use_gpu": use_gpu,
                            "threshold": threshold,
                            "n_alltask_trials": n_alltask_trials,
                            "save_datasets": save_datasets,
                        }
                    )

    workers = _resolve_num_workers(num_workers, len(jobs))
    print(f"Total jobs: {len(jobs)} | parallel workers: {workers}")

    # Per-worker thread budget (avoid oversubscription); verbose only if serial.
    cpu = os.cpu_count() or 4
    threads_per_worker = max(1, cpu // max(1, workers))
    for j in jobs:
        j["torch_threads"] = threads_per_worker
        j["verbose"] = workers == 1

    if workers == 1:
        results = [_finetune_and_save(j) for j in jobs]
    else:
        # GPU + multiprocessing don't mix well here; workers share CPU.
        with Pool(processes=workers) as pool:
            results = pool.map(_finetune_and_save, jobs)

    summary = [r for r in results if r is not None]

    # Unique summary filename per invocation so parallel array elements sharing
    # this output_dir (different seeds/tasks) don't clobber each other. Aggregate
    # analyses should glob "summary_*.json" (or the per-run results.json files).
    seeds_tag = "-".join(str(s) for s in seeds)
    tasks_tag = "-".join(excluded_tasks)
    tag = f"seed{seeds_tag}_{tasks_tag}"
    if len(tag) > 80:
        tag = f"seed{seeds_tag}_" + hashlib.md5(tasks_tag.encode()).hexdigest()[:10]
    summary_path = output_dir / f"summary_{tag}.json"

    with open(summary_path, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
                "runs": summary,
            },
            f,
            indent=2,
        )
    print(f"\nDone. Summary -> {summary_path} ({len(summary)} runs).")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LOO PLRNN models on their excluded task, sweeping "
        "the new-task p_vector init over trained tasks (+ random). Records "
        "per-epoch loss/accuracy and epochs-to-threshold."
    )
    parser.add_argument(
        "--loo-dir",
        type=str,
        default="C:/Users/garcias/Downloads/leave_one_out/leave_one_out",
        help="Root dir: <excluded_task>/seed_<s>/p*_n*_h*_s*/model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "results" / "finetune_transfer"),
    )
    parser.add_argument(
        "--excluded-tasks",
        nargs="+",
        default=None,
        help="Tasks to fine-tune (default: all tasks in TASKS_MAP).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS.keys()),
        help=f"Which conditions to run. Choices: {list(CONDITIONS.keys())}",
    )
    parser.add_argument(
        "--sweep-conditions",
        nargs="+",
        default=["p only"],
        help="Conditions that get the full p-init source sweep. Others use "
        "a single 'new' (random) init. Pass 'all' to sweep every condition.",
    )
    parser.add_argument(
        "--init-sources",
        nargs="+",
        default=["all"],
        help="p_vector init sources for swept conditions: 'all' (every trained "
        "task + 'new'), or an explicit list e.g. 'new' DelayPro CatAnti.",
    )
    parser.add_argument(
        "--exclude-init-sources",
        nargs="+",
        default=[],
        help="Task names to never use as a p_vector init source (filtered out "
        "of the swept sources). E.g. CopyTask ArithAdd ArithMultiply.",
    )
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DataLoader batch size for train/test/all-task loaders.",
    )
    parser.add_argument("--n-test-trials", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--finetune-seed",
        type=int,
        default=42,
        help="Seed for random 'new' init + train/test data; fixed across init "
        "sources and conditions for a fair, reproducible comparison.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate loss+acc every N epochs. Default 1 = every epoch.",
    )
    parser.add_argument(
        "--hierarchisation", type=str, default="all", choices=["all", "AW", "CD"]
    )
    parser.add_argument(
        "--alltask-trials",
        type=int,
        default=100,
        help="Trials per task for the before/after all-task accuracy snapshot "
        "(forgetting check). 0 disables it.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel processes for fine-tuning jobs. 0 = auto-detect "
        "(min(cpu-1, n_jobs, 8)); 1 = sequential.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--no-datasets",
        action="store_true",
        help="Skip saving datasets.pt per run (biggest file; reproducible from "
        "--finetune-seed). Use to save disk space / inodes.",
    )
    args = parser.parse_args()

    excluded_tasks = args.excluded_tasks or list(TASKS_MAP.keys())

    for c in args.conditions:
        if c not in CONDITIONS:
            raise ValueError(f"Unknown condition '{c}'. Choices: {list(CONDITIONS)}")
    if args.sweep_conditions == ["all"]:
        sweep_conditions = set(args.conditions)
    else:
        sweep_conditions = set(args.sweep_conditions)
        for c in sweep_conditions:
            if c not in CONDITIONS:
                raise ValueError(
                    f"Unknown --sweep-conditions '{c}'. Choices: {list(CONDITIONS)}"
                )

    run(
        loo_dir=args.loo_dir,
        output_dir=args.output_dir,
        excluded_tasks=excluded_tasks,
        seeds=args.seeds,
        conditions=args.conditions,
        sweep_conditions=sweep_conditions,
        init_sources=args.init_sources,
        exclude_init_sources=args.exclude_init_sources,
        num_epochs=args.num_epochs,
        lr=args.lr,
        sample_size=args.sample_size,
        n_test_trials=args.n_test_trials,
        threshold=args.threshold,
        finetune_seed=args.finetune_seed,
        eval_every=args.eval_every,
        hierarchisation=args.hierarchisation,
        use_gpu=args.use_gpu,
        num_workers=args.num_workers,
        n_alltask_trials=args.alltask_trials,
        batch_size=args.batch_size,
        save_datasets=not args.no_datasets,
    )


if __name__ == "__main__":
    main()
