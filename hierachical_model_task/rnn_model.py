import torch
import torch.nn.functional as F
import numpy as np

THRESHOLD = (
    36 * np.pi / 180
)  # 36 degrees in radians (Yang-2019 standard, relaxed 1.8x from 20 deg);
#    used for single-timestep angular tasks (not CopyTask)

BINARY_THRESHOLD = 0.5
BINARY_CHANNEL = 3  # Dedicated channel for binary (go/nogo) responses
FIX_CHANNEL = 0  # Fixation channel (Yang-2019's z_fix); scored via apply_fixation_gate
FIXATION_GATE = True  # Include the fixation channel in accuracy (Yang 2019). Set
#    False to recover the pre-adoption metric, which ignored channel 0 entirely.
#    Verified equivalent on the M64 N16/P16 champions (seeds 2/3/4): the battery
#    mean is 94.69% either way, because channel 0 is supervised only over the
#    response window, where 98/100 tasks want it released.
NON_ANGULAR_THRESHOLD = 0.1  # scalar/vector |err| tolerance (was 0.05); fallback only

# Scale-relative accuracy for the continuous (scalar/vector) tasks: a trial is
# correct if the readout error is within TAU_SIGMA * sigma_task, where sigma_task
# is the std of that task's targets. TAU_SIGMA = 0.35 is angular-anchored -- the
# angular tasks accept within 36 deg and a uniform direction has std ~104 deg, so
# 36/104 ~= 0.35 (the same tolerance-to-answer-spread ratio). This holds the
# continuous tasks to the same strictness as the angular ones instead of the
# scale-blind fixed 0.1. See foundation_model/metric_proposal/accuracy_proposal.md.
TAU_SIGMA = 0.35


def compute_scalar_vector_tol(tasks, tau=TAU_SIGMA, n_trials=500):
    """Per-task tau*sigma tolerance for the scalar/vector tasks.

    sigma = std of the task's targets at the scored (last-masked) step, estimated
    from n_trials samples; for vector tasks it is the total spread
    sqrt(sum_c Var[target_c]) so it matches the Euclidean error. Returns
    {task_id: tolerance}; non-continuous tasks are absent (callers fall back to
    NON_ANGULAR_THRESHOLD). The numpy/torch RNG state is snapshotted and restored
    so this sampling does not perturb training reproducibility.
    """
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        tol = {}
        for i, task in enumerate(tasks):
            is_scalar = getattr(task, "is_scalar_task", False)
            is_vector = getattr(task, "is_vector_task", False)
            if not (is_scalar or is_vector):
                continue
            channels = task.loss_channels if is_vector else [BINARY_CHANNEL]
            vals = []
            for _ in range(n_trials):
                _, tgt, mask = task.generate_trial()
                midx = torch.where(mask > 0)[0]
                if len(midx) == 0:
                    continue
                row = tgt if tgt.dim() == 1 else tgt[midx[-1]]
                vals.append(row[channels].numpy())
            vals = np.asarray(vals)  # (n, len(channels))
            sigma = float(np.sqrt(np.sum(np.var(vals, axis=0))))
            tol[i] = tau * sigma
        return tol
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def normalized_mse_loss(
    masked_outputs, masked_targets, masks, task_ids=None, task_loss_channels=None
):
    """
    Compute MSE loss normalized by number of masked timesteps per sample.

    task_loss_channels: dict mapping task_id -> list of channel indices to use for loss.
                        Supports negative indexing (e.g. [-1] for last channel).
                        If a task is not in the dict, all channels except the first are used.
    """
    batch, _, C = masked_outputs.shape
    n_masked = masks.sum(dim=1).clamp(min=1)  # (batch,)

    # Per-element MSE: (batch, T, C)
    full_mse = F.mse_loss(masked_outputs, masked_targets, reduction="none")

    # Build per-sample channel mask: (batch, C)
    channel_mask = torch.zeros(batch, C, device=masked_outputs.device)
    # Always include channel 0 (fixation)
    channel_mask[:, 0] = 1.0
    if task_ids is not None and task_loss_channels:
        for i, tid in enumerate(task_ids):
            channels = task_loss_channels.get(tid.item(), list(range(1, C)))
            for ch in channels:
                channel_mask[i, ch] = 1.0
    else:
        channel_mask[:, 1:] = 1.0

    # Apply channel mask: (batch, T, C) * (batch, 1, C)
    masked_mse = full_mse * channel_mask.unsqueeze(1)

    # Sum over timesteps and channels, normalize per sample
    n_channels = channel_mask.sum(dim=1).clamp(min=1)  # (batch,)
    per_sample_loss = masked_mse.sum(dim=(1, 2)) / (n_masked * n_channels)

    return per_sample_loss.mean()


def per_task_mse_loss(
    masked_outputs, masked_targets, masks, task_ids, n_tasks, task_loss_channels=None
):
    """
    Compute MSE loss per task, returning a dict {task_id: avg_loss}.
    """
    batch, _, C = masked_outputs.shape
    n_masked = masks.sum(dim=1).clamp(min=1)  # (batch,)

    full_mse = F.mse_loss(masked_outputs, masked_targets, reduction="none")

    channel_mask = torch.zeros(batch, C, device=masked_outputs.device)
    # Always include channel 0 (fixation)
    channel_mask[:, 0] = 1.0
    if task_ids is not None and task_loss_channels:
        for i, tid in enumerate(task_ids):
            channels = task_loss_channels.get(tid.item(), list(range(1, C)))
            for ch in channels:
                channel_mask[i, ch] = 1.0
    else:
        channel_mask[:, 1:] = 1.0

    masked_mse = full_mse * channel_mask.unsqueeze(1)
    n_channels = channel_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = masked_mse.sum(dim=(1, 2)) / (n_masked * n_channels)

    # Aggregate per task
    task_losses = {i: [] for i in range(n_tasks)}
    for i, tid in enumerate(task_ids):
        task_losses[tid.item()].append(per_sample_loss[i].item())

    return {tid: np.mean(vals) if vals else 0.0 for tid, vals in task_losses.items()}


def variance_normalized_mse_loss(
    masked_outputs, masked_targets, masks, task_ids, task_loss_channels, task_ref
):
    """Same masked, channel-selected, per-sample MSE as `normalized_mse_loss`, but each
    sample's loss is divided by a FIXED per-task reference scale `task_ref[task_id]`
    (e.g. the task's initial loss L_i(0) ~= target variance). This makes every task
    contribute equally to the joint loss regardless of its target scale, so small-target
    tasks are no longer starved of gradient. task_ref: dict {task_id: float}."""
    batch, _, C = masked_outputs.shape
    n_masked = masks.sum(dim=1).clamp(min=1)
    full_mse = F.mse_loss(masked_outputs, masked_targets, reduction="none")
    channel_mask = torch.zeros(batch, C, device=masked_outputs.device)
    channel_mask[:, 0] = 1.0
    if task_ids is not None and task_loss_channels:
        for i, tid in enumerate(task_ids):
            channels = task_loss_channels.get(tid.item(), list(range(1, C)))
            for ch in channels:
                channel_mask[i, ch] = 1.0
    else:
        channel_mask[:, 1:] = 1.0
    masked_mse = full_mse * channel_mask.unsqueeze(1)
    n_channels = channel_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = masked_mse.sum(dim=(1, 2)) / (n_masked * n_channels)
    # divide each sample by its task's frozen reference scale -> relative (scale-free) loss
    ref = torch.tensor(
        [task_ref.get(int(t.item()), 1.0) for t in task_ids],
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    )
    per_sample_loss = per_sample_loss / (ref + 1e-8)
    return per_sample_loss.mean()


def compute_loss(
    model, data_loader, device, tau=0.01, M_reg=10, task_loss_channels=None,
    task_ref=None,
):
    """Compute loss on a dataloader without gradients.

    task_ref: optional {task_id: float} frozen per-task reference scale. When given,
    the loss is `variance_normalized_mse_loss` (each sample divided by its task's
    reference), so the test monitor matches a LOSS_VARNORM=1 training objective.
    When None (default), the plain `normalized_mse_loss` is used (original
    behaviour, and what all pre-2026-08-11 runs were selected on)."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            task_ids = task_ids.to(device)

            # Forward pass: (batch, T, N)
            outputs = model(inputs, task_ids)

            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)  # (batch, T, N)
            masked_outputs = outputs * masks_expanded

            # Handle target format
            if targets.dim() == 2:  # Shape: (batch, N) - single target per sample
                targets_expanded = targets.unsqueeze(1).expand(
                    -1, outputs.shape[1], -1
                )  # (batch, T, N)
                masked_targets = targets_expanded * masks_expanded
            else:  # Shape: (batch, T, N) - sequential targets
                masked_targets = targets * masks_expanded

            if task_ref is not None:
                loss = variance_normalized_mse_loss(
                    masked_outputs, masked_targets, masks, task_ids,
                    task_loss_channels, task_ref,
                )
            else:
                loss = normalized_mse_loss(
                    masked_outputs, masked_targets, masks, task_ids,
                    task_loss_channels,
                )
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)


def compute_per_task_loss(model, data_loader, device, n_tasks, task_loss_channels=None):
    """Compute per-task loss on a dataloader without gradients."""
    model.eval()
    task_loss_accum = {i: [] for i in range(n_tasks)}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            task_ids = task_ids.to(device)

            outputs = model(inputs, task_ids)

            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)
            masked_outputs = outputs * masks_expanded

            if targets.dim() == 2:
                targets_expanded = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                masked_targets = targets_expanded * masks_expanded
            else:
                masked_targets = targets * masks_expanded

            batch_task_losses = per_task_mse_loss(
                masked_outputs,
                masked_targets,
                masks,
                task_ids,
                n_tasks,
                task_loss_channels,
            )
            for tid, val in batch_task_losses.items():
                if val > 0.0 or any(t.item() == tid for t in task_ids):
                    task_loss_accum[tid].append(val)

    model.train()
    return {
        tid: np.mean(vals) if vals else 0.0 for tid, vals in task_loss_accum.items()
    }


def apply_fixation_gate(acc, out_t, tgt_t):
    """Gate a response-criterion score with the fixation channel (Yang et al. 2019).

    Yang's get_perf reads the fixation output z_fix at the scored timestep:

        perf = should_fix*fixating + (1-should_fix)*corr_loc*(1-fixating)

    i.e. on a withhold trial the fixation channel IS the answer (the response
    channels are not consulted at all), and on a response trial the network must
    release fixation AND meet the response criterion. `acc` is the score from the
    per-task branches; `out_t` / `tgt_t` are the output and target vectors at the
    scored (last masked) timestep.

    Withhold vs respond is read off the TARGET, matching Yang's `y_loc < 0`.
    """
    if not FIXATION_GATE:
        return acc
    should_hold = bool(tgt_t[FIX_CHANNEL] > BINARY_THRESHOLD)
    fixating = bool(out_t[FIX_CHANNEL] > BINARY_THRESHOLD)
    if should_hold:
        return 1.0 if fixating else 0.0
    return 0.0 if fixating else acc


def compute_batch_accuracies(
    outputs,
    targets_exp,
    masks,
    task_ids,
    binary_task_ids,
    scalar_task_ids,
    copytask_ids,
    argmax_task_info,
    task_thresholds=None,
    flipflop_task_info=None,
    perstep_binary_task_ids=None,
    reaction_task_info=None,
    vector_task_info=None,
    scalar_vector_tol=None,
):
    """
    Compute per-sample accuracy for one batch.

    Returns a list of (task_id: int, acc: float) — one entry per sample.
    Callers accumulate these into epoch-level correct/total counters.

    outputs:      (batch, T, N)
    targets_exp:  (batch, T, N)  — already expanded from 2-D if needed
    masks:        (batch, T)
    task_ids:     (batch,)
    """
    mask_bool = masks > 0
    masked_counts = mask_bool.sum(dim=1).clamp(min=1).float()

    # Pre-compute angular error (batch, T) — used by angular + copytask branches
    pred_angle = torch.atan2(outputs[..., 2], outputs[..., 1])
    target_angle = torch.atan2(targets_exp[..., 2], targets_exp[..., 1])
    angle_error = torch.abs(pred_angle - target_angle)
    angle_error = torch.min(angle_error, 2 * np.pi - angle_error)

    results = []
    for i, task_id in enumerate(task_ids):
        task_id = task_id.item()
        n_masked = int(masked_counts[i].item())
        mask_idx = torch.where(mask_bool[i])[0]

        if binary_task_ids and task_id in binary_task_ids:
            if len(mask_idx) > 0:
                t_last = mask_idx[-1]
                pred_bin = outputs[i, t_last, BINARY_CHANNEL] > BINARY_THRESHOLD
                tgt_bin = targets_exp[i, t_last, BINARY_CHANNEL] > BINARY_THRESHOLD
                acc = 1.0 if pred_bin == tgt_bin else 0.0
            else:
                acc = 0.0

        elif perstep_binary_task_ids and task_id in perstep_binary_task_ids:
            # Per-timestep binary decision on BINARY_CHANNEL, averaged over
            # masked timesteps (a decision is required at every step, e.g.
            # n-back). Mirrors the flip-flop / copytask per-step metrics.
            if len(mask_idx) > 0:
                pred_bin = outputs[i][:, BINARY_CHANNEL] > BINARY_THRESHOLD
                tgt_bin = targets_exp[i][:, BINARY_CHANNEL] > BINARY_THRESHOLD
                per_correct = (pred_bin == tgt_bin).float() * mask_bool[i].float()
                acc = per_correct.sum().item() / n_masked
            else:
                acc = 0.0

        elif scalar_task_ids and task_id in scalar_task_ids:
            if len(mask_idx) > 0:
                t_last = mask_idx[-1]
                scalar_error = torch.abs(
                    outputs[i, t_last, BINARY_CHANNEL]
                    - targets_exp[i, t_last, BINARY_CHANNEL]
                )
                tol = (
                    scalar_vector_tol.get(task_id, NON_ANGULAR_THRESHOLD)
                    if scalar_vector_tol
                    else NON_ANGULAR_THRESHOLD
                )
                acc = 1.0 if scalar_error.item() < tol else 0.0
            else:
                acc = 0.0

        elif vector_task_info and task_id in vector_task_info:
            # Vector-scalar task (e.g. 2D position): correct if the EUCLIDEAN
            # (L2) distance across readout channels is within the task's
            # tau*sigma tolerance (fallback NON_ANGULAR_THRESHOLD) at the last
            # masked step.
            channels = vector_task_info[task_id]
            if len(mask_idx) > 0:
                t_last = mask_idx[-1]
                diff = outputs[i, t_last, channels] - targets_exp[i, t_last, channels]
                tol = (
                    scalar_vector_tol.get(task_id, NON_ANGULAR_THRESHOLD)
                    if scalar_vector_tol
                    else NON_ANGULAR_THRESHOLD
                )
                acc = 1.0 if torch.linalg.norm(diff).item() < tol else 0.0
            else:
                acc = 0.0

        elif task_id in argmax_task_info:
            channels = argmax_task_info[task_id]
            if len(mask_idx) > 0:
                t_last = mask_idx[-1]
                pred_class = outputs[i, t_last, channels].argmax().item()
                tgt_class = targets_exp[i, t_last, channels].argmax().item()
                acc = 1.0 if pred_class == tgt_class else 0.0
            else:
                acc = 0.0

        elif flipflop_task_info and task_id in flipflop_task_info:
            # Per-timestep sign match across all bit channels; average over
            # masked timesteps (memory must be held at every step).
            channels = flipflop_task_info[task_id]
            if len(mask_idx) > 0:
                ch = torch.tensor(channels, device=outputs.device)
                pred_sign = torch.sign(outputs[i][:, ch])
                tgt_sign = torch.sign(targets_exp[i][:, ch])
                correct_t = (pred_sign == tgt_sign).all(dim=1).float()  # (T,)
                per_correct = correct_t * mask_bool[i].float()
                acc = per_correct.sum().item() / n_masked
            else:
                acc = 0.0

        elif task_id in copytask_ids:
            per_correct = (angle_error[i] < THRESHOLD).float() * mask_bool[i].float()
            acc = int(per_correct.sum().item()) / n_masked

        else:
            # Angular task: last masked timestep -- or an early "reaction" step
            # for reaction-time tasks (scored right after stimulus onset).
            threshold = (
                task_thresholds.get(task_id, THRESHOLD)
                if task_thresholds
                else THRESHOLD
            )
            if len(mask_idx) > 0:
                if reaction_task_info and task_id in reaction_task_info:
                    k = min(reaction_task_info[task_id], len(mask_idx) - 1)
                    t_eval = mask_idx[k]
                else:
                    t_eval = mask_idx[-1]
                acc = 1.0 if angle_error[i, t_eval].item() < threshold else 0.0
            else:
                acc = 0.0

        # Yang-2019 fixation scoring, applied uniformly on top of every response
        # criterion above: read the fixation channel at the last masked step.
        if len(mask_idx) > 0:
            t_fix = mask_idx[-1]
            acc = apply_fixation_gate(acc, outputs[i, t_fix], targets_exp[i, t_fix])

        results.append((task_id, acc))
    return results


def get_latent_states(model, data_loader, device, task_names):
    """Extract hidden states z for every timestep, grouped by task."""
    model.eval()
    n_tasks = len(task_names)
    all_latents = {i: [] for i in range(n_tasks)}
    all_info = {i: {"targets": [], "masks": [], "inputs": []} for i in range(n_tasks)}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            task_ids_dev = task_ids.to(device)
            B, T, _ = inputs.shape

            # Replicate forward pass to capture z at every timestep
            A, W, h, C, D = model.hierarchisation_scheme.get_parameters(task_ids_dev)
            z = torch.zeros(B, model.M, device=device)
            z_all = torch.empty(B, T, model.M, device=device)

            for t in range(T):
                if model.L > 0:
                    z_non_latent = z[:, : -model.L]
                    z_latent = z[:, -model.L :]
                    z_latent_scaled = A * z_latent
                    z_latent_act = F.relu(z_latent)
                    z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                    z_update = torch.cat(
                        [torch.zeros_like(z_non_latent), z_latent_scaled], dim=1
                    )
                else:
                    z_combined = z
                    z_update = torch.zeros_like(z)

                C_output = torch.einsum("bij,bj->bi", C, inputs[:, t])
                W_output = torch.einsum("bij,bj->bi", W, z_combined)
                z = z_update + W_output + C_output + h
                z_all[:, t] = z

            # Store by task
            z_all_cpu = z_all.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            inputs_cpu = inputs.cpu().numpy()
            for i in range(B):
                tid = task_ids[i].item()
                all_latents[tid].append(z_all_cpu[i])
                all_info[tid]["targets"].append(targets_cpu[i])
                all_info[tid]["masks"].append(masks_cpu[i])
                all_info[tid]["inputs"].append(inputs_cpu[i])

    def _pad_and_stack(arrays):
        """Pad arrays with different T to the max length, then stack."""
        max_T = max(a.shape[0] for a in arrays)
        padded = []
        for a in arrays:
            pad_width = [(0, max_T - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
            padded.append(np.pad(a, pad_width, mode="constant", constant_values=0))
        return np.stack(padded, axis=0)

    latent_states = {}
    trial_info = {}
    for tid in range(n_tasks):
        if all_latents[tid]:
            latent_states[tid] = _pad_and_stack(all_latents[tid])
            trial_info[tid] = {
                "targets": _pad_and_stack(all_info[tid]["targets"]),
                "masks": _pad_and_stack(all_info[tid]["masks"]),
                "inputs": _pad_and_stack(all_info[tid]["inputs"]),
            }
        else:
            latent_states[tid] = np.array([])
            trial_info[tid] = {
                "targets": np.array([]),
                "masks": np.array([]),
                "inputs": np.array([]),
            }

    model.train()
    return latent_states, trial_info


def compute_accuracies(
    model,
    data_loader,
    device,
    n_tasks,
    task_thresholds=None,
    tasks=None,
    scalar_vector_tol=None,
):
    """Compute accuracies per task on a dataloader.

    scalar_vector_tol: optional {task_id: tau*sigma} for the scalar/vector tasks
    (from compute_scalar_vector_tol). When None, those tasks fall back to the
    fixed NON_ANGULAR_THRESHOLD (legacy behaviour).
    """
    binary_task_ids = set()
    scalar_task_ids = set()
    copytask_ids = set()
    perstep_binary_task_ids = set()
    argmax_task_info = {}
    flipflop_task_info = {}
    reaction_task_info = {}
    vector_task_info = {}
    if tasks is not None:
        for i, task in enumerate(tasks):
            if getattr(task, "is_binary_task", False):
                binary_task_ids.add(i)
            if getattr(task, "is_scalar_task", False):
                scalar_task_ids.add(i)
            if getattr(task, "is_copytask", False):
                copytask_ids.add(i)
            if getattr(task, "is_perstep_binary_task", False):
                perstep_binary_task_ids.add(i)
            if getattr(task, "is_argmax_task", False):
                argmax_task_info[i] = task.loss_channels
            if getattr(task, "is_flipflop_task", False):
                flipflop_task_info[i] = task.loss_channels
            if getattr(task, "is_reaction_task", False):
                reaction_task_info[i] = getattr(task, "reaction_step", 1)
            if getattr(task, "is_vector_task", False):
                vector_task_info[i] = task.loss_channels

    model.eval()
    task_correct = {i: 0.0 for i in range(n_tasks)}
    task_total = {i: 0 for i in range(n_tasks)}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            task_ids = task_ids.to(device)

            outputs = model(inputs, task_ids)
            targets_exp = (
                targets.unsqueeze(1).expand_as(outputs)
                if targets.dim() == 2
                else targets
            )

            for task_id, acc in compute_batch_accuracies(
                outputs,
                targets_exp,
                masks,
                task_ids,
                binary_task_ids,
                scalar_task_ids,
                copytask_ids,
                argmax_task_info,
                task_thresholds,
                flipflop_task_info,
                perstep_binary_task_ids,
                reaction_task_info,
                vector_task_info,
                scalar_vector_tol,
            ):
                task_correct[task_id] += acc
                task_total[task_id] += 1

    task_accuracies = {
        i: task_correct[i] / task_total[i] if task_total[i] > 0 else 0.0
        for i in range(n_tasks)
    }
    model.train()
    return task_accuracies
