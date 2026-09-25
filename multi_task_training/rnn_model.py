import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from .MAR import regularization_loss

THRESHOLD = (
    5 * 4 * np.pi / 180
)  # 20 degrees in radians, used for single-timestep tasks (not CopyTask)

BINARY_THRESHOLD = 0.5
BINARY_CHANNEL = 3  # Dedicated channel for binary (go/nogo) responses
NON_ANGULAR_THRESHOLD = 0.1


# ============================================================================
# Modified AL-RNN Model
# ============================================================================
class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim

        # Diagonal matrix A for latent dynamics
        self.A = nn.Parameter(torch.randn(L) * 0.1 + 0.9)

        # Recurrent weight matrix
        self.W = nn.Parameter(torch.randn(M, M) * 0.1 / np.sqrt(M))

        # Bias
        self.h = nn.Parameter(torch.zeros(M))

        # Input weights
        self.C = nn.Parameter(torch.randn(M, input_dim) * 0.1)

        # Output weights
        self.D = nn.Parameter(torch.randn(N, M) * 0.1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.M)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, T, input_dim)
        Returns:
            outputs: (batch, N)
        """
        batch_size, T, _ = inputs.shape

        # Initialize hidden state
        z = self.init_hidden(batch_size).to(inputs.device)

        traj = []

        # Process sequence
        for t in range(T):
            # Split into non-latent and latent parts
            if self.L > 0:
                z_non_latent = z[:, : -self.L]  # excluim les L últimes dimensions
                z_latent = z[:, -self.L :]  # només les L últimes dimensions

                # Apply diagonal A to latent part only
                z_latent_scaled = self.A * z_latent

                # Apply ReLU to latent part
                z_latent_act = F.relu(z_latent)

                # Reconstruct z for matrix multiplication
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)

                # Update
                z_update = z_latent_scaled
                z_update = torch.cat([torch.zeros_like(z_non_latent), z_update], dim=1)
            else:
                z_combined = z
                z_update = torch.zeros_like(z)

            # RNN update
            z = z_update + z_combined @ self.W.t() + inputs[:, t] @ self.C.t() + self.h

            # Generate output at this timestep
            output_t = z @ self.D.t()
            traj.append(output_t)

        # Stack all outputs: (batch, T, N)
        return torch.stack(traj, dim=1)


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
    """Compute MSE loss per task, returning a dict {task_id: avg_loss}."""
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

    task_losses = {i: [] for i in range(n_tasks)}
    for i, tid in enumerate(task_ids):
        task_losses[tid.item()].append(per_sample_loss[i].item())

    return {tid: np.mean(vals) if vals else 0.0 for tid, vals in task_losses.items()}


def compute_per_task_loss(model, data_loader, device, n_tasks, task_loss_channels=None):
    """Compute per-task loss on a dataloader without gradients."""
    model.eval()
    task_loss_accum = {i: [] for i in range(n_tasks)}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = model(inputs)

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


def compute_loss(
    model, data_loader, device, tau=0.01, M_reg=10, task_loss_channels=None
):
    """Compute loss on a dataloader without gradients"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # Forward pass: (batch, T, N)
            outputs = model(inputs)

            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)  # (batch, T, N)
            masked_outputs = outputs * masks_expanded

            # Handle target format
            if targets.dim() == 2:  # Shape: (batch, N) - single target per sample
                # Expand targets to match sequence length, then mask
                targets_expanded = targets.unsqueeze(1).expand(
                    -1, outputs.shape[1], -1
                )  # (batch, T, N)
                masked_targets = targets_expanded * masks_expanded
            else:  # Shape: (batch, T, N) - sequential targets
                masked_targets = targets * masks_expanded

            # Compute loss (normalized per sample by masked timesteps)
            loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels
            )
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)


def compute_accuracies(
    model, data_loader, device, n_tasks, task_thresholds=None, tasks=None
):
    """Compute accuracies per task on a dataloader"""
    # Derive task type flags from task definitions
    binary_task_ids = set()
    scalar_task_ids = set()
    copytask_ids = set()
    argmax_task_info = {}
    if tasks is not None:
        for i, task in enumerate(tasks):
            if getattr(task, "is_binary_task", False):
                binary_task_ids.add(i)
            if getattr(task, "is_scalar_task", False):
                scalar_task_ids.add(i)
            if getattr(task, "is_copytask", False):
                copytask_ids.add(i)
            if getattr(task, "is_argmax_task", False):
                argmax_task_info[i] = task.loss_channels

    model.eval()
    task_correct = {i: 0.0 for i in range(n_tasks)}
    task_total = {i: 0 for i in range(n_tasks)}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # Forward pass: (batch, T, N)
            outputs = model(inputs)

            pred_angle = torch.atan2(outputs[..., 2], outputs[..., 1])
            target_angle = torch.atan2(targets[..., 2], targets[..., 1])
            angle_error = torch.abs(pred_angle - target_angle)
            angle_error = torch.min(angle_error, 2 * np.pi - angle_error)

            # Count masked timesteps per sample
            mask_bool = masks > 0
            masked_counts = mask_bool.sum(dim=1).clamp(min=1).float()

            # Strict all-or-nothing accuracy for all tasks
            for i, task_id in enumerate(task_ids):
                task_id = task_id.item()
                n_masked = int(masked_counts[i].item())

                if binary_task_ids and task_id in binary_task_ids:
                    # Binary task: BINARY_CHANNEL > 0.5 at last masked timestep
                    tgt_exp = (
                        targets.unsqueeze(1).expand_as(outputs)
                        if targets.dim() == 2
                        else targets
                    )
                    mask_idx = torch.where(mask_bool[i])[0]
                    if len(mask_idx) > 0:
                        t_last = mask_idx[-1]
                        pred_bin = outputs[i, t_last, BINARY_CHANNEL] > BINARY_THRESHOLD
                        tgt_bin = tgt_exp[i, t_last, BINARY_CHANNEL] > BINARY_THRESHOLD
                        acc = 1.0 if pred_bin == tgt_bin else 0.0
                    else:
                        acc = 0.0
                elif scalar_task_ids and task_id in scalar_task_ids:
                    # Scalar task: compare last channel value at last masked timestep
                    tgt_exp = (
                        targets.unsqueeze(1).expand_as(outputs)
                        if targets.dim() == 2
                        else targets
                    )
                    mask_idx = torch.where(mask_bool[i])[0]
                    if len(mask_idx) > 0:
                        t_last = mask_idx[-1]
                        scalar_error = torch.abs(
                            outputs[i, t_last, BINARY_CHANNEL]
                            - tgt_exp[i, t_last, BINARY_CHANNEL]
                        )
                        acc = (
                            1.0 if scalar_error.item() < NON_ANGULAR_THRESHOLD else 0.0
                        )
                    else:
                        acc = 0.0
                elif task_id in argmax_task_info:
                    # Argmax task: compare argmax over loss channels
                    channels = argmax_task_info[task_id]
                    tgt_exp = (
                        targets.unsqueeze(1).expand_as(outputs)
                        if targets.dim() == 2
                        else targets
                    )
                    mask_idx = torch.where(mask_bool[i])[0]
                    if len(mask_idx) > 0:
                        t_last = mask_idx[-1]
                        pred_class = outputs[i, t_last, channels].argmax().item()
                        tgt_class = tgt_exp[i, t_last, channels].argmax().item()
                        acc = 1.0 if pred_class == tgt_class else 0.0
                    else:
                        acc = 0.0
                elif task_id in copytask_ids:
                    # Copy task: per-symbol accuracy (fraction of correct timesteps)
                    threshold = THRESHOLD
                    per_sample_correct = (
                        angle_error[i] < threshold
                    ).float() * mask_bool[i].float()
                    n_correct = int(per_sample_correct.sum().item())
                    acc = n_correct / n_masked
                else:
                    # Angular tasks: check at last masked timestep
                    if task_thresholds is not None:
                        threshold = task_thresholds.get(task_id, THRESHOLD)
                    else:
                        threshold = THRESHOLD
                    mask_idx = torch.where(mask_bool[i])[0]
                    if len(mask_idx) > 0:
                        t_last = mask_idx[-1]
                        acc = 1.0 if angle_error[i, t_last].item() < threshold else 0.0
                    else:
                        acc = 0.0

                task_correct[task_id] += acc
                task_total[task_id] += 1

    task_accuracies = {}
    for task_id in range(n_tasks):
        if task_total[task_id] > 0:
            task_accuracies[task_id] = task_correct[task_id] / task_total[task_id]
        else:
            task_accuracies[task_id] = 0.0

    model.train()
    return task_accuracies


def get_ewc_loss(model, fisher, p_old):
    loss = 0
    for n, p in model.named_parameters():
        _loss = fisher[n] * (p - p_old[n]) ** 2
        loss += _loss.sum()
    return loss


def quantize_copy_task_outputs(outputs, masks):
    """
    Quantize outputs for CopyTask (sequence tasks with multiple masked timesteps).
    Maps continuous values to discrete {-1, 0, 1}:
    - value > 0.5 → 1
    - -0.5 <= value <= 0.5 → 0
    - value < -0.5 → -1

    Args:
        outputs: (batch, T, N) model outputs
        masks: (batch, T) mask indicating which timesteps to evaluate

    Returns:
        outputs: (batch, T, N) with quantized values for sequence tasks
    """
    # Identify sequence tasks (multiple masked timesteps)
    mask_counts = masks.sum(dim=1)  # (batch,)
    is_sequence_task = mask_counts > 1  # (batch,)

    if not is_sequence_task.any():
        return outputs

    # Clone outputs to avoid in-place modification
    outputs_quantized = outputs.clone()

    # Apply quantization only to sequence task samples
    for i in range(outputs.shape[0]):
        if is_sequence_task[i]:
            # Quantize direction channels (cos, sin) - channels 1 and 2
            for ch in [1, 2]:
                vals = outputs_quantized[i, :, ch]
                vals = torch.where(vals > 0.5, torch.ones_like(vals), vals)
                vals = torch.where(
                    (vals >= -0.5) & (vals <= 0.5), torch.zeros_like(vals), vals
                )
                vals = torch.where(vals < -0.5, -torch.ones_like(vals), vals)
                outputs_quantized[i, :, ch] = vals

    return outputs_quantized


def train_multitask(
    model,
    train_loader,
    optimizer,
    device,
    tau=0.01,
    M_reg=10,
    num_epochs=100,
    test_loader=None,
    fine_tune=False,
    fisher=None,
    old_params=None,
    lambda_ewc=100000000,
    task_thresholds=None,
    verbose=True,
    early_stopping_patience=None,
):
    """Train the model on multiple tasks"""
    # Derive loss channels and task type flags from task definitions
    tasks = train_loader.dataset.tasks
    task_loss_channels = {}
    binary_task_ids = set()
    scalar_task_ids = set()
    argmax_task_info = {}
    for i, task in enumerate(tasks):
        if getattr(task, "loss_channels", None) is not None:
            task_loss_channels[i] = task.loss_channels
        if getattr(task, "is_binary_task", False):
            binary_task_ids.add(i)
        if getattr(task, "is_scalar_task", False):
            scalar_task_ids.add(i)
        if getattr(task, "is_argmax_task", False):
            argmax_task_info[i] = task.loss_channels

    loss_history = []
    test_loss_history = []
    train_task_accuracies = {i: [] for i in range(len(tasks))}
    test_task_accuracies = {i: [] for i in range(len(tasks))}
    test_task_losses = {i: [] for i in range(len(tasks))}

    # Early stopping state
    best_test_acc_es = -float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_task_correct = {i: 0.0 for i in range(len(train_loader.dataset.tasks))}
        epoch_task_total = {i: 0 for i in range(len(train_loader.dataset.tasks))}

        for batch_idx, (inputs, targets, masks, task_ids) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass: (batch, T, N)
            outputs = model(inputs)

            # Always apply mask to select which timesteps to evaluate
            masks_expanded = masks.unsqueeze(-1).expand_as(outputs)  # (batch, T, 3)
            masked_outputs = outputs * masks_expanded

            # Handle target format
            if targets.dim() == 2:  # Shape: (batch, 3) - single target per sample
                # Expand targets to match sequence length, then mask
                targets_expanded = targets.unsqueeze(1).expand(
                    -1, outputs.shape[1], -1
                )  # (batch, T, 3)
                masked_targets = targets_expanded * masks_expanded
            else:  # Shape: (batch, T, 3) - sequential targets
                masked_targets = targets * masks_expanded

            # Compute loss (normalized per sample by masked timesteps)
            loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels
            )

            fix_loss = F.mse_loss(outputs[:, 0], targets[:, 0])

            reg_loss = regularization_loss(model, tau, M_reg)

            loss += reg_loss + fix_loss

            if fine_tune:
                ewc_loss = get_ewc_loss(model, fisher, old_params)
                loss += lambda_ewc * ewc_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy (angle error < threshold)
            with torch.no_grad():

                if targets.dim() == 2:
                    targets_exp = targets.unsqueeze(1).expand_as(outputs)
                else:
                    targets_exp = targets

                # Per-timestep angle error (batch, T)
                pred_angle = torch.atan2(outputs[..., 2], outputs[..., 1])
                target_angle = torch.atan2(targets_exp[..., 2], targets_exp[..., 1])
                angle_error = torch.abs(pred_angle - target_angle)
                angle_error = torch.min(angle_error, 2 * np.pi - angle_error)

                # Count masked timesteps per sample
                mask_bool = masks > 0
                masked_counts = mask_bool.sum(dim=1).clamp(min=1).float()

                # Last-timestep accuracy (aligned with hierarchical pipeline)
                copytask_ids = set()
                for idx, task in enumerate(tasks):
                    if getattr(task, "is_copytask", False):
                        copytask_ids.add(idx)

                for i, task_id in enumerate(task_ids):
                    task_id = task_id.item()
                    n_masked = int(masked_counts[i].item())
                    mask_idx = torch.where(mask_bool[i])[0]

                    if binary_task_ids and task_id in binary_task_ids:
                        # Binary task: last masked timestep
                        if len(mask_idx) > 0:
                            t_last = mask_idx[-1]
                            pred_bin = (
                                outputs[i, t_last, BINARY_CHANNEL] > BINARY_THRESHOLD
                            )
                            tgt_bin = (
                                targets_exp[i, t_last, BINARY_CHANNEL]
                                > BINARY_THRESHOLD
                            )
                            acc = 1.0 if pred_bin == tgt_bin else 0.0
                        else:
                            acc = 0.0
                    elif scalar_task_ids and task_id in scalar_task_ids:
                        # Scalar task: last masked timestep
                        if len(mask_idx) > 0:
                            t_last = mask_idx[-1]
                            scalar_error = torch.abs(
                                outputs[i, t_last, BINARY_CHANNEL]
                                - targets_exp[i, t_last, BINARY_CHANNEL]
                            )
                            acc = (
                                1.0
                                if scalar_error.item() < NON_ANGULAR_THRESHOLD
                                else 0.0
                            )
                        else:
                            acc = 0.0
                    elif task_id in argmax_task_info:
                        # Argmax task: last masked timestep
                        channels = argmax_task_info[task_id]
                        if len(mask_idx) > 0:
                            t_last = mask_idx[-1]
                            pred_class = outputs[i, t_last, channels].argmax().item()
                            tgt_class = targets_exp[i, t_last, channels].argmax().item()
                            acc = 1.0 if pred_class == tgt_class else 0.0
                        else:
                            acc = 0.0
                    elif task_id in copytask_ids:
                        # Copy task: per-symbol accuracy (fraction of correct timesteps)
                        per_correct = (angle_error[i] < THRESHOLD).float() * mask_bool[
                            i
                        ].float()
                        acc = int(per_correct.sum().item()) / n_masked
                    else:
                        # Angular task: last masked timestep
                        threshold = (
                            task_thresholds.get(task_id, THRESHOLD)
                            if task_thresholds
                            else THRESHOLD
                        )
                        if len(mask_idx) > 0:
                            t_last = mask_idx[-1]
                            acc = (
                                1.0
                                if angle_error[i, t_last].item() < threshold
                                else 0.0
                            )
                        else:
                            acc = 0.0

                    epoch_task_correct[task_id] += acc
                    epoch_task_total[task_id] += 1

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Compute train task accuracies
        for task_id in range(len(train_loader.dataset.tasks)):
            if epoch_task_total[task_id] > 0:
                acc = epoch_task_correct[task_id] / epoch_task_total[task_id]
                train_task_accuracies[task_id].append(acc)
            else:
                train_task_accuracies[task_id].append(0)

        # Compute test accuracies
        if test_loader is not None:
            test_loss = compute_loss(
                model, test_loader, device, tau, M_reg, task_loss_channels
            )
            test_loss_history.append(test_loss)
            epoch_test_task_losses = compute_per_task_loss(
                model,
                test_loader,
                device,
                len(tasks),
                task_loss_channels,
            )
            for tid in range(len(tasks)):
                test_task_losses[tid].append(epoch_test_task_losses[tid])
            test_accs = compute_accuracies(
                model,
                test_loader,
                device,
                len(tasks),
                task_thresholds,
                tasks=tasks,
            )
            for task_id in range(len(train_loader.dataset.tasks)):
                test_task_accuracies[task_id].append(test_accs[task_id])

        # Early stopping check (based on mean test accuracy)
        if early_stopping_patience is not None and test_loader is not None:
            current_test_acc = np.mean(
                [test_task_accuracies[i][-1] for i in range(len(tasks))]
            )
            if current_test_acc > best_test_acc_es:
                best_test_acc_es = current_test_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", end="")
            if test_loader is not None:
                mean_test_acc = np.mean(
                    [
                        test_task_accuracies[i][-1]
                        for i in range(len(train_loader.dataset.tasks))
                    ]
                )
                print(
                    f", Test Loss: {test_loss_history[-1]:.4f}, Test Acc: {mean_test_acc:.2%}",
                    end="",
                )
            print()
            for task_id, acc_list in train_task_accuracies.items():
                print(f"  Task {task_id} Train Acc: {acc_list[-1]:.2%}", end="")
                if test_loader is not None:
                    print(
                        f", Test Acc: {test_task_accuracies[task_id][-1]:.2%}", end=""
                    )
                print()

    # Restore best model weights if early stopping was used
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # determine best epoch from mean test accuracy (preferred) or train loss
    best_idx = None
    best_epoch = None
    best_loss = None
    if len(test_task_accuracies) > 0 and len(test_task_accuracies[0]) > 0:
        n_epochs_recorded = len(test_task_accuracies[0])
        mean_accs = np.array(
            [
                np.mean([test_task_accuracies[tid][e] for tid in range(len(tasks))])
                for e in range(n_epochs_recorded)
            ]
        )
        best_idx = int(np.argmax(mean_accs))
        best_loss = (
            float(test_loss_history[best_idx])
            if len(test_loss_history) > best_idx
            else None
        )
        best_epoch = best_idx

    # Compute per-task best test accuracies at best_idx (if available) and the mean across tasks
    if best_idx is not None:
        best_test_accuracies = {}
        vals = []
        for tid, acc_list in test_task_accuracies.items():
            if len(acc_list) > best_idx:
                val = float(acc_list[best_idx])
            else:
                val = None
            best_test_accuracies[tid] = val
            if val is not None:
                vals.append(val)

        best_test_accuracy = float(np.mean(vals)) if len(vals) > 0 else None
    else:
        best_test_accuracies = {tid: None for tid in test_task_accuracies.keys()}
        best_test_accuracy = None

    return (
        loss_history,
        train_task_accuracies,
        test_loss_history,
        test_task_accuracies,
        test_task_losses,
        best_test_accuracies,
        best_test_accuracy,
        best_loss,
        best_epoch,
        outputs.detach().cpu().numpy(),
    )


def get_latent_states(model, test_loader, device, task_names):
    model.eval()

    # Collect all trials first
    all_latents = {i: [] for i in range(len(task_names))}
    all_info = {
        i: {"targets": [], "masks": [], "inputs": []} for i in range(len(task_names))
    }

    with torch.no_grad():
        for inputs, targets, masks, task_ids in test_loader:
            inputs = inputs.to(device)
            batch_size, T, _ = inputs.shape

            # Store states at each timestep
            z_all = torch.zeros(batch_size, T, model.M)
            z = model.init_hidden(batch_size).to(device)

            for t in range(T):
                # Split into non-latent and latent parts
                if model.L > 0:
                    z_non_latent = z[:, : -model.L]
                    z_latent = z[:, -model.L :]

                    z_latent_scaled = model.A * z_latent
                    z_latent_act = torch.nn.functional.relu(z_latent)
                    z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                    z_update = torch.cat(
                        [torch.zeros_like(z_non_latent), z_latent_scaled], dim=1
                    )
                else:
                    z_combined = z
                    z_update = torch.zeros_like(z)

                z = (
                    z_update
                    + z_combined @ model.W.t()
                    + inputs[:, t] @ model.C.t()
                    + model.h
                )
                z_all[:, t] = z.cpu()

            # Store by task
            for i in range(batch_size):
                task_id = task_ids[i].item()
                all_latents[task_id].append(z_all[i].numpy())
                all_info[task_id]["targets"].append(targets[i].cpu().numpy())
                all_info[task_id]["masks"].append(masks[i].cpu().numpy())
                all_info[task_id]["inputs"].append(inputs[i].cpu().numpy())

    # Convert lists to arrays: (n_trials, T_max, M)
    latent_states = {}
    trial_info = {}

    for task_id in range(len(task_names)):
        if len(all_latents[task_id]) > 0:
            latent_states[task_id] = np.stack(
                all_latents[task_id], axis=0
            )  # (n_trials, T, M)
            trial_info[task_id] = {
                "targets": np.stack(
                    all_info[task_id]["targets"], axis=0
                ),  # (n_trials, 3)
                "masks": np.stack(all_info[task_id]["masks"], axis=0),  # (n_trials, T)
                "inputs": np.stack(
                    all_info[task_id]["inputs"], axis=0
                ),  # (n_trials, T, input_dim)
            }
        else:
            latent_states[task_id] = np.array([])
            trial_info[task_id] = {
                "targets": np.array([]),
                "masks": np.array([]),
                "inputs": np.array([]),
            }

    return latent_states, trial_info


# Usage example:
def run_test_evaluation(model, tasks, task_names, device, n_test_trials=500):
    """Complete test evaluation pipeline"""
    from tasks.dataset import MultiTaskDataset, collate_fn

    # Create test dataset
    test_dataset = MultiTaskDataset(tasks, n_trials=n_test_trials)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Get test accuracies
    print("Evaluating test accuracy...")
    test_accuracies = evaluate_model(model, test_loader, device, task_names)

    print("\nTest Accuracies:")
    for task_id, acc in test_accuracies.items():
        print(f"  {task_names[task_id]}: {acc:.2%}")

    # Get latent states
    print("\nExtracting latent states...")
    latent_states, trial_info = get_latent_states(
        model, test_loader, device, task_names
    )

    print("\nLatent states collected:")
    for task_id in range(len(task_names)):
        print(f"  {task_names[task_id]}: {len(latent_states[task_id])} trials")
        if len(latent_states[task_id]) > 0:
            print(f"    Shape per trial: {latent_states[task_id][0].shape}")

    return test_accuracies, latent_states, trial_info


def evaluate_model(
    model,
    test_loader,
    device,
    task_names,
    task_thresholds=None,
    tasks=None,
):
    """Evaluate model on test set and return accuracies"""
    # Derive task type flags from task definitions
    binary_task_ids = set()
    scalar_task_ids = set()
    copytask_ids = set()
    argmax_task_info = {}
    if tasks is not None:
        for i, task in enumerate(tasks):
            if getattr(task, "is_binary_task", False):
                binary_task_ids.add(i)
            if getattr(task, "is_scalar_task", False):
                scalar_task_ids.add(i)
            if getattr(task, "is_copytask", False):
                copytask_ids.add(i)
            if getattr(task, "is_argmax_task", False):
                argmax_task_info[i] = task.loss_channels

    model.eval()

    task_correct = {i: 0.0 for i in range(len(task_names))}
    task_total = {i: 0 for i in range(len(task_names))}

    with torch.no_grad():
        for inputs, targets, masks, task_ids in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = model(inputs)  # Shape: (batch, T, N)

            # Extract response phase output (last timestep where mask > 0)
            batch_size = outputs.shape[0]
            response_outputs = []
            response_targets = []
            for b in range(batch_size):
                mask_idx = torch.where(masks[b] > 0)[0]
                if len(mask_idx) > 0:
                    t_idx = mask_idx[-1]
                else:
                    t_idx = outputs.shape[1] - 1
                response_outputs.append(outputs[b, t_idx, :])
                response_targets.append(
                    targets[b, t_idx, :] if targets.dim() == 3 else targets[b, :]
                )
            response_outputs = torch.stack(response_outputs)  # Shape: (batch, N)
            response_targets = torch.stack(response_targets)  # Shape: (batch, N)

            # Compute angle error (for standard cos/sin tasks)
            pred_angle = torch.atan2(response_outputs[:, 2], response_outputs[:, 1])
            target_angle = torch.atan2(response_targets[:, 2], response_targets[:, 1])
            angle_error = torch.abs(pred_angle - target_angle)
            angle_error = torch.min(angle_error, 2 * np.pi - angle_error)

            for i, task_id in enumerate(task_ids):
                task_id = task_id.item()
                if binary_task_ids and task_id in binary_task_ids:
                    # Binary task: BINARY_CHANNEL > 0.5 classification
                    pred_bin = response_outputs[i, BINARY_CHANNEL] > BINARY_THRESHOLD
                    tgt_bin = response_targets[i, BINARY_CHANNEL] > BINARY_THRESHOLD
                    task_correct[task_id] += int((pred_bin == tgt_bin).item())
                elif scalar_task_ids and task_id in scalar_task_ids:
                    # Scalar task: compare last channel at last masked timestep
                    mask_idx = torch.where(masks[i] > 0)[0]
                    if len(mask_idx) > 0:
                        t_last = mask_idx[-1]
                        pred_val = outputs[i, t_last, BINARY_CHANNEL]
                        tgt_val = (
                            targets[i, t_last, BINARY_CHANNEL]
                            if targets.dim() == 3
                            else targets[i, BINARY_CHANNEL]
                        )
                        scalar_error = torch.abs(pred_val - tgt_val)
                        task_correct[task_id] += int(
                            (scalar_error < NON_ANGULAR_THRESHOLD).item()
                        )
                elif task_id in argmax_task_info:
                    # Argmax task: compare argmax over loss channels
                    channels = argmax_task_info[task_id]
                    pred_class = response_outputs[i, channels].argmax().item()
                    tgt_class = response_targets[i, channels].argmax().item()
                    task_correct[task_id] += int(pred_class == tgt_class)
                elif copytask_ids and task_id in copytask_ids:
                    # CopyTask: per-timestep angular accuracy over the response window
                    mask_idx = torch.where(masks[i] > 0)[0]
                    if len(mask_idx) > 0:
                        if task_thresholds is not None:
                            threshold = task_thresholds.get(task_id, THRESHOLD)
                        else:
                            threshold = THRESHOLD
                        n_correct = 0
                        for t in mask_idx:
                            p_ang = torch.atan2(outputs[i, t, 2], outputs[i, t, 1])
                            t_ang = torch.atan2(
                                (
                                    targets[i, t, 2]
                                    if targets.dim() == 3
                                    else targets[i, 2]
                                ),
                                (
                                    targets[i, t, 1]
                                    if targets.dim() == 3
                                    else targets[i, 1]
                                ),
                            )
                            err = torch.abs(p_ang - t_ang)
                            err = torch.min(err, 2 * np.pi - err)
                            n_correct += int((err < threshold).item())
                        task_correct[task_id] += n_correct / len(mask_idx)
                else:
                    if task_thresholds is not None:
                        threshold = task_thresholds.get(task_id, THRESHOLD)
                    else:
                        threshold = THRESHOLD
                    task_correct[task_id] += int((angle_error[i] < threshold).item())
                task_total[task_id] += 1

    # Compute accuracies
    task_accuracies = {}
    for task_id in range(len(task_names)):
        if task_total[task_id] > 0:
            task_accuracies[task_id] = task_correct[task_id] / task_total[task_id]
        else:
            task_accuracies[task_id] = 0.0

    return task_accuracies
