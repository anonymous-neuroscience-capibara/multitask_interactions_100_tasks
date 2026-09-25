import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from hierachical_model_task.utils import HiearchicalModelConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from hierachical_model_task.model import HierarchicalPLRNN
from hierachical_model_task.rnn_model import (
    compute_accuracies,
    compute_batch_accuracies,
    compute_scalar_vector_tol,
    compute_loss,
    compute_per_task_loss,
    normalized_mse_loss,
    per_task_mse_loss,
    variance_normalized_mse_loss,
)

from tasks.dataset import HierarchicalTasksDataset


class BPTT:
    def __init__(self, model, args: HiearchicalModelConfig):
        self.args = args
        self.device = args.device

        if model is not None:
            self.model = model
        else:
            self.model = HierarchicalPLRNN(args, HierarchicalTasksDataset)

        # compile model if specified
        if args.compile:
            self.model = torch.compile(self.model)

        # initialize optimizers
        shared, individual = self.model.hierarchisation_scheme.grouped_parameters()
        self.shared_optimizer = Adam(
            shared, lr=args.learning_rate[0], weight_decay=args.weight_decay
        )
        self.individual_optimizer = Adam(
            individual, lr=args.learning_rate[1], weight_decay=args.weight_decay
        )
        # exponential LR schedule leads to compilation issues due to float multiplication (?)
        # therefore, we use a lambda function to decay the LR and do the multiplication with a tensor
        self.shared_scheduler = LambdaLR(
            self.shared_optimizer, lambda epoch: torch.tensor(0.999) ** epoch
        )
        self.individual_scheduler = LambdaLR(
            self.individual_optimizer, lambda epoch: torch.tensor(0.999) ** epoch
        )

        # LR warmup + adaptive-on-instability control. base_lr is the full LR per
        # optimizer; each epoch the effective LR = base_lr * warmup_factor * lr_scale.
        # lr_scale is halved whenever a non-finite (exploding) batch is detected, so
        # a run that starts to blow up cools itself down instead of dying.
        self.base_lr = (args.learning_rate[0], args.learning_rate[1])
        self.lr_warmup_epochs = getattr(args, "lr_warmup_epochs", 0)
        self.lr_scale = 1.0

        # move model to device
        self.model.to(args.device)

    def _apply_lr(self, epoch):
        """Set the effective LR for this epoch (warmup ramp * adaptive scale)."""
        warm = self.lr_warmup_epochs
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for opt, base in (
            (self.shared_optimizer, self.base_lr[0]),
            (self.individual_optimizer, self.base_lr[1]),
        ):
            for g in opt.param_groups:
                g["lr"] = base * factor * self.lr_scale

    def train(
        self,
        train_loader,
        test_loader,
        verbose,
        task_thresholds=None,
        task_names=None,
        eval_interval=1,
        checkpoint_path=None,
        checkpoint_acc_path=None,
    ):
        """Train the model on multiple tasks.

        eval_interval: run the (expensive) full test-set evaluation only every
            this many epochs (plus the final epoch). Default 1 = every epoch
            (original behaviour).
        checkpoint_path: if set, write the best-by-test-loss model state to this
            path whenever it improves, so a walltime kill still leaves a usable
            model on disk.
        checkpoint_acc_path: if set, ALSO write the best-by-mean-test-accuracy
            model state to this path whenever the mean test accuracy improves.
            Stopping is still governed by test loss; this only tracks/saves the
            accuracy-optimal checkpoint in parallel (see self.best_acc_epoch and
            self.best_mean_acc after training).
        """
        # Derive loss channels and task type flags from task definitions
        tasks = train_loader.dataset.tasks
        task_loss_channels = {}
        binary_task_ids = set()
        scalar_task_ids = set()
        argmax_task_info = {}
        copytask_ids = set()
        perstep_binary_task_ids = set()
        flipflop_task_info = {}
        reaction_task_info = {}
        vector_task_info = {}
        for i, task in enumerate(tasks):
            if getattr(task, "loss_channels", None) is not None:
                task_loss_channels[i] = task.loss_channels
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

        # tau*sigma tolerance for the 7 continuous-readout (scalar/vector) tasks: a trial
        # is correct if the readout error is within TAU_SIGMA * sigma_task, where sigma_task
        # is the std of that task's targets (compute_scalar_vector_tol, tau=0.35). Returns
        # {task_id: tol} for those tasks only; every other task falls back to its own fixed
        # criterion (NON_ANGULAR_THRESHOLD for scalar/vector, THRESHOLD for angular, etc.).
        scalar_vector_tol = compute_scalar_vector_tol(tasks)

        # --- optional fixed variance-normalized loss (env var LOSS_VARNORM=1) ---
        # OFF by default: the entry script and all existing runs are unaffected. When on,
        # we precompute a frozen per-task reference scale L_i(0) (the initial per-task loss,
        # ~= target variance) and divide each task's loss by it during training, so every
        # task contributes equally regardless of target scale.
        self._varnorm = os.environ.get("LOSS_VARNORM", "").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )

        task_ref = {}
        if self._varnorm:
            self.model.eval()
            _acc = {i: [] for i in range(len(tasks))}
            with torch.no_grad():
                for inputs, targets, masks, task_ids in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    masks = masks.to(self.device)
                    task_ids = task_ids.to(self.device)
                    outputs = self.model(inputs, task_ids)
                    me = masks.unsqueeze(-1).expand_as(outputs)
                    mo = outputs * me
                    if targets.dim() == 2:
                        mt = targets.unsqueeze(1).expand(-1, outputs.shape[1], -1) * me
                    else:
                        mt = targets * me
                    d = per_task_mse_loss(
                        mo, mt, masks, task_ids, len(tasks), task_loss_channels
                    )
                    for tid, v in d.items():
                        if v > 0:
                            _acc[tid].append(v)
            task_ref = {
                i: (float(np.mean(_acc[i])) if _acc[i] else 1.0)
                for i in range(len(tasks))
            }
            self.model.train()
            _vals = list(task_ref.values())
            print(
                f"[LOSS_VARNORM] fixed variance-normalized loss ON | per-task reference L_i(0) "
                f"for {len(task_ref)} tasks: min={min(_vals):.4f} max={max(_vals):.4f} "
                f"(max/min={max(_vals)/max(min(_vals),1e-9):.0f}x)",
                flush=True,
            )

        loss_history = []
        test_loss_history = []
        train_task_losses = {i: [] for i in range(len(tasks))}
        test_task_losses = {i: [] for i in range(len(tasks))}
        train_task_accuracies = {i: [] for i in range(len(tasks))}
        test_task_accuracies = {i: [] for i in range(len(tasks))}
        best_test_loss = float("inf")
        best_state = None
        best_epoch = 0
        # Parallel accuracy-optimal tracking. Stopping still uses test loss; this
        # only records/saves the epoch with the highest MEAN test accuracy so the
        # caller can report/checkpoint the accuracy-optimal model alongside it.
        best_mean_acc = -1.0
        best_state_acc = None
        best_acc_epoch = 0

        self.model.train()

        for epoch in trange(self.args.num_epochs):
            epoch_losses = {"rnn": 0, "hier": 0, "total": 0}
            epoch_max_gnorm = 0.0  # largest pre-clip grad norm seen this epoch
            epoch_nonfinite = 0  # count of skipped (exploding) batches this epoch
            # Init so an all-skipped (fully diverged) epoch logs nan and lets the
            # cool-down below recover, instead of an UnboundLocalError at logging.
            avg_loss = float("nan")
            self._apply_lr(epoch)  # warmup ramp * adaptive scale
            epoch_task_loss_accum = {i: [] for i in range(len(tasks))}
            epoch_task_correct = {i: 0.0 for i in range(len(tasks))}
            epoch_task_total = {i: 0 for i in range(len(tasks))}
            for inputs, targets, masks, task_ids in train_loader:
                # moving data to the same device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                task_ids = task_ids.to(self.device)

                # clearing up gradients from previous batch
                self.shared_optimizer.zero_grad()
                self.individual_optimizer.zero_grad()

                # forward pass: (batch, T, N)
                outputs = self.model(inputs, task_ids)

                # Apply mask to select which timesteps to evaluate
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

                if self._varnorm:
                    task_loss = variance_normalized_mse_loss(
                        masked_outputs,
                        masked_targets,
                        masks,
                        task_ids,
                        task_loss_channels,
                        task_ref,
                    )
                else:
                    task_loss = normalized_mse_loss(
                        masked_outputs,
                        masked_targets,
                        masks,
                        task_ids,
                        task_loss_channels,
                    )

                reg_loss = self.regularization_loss(self.args.tau, self.args.M_reg)
                loss = task_loss + reg_loss

                # Non-finite guard: if the loss blew up (inf/nan), skip this batch
                # entirely so a single spike can't corrupt the weights with NaN.
                if not torch.isfinite(loss):
                    epoch_nonfinite += 1
                    self.shared_optimizer.zero_grad()
                    self.individual_optimizer.zero_grad()
                    continue

                (loss).backward()

                # Gradient clipping (global norm over all params). clip_grad_norm_
                # returns the PRE-clip norm, so we log it to calibrate the threshold.
                grad_clip = getattr(self.args, "grad_clip", 0.0)
                if grad_clip and grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )
                    if not torch.isfinite(gn):
                        # exploding gradient -> skip the step, don't corrupt weights
                        epoch_nonfinite += 1
                        self.shared_optimizer.zero_grad()
                        self.individual_optimizer.zero_grad()
                        continue
                    epoch_max_gnorm = max(epoch_max_gnorm, float(gn))

                self.shared_optimizer.step()
                self.individual_optimizer.step()

                # Accumulate losses
                epoch_losses["rnn"] += task_loss.item()
                epoch_losses["hier"] += 0
                epoch_losses["total"] += loss.item()

                # Per-task losses
                batch_task_losses = per_task_mse_loss(
                    masked_outputs,
                    masked_targets,
                    masks,
                    task_ids,
                    len(tasks),
                    task_loss_channels,
                )
                for tid, val in batch_task_losses.items():
                    epoch_task_loss_accum[tid].append(val)

                with torch.no_grad():
                    if targets.dim() == 2:
                        targets_exp = targets.unsqueeze(1).expand_as(outputs)
                    else:
                        targets_exp = targets

                    # Same scorer as the test-side compute_accuracies (incl. the
                    # Yang-2019 fixation gate), so train and test accuracy use
                    # the identical criterion and cannot drift apart.
                    for tid_i, acc in compute_batch_accuracies(
                        outputs,
                        targets_exp,
                        masks,
                        task_ids,
                        binary_task_ids,
                        scalar_task_ids,
                        copytask_ids,
                        argmax_task_info,
                        task_thresholds,
                        flipflop_task_info=flipflop_task_info,
                        perstep_binary_task_ids=perstep_binary_task_ids,
                        reaction_task_info=reaction_task_info,
                        vector_task_info=vector_task_info,
                        scalar_vector_tol=scalar_vector_tol,
                    ):
                        epoch_task_correct[tid_i] += acc
                        epoch_task_total[tid_i] += 1

            avg_loss = epoch_losses["total"] / max(len(train_loader) - epoch_nonfinite, 1)
            loss_history.append(avg_loss)

            # Adaptive cool-down: if any batch exploded this epoch, halve the LR
            # scale so the run recovers and continues instead of blowing up. The
            # weights were never corrupted (the bad batches were skipped above).
            if epoch_nonfinite > 0:
                self.lr_scale = max(self.lr_scale * 0.5, 1e-3)
                if verbose:
                    print(
                        f"\n[stability] epoch {epoch + 1}: {epoch_nonfinite} "
                        f"non-finite batch(es) skipped; lr_scale -> {self.lr_scale:.4g}"
                    )

            # Compute train task accuracies and losses
            for task_id in range(len(tasks)):
                if epoch_task_total[task_id] > 0:
                    acc = epoch_task_correct[task_id] / epoch_task_total[task_id]
                    train_task_accuracies[task_id].append(acc)
                else:
                    train_task_accuracies[task_id].append(0)
                vals = epoch_task_loss_accum[task_id]
                train_task_losses[task_id].append(np.mean(vals) if vals else 0.0)

            # Test evaluation: EVERY epoch (unconditional). Test loss and accuracy are
            # measured once per epoch, so best_epoch is exact rather than quantised to
            # eval_interval. eval_interval no longer gates the test evaluation.
            do_eval = test_loader is not None
            if do_eval:
                # With LOSS_VARNORM=1 the test loss uses the SAME variance-normalized
                # objective as training (same frozen task_ref), so early stopping and
                # best-epoch selection monitor the objective being optimized.
                test_loss = compute_loss(
                    self.model,
                    test_loader,
                    self.device,
                    self.args.tau,
                    self.args.M_reg,
                    task_loss_channels,
                    task_ref=task_ref if self._varnorm else None,
                )
                test_loss_history.append(test_loss)
                epoch_test_task_losses = compute_per_task_loss(
                    self.model,
                    test_loader,
                    self.device,
                    len(tasks),
                    task_loss_channels,
                )
                for tid in range(len(tasks)):
                    test_task_losses[tid].append(epoch_test_task_losses[tid])
                test_accs = compute_accuracies(
                    self.model,
                    test_loader,
                    self.device,
                    len(tasks),
                    task_thresholds,
                    tasks=tasks,
                    scalar_vector_tol=scalar_vector_tol,
                )
                for task_id in range(len(tasks)):
                    test_task_accuracies[task_id].append(test_accs[task_id])

                # --- accuracy-optimal checkpoint (parallel to the loss one) ---
                # Track the highest mean test accuracy independently of the loss.
                mean_test_acc_now = float(
                    np.mean([test_accs[i] for i in range(len(tasks))])
                )
                if mean_test_acc_now > best_mean_acc:
                    best_mean_acc = mean_test_acc_now
                    best_acc_epoch = epoch + 1
                    if checkpoint_acc_path is not None:
                        best_state_acc = deepcopy(self.model.state_dict())
                        torch.save(best_state_acc, checkpoint_acc_path)

                current_test_loss = test_loss_history[-1]
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_state = deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                    # Periodic checkpoint: persist the best model so a walltime
                    # kill still leaves a usable model on disk.
                    if checkpoint_path is not None:
                        torch.save(best_state, checkpoint_path)

                # Early stopping, measured in EPOCHS (independent of eval_interval),
                # but only allowed AFTER a warmup of `early_stopping_start` epochs
                # (default 0 = stop from the start, i.e. unchanged behavior). The
                # best model is still tracked from epoch 0; only the *stopping* is
                # gated -> "train at least `start` epochs, then patience applies".
                es_start = getattr(self.args, "early_stopping_start", 0)
                if (epoch + 1) >= es_start and (
                    (epoch + 1) - best_epoch >= self.args.early_stopping_patience
                ):
                    print(
                        f"\nEarly stopping at epoch {epoch + 1} (no test-loss "
                        f"improvement for {self.args.early_stopping_patience} epochs; "
                        f"warmup={es_start})"
                    )
                    break

            if verbose:
                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.args.num_epochs}, Loss: {avg_loss:.4f}"
                        f", gradmax: {epoch_max_gnorm:.2f}",
                        end="",
                    )
                    if test_loader is not None and test_loss_history:
                        mean_test_acc = np.mean(
                            [test_task_accuracies[i][-1] for i in range(len(tasks))]
                        )
                        print(
                            f", Test Loss: {test_loss_history[-1]:.4f}, Test Acc: {mean_test_acc:.2%}",
                            end="",
                        )
                    print()
                    max_label_len = max(
                        len(task_names[tid] if task_names else f"Task {tid}")
                        for tid in train_task_accuracies
                    )
                    for task_id, acc_list in train_task_accuracies.items():
                        label = task_names[task_id] if task_names else f"Task {task_id}"
                        print(
                            f"  {label:<{max_label_len}} Train Acc: {acc_list[-1]:>7.2%}",
                            end="",
                        )
                        if test_loader is not None and test_loss_history:
                            print(
                                f", Test Acc: {test_task_accuracies[task_id][-1]:>7.2%}",
                                end="",
                            )
                        print()

        # Expose accuracy-optimal selection to the caller (for metadata/reporting).
        # The RETURNED model is still the loss-best checkpoint (unchanged behavior);
        # the accuracy-best weights live on disk at checkpoint_acc_path.
        self.best_acc_epoch = best_acc_epoch
        self.best_mean_acc = best_mean_acc

        if best_state is not None:
            # restore best-performing checkpoint (by test loss)
            self.model.load_state_dict(best_state)

        return (
            loss_history,
            train_task_accuracies,
            test_loss_history,
            test_task_accuracies,
            train_task_losses,
            test_task_losses,
            best_epoch,
            outputs.detach().cpu().numpy(),
        )

    def regularization_loss(self, tau, M_reg):
        """
        Compute the regularization loss for each subject individually.
        Regularizes the first M_reg units per subject.
        Encourages W_ii to be close to 1, off-diagonal W_ij to be close to 0, and biases h_i to be close to 0.

        Args:
            tau: Regularization strength (float).
            M_reg: Number of units to regularize (int).
        Returns:
            The combined regularization loss averaged over subjects (torch scalar).
        """
        num_subjects = self.args.num_subjects
        M = self.model.M
        L = self.model.L
        linear = M - L

        # Vectorized over all subjects (same math as the per-subject loop, but
        # one get_parameters call + batched tensor ops instead of an
        # O(num_subjects * M_reg) Python loop).
        subjects = torch.arange(num_subjects, device=self.device)
        A, W, h, _, _ = self.model.hierarchisation_scheme.get_parameters(subjects)
        # A: (S, L)   W: (S, M, M)   h: (S, M)
        idx = torch.arange(M_reg, device=self.device)

        W_diag = W[:, idx, idx]  # (S, M_reg) diagonal entries
        diag_eff = W_diag.clone()
        nl = idx[idx >= linear]  # reg units that are nonlinear
        if nl.numel() > 0:  # add their A self-recurrence
            diag_eff[:, nl] = diag_eff[:, nl] + A[:, nl - linear]
        diag_loss = ((diag_eff - 1.0) ** 2).sum(dim=1)  # (S,)

        row_sq = (W[:, idx, :] ** 2).sum(dim=2)  # (S, M_reg) full-row sums
        offdiag_loss = (row_sq - W_diag**2).sum(dim=1)  # (S,)

        bias_loss = (h[:, idx] ** 2).sum(dim=1)  # (S,)

        loss_per_subject = diag_loss + offdiag_loss + bias_loss
        return tau * loss_per_subject.mean()
