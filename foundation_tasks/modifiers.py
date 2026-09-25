"""Task modifiers: wrappers that turn a base CognitiveTask into a harder variant.

A modifier takes an existing task and produces a NEW task (its own task_id when
registered) by surgically editing the trial structure. Write the rule once,
apply it to many base tasks -- this is how the battery scales (Mod-Cog int/seq
idea). The wrapped task keeps the base's accuracy type and loss channels, so the
training/eval pipeline treats it identically.

Implemented here: DelayModifier (+delay). Others (+seq, +int) follow the same
shape and will be added next.
"""

from typing import Optional

import numpy as np
import torch

from tasks.dataset import CognitiveTask

# Accuracy/eval attributes that must follow the base task through the wrapper.
_INHERITED_FLAGS = [
    "is_binary_task",
    "is_scalar_task",
    "is_copytask",
    "is_argmax_task",
    "is_flipflop_task",
    "is_population_task",
]


class TaskModifier(CognitiveTask):
    """Base class for modifiers. Wraps a base task and inherits its eval config."""

    def __init__(self, base_task: CognitiveTask):
        self.base = base_task
        self.duration_params = base_task.duration_params
        self.output_dim = base_task.output_dim
        # Inherit accuracy type + loss channels so the pipeline scores it the same.
        self.loss_channels = getattr(base_task, "loss_channels", None)
        for flag in _INHERITED_FLAGS:
            setattr(self, flag, getattr(base_task, flag, False))

    def generate_trial(self):
        raise NotImplementedError

    def generate_trial_with_phases(self):
        # Default: run generate_trial and label everything as one phase.
        inputs, targets, mask = self.generate_trial()
        phases = np.zeros(inputs.shape[0], dtype=np.int64)
        return inputs, targets, mask, phases


class DelayModifier(TaskModifier):
    """+delay : insert a blank retention gap just before the response window.

    The gap (fixation on, stimulus off, mask = 0) is spliced in right before the
    first evaluated timestep, forcing the network to hold its pre-response state
    across the delay. Works for both single-target and sequential-target tasks.
    """

    PHASE_NAMES = ["base", "delay", "response"]

    def __init__(self, base_task: CognitiveTask, delay_range=(10, 20)):
        super().__init__(base_task)
        self.delay_range = delay_range

    def _insert_delay(self, inputs, targets, mask):
        T, d = inputs.shape
        masked = mask > 0
        # Response onset = first evaluated timestep (append at end if none).
        s = int(torch.where(masked)[0][0]) if masked.any() else T
        T_delay = int(np.random.randint(*self.delay_range))

        # Delay input: fixation channel on, everything else off.
        delay_in = torch.zeros(T_delay, d)
        if d > 0:
            delay_in[:, 0] = 1.0
        new_inputs = torch.cat([inputs[:s], delay_in, inputs[s:]], dim=0)

        delay_mask = torch.zeros(T_delay)
        new_mask = torch.cat([mask[:s], delay_mask, mask[s:]], dim=0)

        if targets.dim() == 1:
            new_targets = targets  # single held target is unchanged
        else:
            delay_tgt = torch.zeros(T_delay, targets.shape[1])
            delay_tgt[:, 0] = 1.0  # hold fixation during the gap (not scored: mask=0)
            new_targets = torch.cat([targets[:s], delay_tgt, targets[s:]], dim=0)

        return new_inputs, new_targets, new_mask

    def generate_trial(self):
        inputs, targets, mask = self.base.generate_trial()
        return self._insert_delay(inputs, targets, mask)


def with_delay(base_task: CognitiveTask, delay_range=(10, 20)) -> DelayModifier:
    """Convenience: wrap a base task with +delay."""
    return DelayModifier(base_task, delay_range=delay_range)


def _is_angular(base_task: CognitiveTask) -> bool:
    """True if the base reports a direction on output channels 1,2 (cos/sin)."""
    return getattr(base_task, "loss_channels", None) == [1, 2]


def _assert_has_spare_channel(base_task: CognitiveTask):
    """+seq/+int append an omega-cue channel; the base must have room within
    BASE_INPUT_DIM. Fail clearly at construction rather than at training time."""
    inp, _, _ = base_task.generate_trial()
    if inp.shape[1] + 1 > CognitiveTask.BASE_INPUT_DIM:
        raise ValueError(
            f"{type(base_task).__name__} uses {inp.shape[1]} input channels; "
            f"adding the modifier cue would exceed BASE_INPUT_DIM="
            f"{CognitiveTask.BASE_INPUT_DIM}."
        )


class SeqModifier(TaskModifier):
    """+seq : turn a static angular response into a rotating output trajectory.

    The reported direction drifts at angular speed omega over the response
    window, so the output traces a path on the ring (Mod-Cog 'seq'). A cue for
    omega is added on a fresh input channel. Only valid for angular bases; the
    accuracy switches to the per-timestep angular metric.
    """

    PHASE_NAMES = ["base", "response"]

    def __init__(self, base_task: CognitiveTask, omega_range=(-0.3, 0.3)):
        if not _is_angular(base_task):
            raise ValueError("SeqModifier (+seq) only applies to angular bases.")
        _assert_has_spare_channel(base_task)
        super().__init__(base_task)
        self.omega_range = omega_range
        self.is_copytask = True  # per-timestep angular accuracy
        self.loss_channels = [1, 2]

    def generate_trial(self):
        inputs, targets, mask = self.base.generate_trial()
        T, d = inputs.shape
        omega = float(np.random.uniform(*self.omega_range))

        cue = torch.full((T, 1), omega)
        inputs = torch.cat([inputs, cue], dim=1)  # omega cue on a fresh channel

        # Base must be single-target angular; rotate it over the response window.
        theta = float(np.arctan2(float(targets[2]), float(targets[1])))
        seq = torch.zeros(T, self.output_dim)
        masked_idx = torch.where(mask > 0)[0]
        for k, t in enumerate(masked_idx.tolist()):
            ang = theta + omega * k
            seq[t, 1] = np.cos(ang)
            seq[t, 2] = np.sin(ang)
        return inputs, seq, mask


class IntModifier(TaskModifier):
    """+int : insert a delay during which the remembered direction rotates.

    A delay is spliced in before the response (fixation on, omega cue present);
    the reported direction is the *integrated* angle theta + omega * T_delay.
    Because the delay length varies, the network must integrate angular velocity
    over the delay -- demanding genuine rotational dynamics (Mod-Cog 'int').
    Keeps the single-target angular accuracy (report at the end).
    """

    PHASE_NAMES = ["base", "delay", "response"]

    def __init__(self, base_task: CognitiveTask, delay_range=(8, 18),
                 omega_range=(-0.2, 0.2)):
        if not _is_angular(base_task):
            raise ValueError("IntModifier (+int) only applies to angular bases.")
        _assert_has_spare_channel(base_task)
        super().__init__(base_task)
        self.delay_range = delay_range
        self.omega_range = omega_range
        self.loss_channels = [1, 2]  # angular, last-timestep accuracy (default)

    def generate_trial(self):
        inputs, targets, mask = self.base.generate_trial()
        T, d = inputs.shape
        masked = mask > 0
        s = int(torch.where(masked)[0][0]) if masked.any() else T
        T_delay = int(np.random.randint(*self.delay_range))
        omega = float(np.random.uniform(*self.omega_range))

        # Splice a delay (fixation on) before the response.
        delay_in = torch.zeros(T_delay, d)
        if d > 0:
            delay_in[:, 0] = 1.0
        new_inputs = torch.cat([inputs[:s], delay_in, inputs[s:]], dim=0)
        # omega cue on a fresh channel for the whole trial.
        cue = torch.full((new_inputs.shape[0], 1), omega)
        new_inputs = torch.cat([new_inputs, cue], dim=1)

        delay_mask = torch.zeros(T_delay)
        new_mask = torch.cat([mask[:s], delay_mask, mask[s:]], dim=0)

        # Reported direction = base angle integrated over the delay.
        theta = float(np.arctan2(float(targets[2]), float(targets[1])))
        new_theta = theta + omega * T_delay
        new_targets = targets.clone() if targets.dim() == 1 else targets
        if new_targets.dim() == 1:
            new_targets = new_targets.clone()
            new_targets[1] = np.cos(new_theta)
            new_targets[2] = np.sin(new_theta)
        return new_inputs, new_targets, new_mask


def with_seq(base_task: CognitiveTask, omega_range=(-0.3, 0.3)) -> SeqModifier:
    return SeqModifier(base_task, omega_range=omega_range)


def with_int(base_task: CognitiveTask, **kw) -> IntModifier:
    return IntModifier(base_task, **kw)
