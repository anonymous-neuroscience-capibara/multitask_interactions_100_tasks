"""Held-out NOVEL tasks for the p_vector transfer experiment.

Four new tasks, each a deliberate sibling of a trained family but with a
mapping the foundation battery never trained on, spanning a gradient of
"novelty distance":

    AngleReflect : report the MIRRORED direction, theta -> -theta.
                   Nearest family: DelayPro/DelayAnti. But a reflection is NOT
                   a rotation (cannot be written as theta + const), so it is
                   not an interpolation of Pro (0 deg) and Anti (180 deg).
    AngleAverage : TWO sequential direction cues; report their circular mean.
                   Novel combination over two remembered items (SpatialWMRetro
                   holds two items but selects one; nothing averages).
    ScalarInvert : an analog level s in [0, 1]; respond 1 - s after a delay.
                   Novel input->output remapping in the scalar family
                   (IntervalReproduction / Numerosity read out scalars, none
                   inverts one).
    FreqDouble   : SineGen's frequency cue, but oscillate at DOUBLE the cued
                   omega. Tests whether one p_vector can RETUNE a limit cycle.

All conform to the shared interface (inputs padded to BASE_INPUT_DIM=7,
OUTPUT_DIM=5) and reuse existing accuracy branches -- no trainer changes:
    AngleReflect / AngleAverage : default angular (last masked step, ch 1,2)
    ScalarInvert                : is_scalar_task (ch 3, NON_ANGULAR_THRESHOLD)
    FreqDouble                  : is_copytask (per-timestep angular, ch 1,2)

None of these are added to FOUNDATION_TASKS_MAP -- the trained battery (now 100
tasks) never trains on them; they are imported by the fine-tune driver as the
held-out transfer task.
Like DelayRotate, none presents a rule cue: the task's identity has to live
entirely in its p_vector.
"""

from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask

from .delay_rotate import DelayRotate
from .angle_compose import AngleCompose


class AngleReflect(CognitiveTask):
    """Delayed response reporting the MIRRORED direction: theta -> -theta.

    Same context/stimulus/delay/response skeleton as DelayedResponse, no rule
    cue. Angular accuracy (default branch), loss_channels = [1, 2].
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_delay + T_response

        theta = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)
        inputs[:T_context, 0] = 1.0
        t0, t1 = T_context, T_context + T_stim
        inputs[t0:t1, 0] = 1.0
        inputs[t0:t1, 1] = np.cos(theta)
        inputs[t0:t1, 2] = np.sin(theta)
        inputs[t1:t1 + T_delay, 0] = 1.0
        t_resp = t1 + T_delay

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(-theta)  # = cos(theta)
        targets[2] = np.sin(-theta)  # = -sin(theta)

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_delay, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases


class AngleAverage(CognitiveTask):
    """Two sequential direction cues; report their circular mean after a delay.

    theta2 = theta1 + delta with |delta| <= 2.1 rad, so the circular mean is
    always well defined (never antipodal). Angular accuracy (default branch),
    loss_channels = [1, 2]. Both cues arrive on the SAME channels (1, 2), so
    the network must store the first before the second overwrites it.
    """

    PHASE_NAMES = ["context", "stim1", "gap", "stim2", "delay", "response"]

    def __init__(self, duration_params: Dict, max_delta: float = 2.1):
        super().__init__(duration_params)
        self.max_delta = float(max_delta)
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim1 = np.random.randint(*dp["stimulus"])
        T_gap = np.random.randint(3, 6)  # short gap separating the two cues
        T_stim2 = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim1 + T_gap + T_stim2 + T_delay + T_response

        theta1 = np.random.uniform(0, 2 * np.pi)
        delta = np.random.uniform(-self.max_delta, self.max_delta)
        theta2 = theta1 + delta
        # circular mean of the two directions
        mean_c = np.cos(theta1) + np.cos(theta2)
        mean_s = np.sin(theta1) + np.sin(theta2)
        theta_mean = np.arctan2(mean_s, mean_c)

        inputs = torch.zeros(T_total, 5)
        t = 0
        inputs[t:t + T_context, 0] = 1.0
        t += T_context
        inputs[t:t + T_stim1, 0] = 1.0
        inputs[t:t + T_stim1, 1] = np.cos(theta1)
        inputs[t:t + T_stim1, 2] = np.sin(theta1)
        t += T_stim1
        inputs[t:t + T_gap, 0] = 1.0
        t += T_gap
        inputs[t:t + T_stim2, 0] = 1.0
        inputs[t:t + T_stim2, 1] = np.cos(theta2)
        inputs[t:t + T_stim2, 2] = np.sin(theta2)
        t += T_stim2
        inputs[t:t + T_delay, 0] = 1.0
        t_resp = t + T_delay

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(theta_mean)
        targets[2] = np.sin(theta_mean)

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        durs = (T_context, T_stim1, T_gap, T_stim2, T_delay, T_response)
        return inputs, targets, mask, durs

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, durs = self._build()
        phases = np.concatenate([
            np.full(d, i, dtype=np.int64) for i, d in enumerate(durs)
        ])
        return inputs, targets, mask, phases


class ScalarInvert(CognitiveTask):
    """Hold an analog level s in [0.05, 0.95]; respond 1 - s after a delay.

    Input: level on channel 3 during the stimulus phase. Output: 1 - s on
    channel 3 (scalar readout). is_scalar_task accuracy: |out - target| <
    NON_ANGULAR_THRESHOLD (0.05) at the last masked step, so always-predict-0.5
    only scores when s is within 0.05 of 0.5 (~5% baseline).
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.loss_channels = [self.BINARY_CHANNEL]  # channel 3

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_delay + T_response

        s = np.random.uniform(0.05, 0.95)

        inputs = torch.zeros(T_total, 5)
        inputs[:T_context, 0] = 1.0
        t0, t1 = T_context, T_context + T_stim
        inputs[t0:t1, 0] = 1.0
        inputs[t0:t1, 3] = s
        inputs[t1:t1 + T_delay, 0] = 1.0
        t_resp = t1 + T_delay

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 - s

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_delay, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases


class FreqDouble(CognitiveTask):
    """SineGen sibling: same frequency cue, oscillate at DOUBLE the cued omega.

    Input channel 1 carries freq_level = omega / omega_max exactly as SineGen;
    the target phase advances at 2 * omega. The cue->frequency map the model
    learned for SineGen is therefore wrong by a factor of two -- can a single
    p_vector retune the limit cycle? is_copytask accuracy (per-timestep angular
    on channels 1, 2).
    """

    PHASE_NAMES = ["context", "response"]
    is_copytask = True

    def __init__(
        self,
        duration_params: Dict,
        n_freqs: int = 4,
        omega_range=(0.15, 0.6),  # SineGen's cue convention; output = 2 * omega
    ):
        super().__init__(duration_params)
        self.omegas = np.linspace(omega_range[0], omega_range[1], n_freqs)
        self.omega_max = float(omega_range[1])
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_response = max(np.random.randint(*dp["response"]), 20)
        T_total = T_context + T_response

        omega = float(np.random.choice(self.omegas))
        freq_level = omega / self.omega_max  # SAME cue scale as SineGen

        inputs = torch.zeros(T_total, 2)
        inputs[:T_context, 0] = 1.0
        inputs[:, 1] = freq_level

        targets = torch.zeros(T_total, self.output_dim)
        targets[:T_context, 0] = 1.0
        t = np.arange(T_response)
        phase = 2.0 * omega * t  # DOUBLE the cued frequency
        targets[T_context:, 1] = torch.from_numpy(np.cos(phase).astype(np.float32))
        targets[T_context:, 2] = torch.from_numpy(np.sin(phase).astype(np.float32))

        mask = torch.zeros(T_total)
        mask[T_context:] = 1.0
        return inputs, targets, mask, (T_context, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tr, 1, dtype=np.int64),
        ])
        return inputs, targets, mask, phases


class AngleQuadrant(CognitiveTask):
    """Remember a direction; after a delay, classify it into one of K equal
    angular sectors (default K=4 quadrants). ARGMAX / categorical output.

    Novel because it is the only MULTI-way categorical task: the trained
    categorical tasks (CatPro/CatAnti, DMC/DNMC) are all BINARY, and the single
    trained argmax task (MetacognitiveOptOut, 3 classes) is a confidence readout,
    not an angular sector map. Tests whether the frozen primitives -- which hold
    a direction well (DelayPro etc.) -- can also CARVE a remembered angle into K
    discrete category channels.

    Input channels (3): [fixation, stim_cos, stim_sin]  (no rule cue).
    Output: argmax over the last K channels (K=4 -> channels 1,2,3,4).
    Accuracy: argmax over loss_channels at the last masked step (chance = 1/K).
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict, n_classes: int = 4):
        super().__init__(duration_params)
        self.is_argmax_task = True
        self.n_classes = int(n_classes)
        self.loss_channels = list(range(-self.n_classes, 0))  # K=4 -> [1,2,3,4]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_delay + T_response

        theta = np.random.uniform(0, 2 * np.pi)
        cls = int(theta // (2 * np.pi / self.n_classes)) % self.n_classes

        inputs = torch.zeros(T_total, 3)
        inputs[:T_context, 0] = 1.0
        t0, t1 = T_context, T_context + T_stim
        inputs[t0:t1, 0] = 1.0
        inputs[t0:t1, 1] = np.cos(theta)
        inputs[t0:t1, 2] = np.sin(theta)
        inputs[t1:t1 + T_delay, 0] = 1.0
        t_resp = t1 + T_delay

        targets = torch.zeros(self.output_dim)
        targets[self.loss_channels[cls]] = 1.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_delay, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases


class _ArgmaxBase(CognitiveTask):
    """Shared machinery for categorical (argmax) held-out tasks.

    Subclasses set self.n_classes and implement _make(): return
    (inputs[T, C_in], class:int, durs:tuple). This base wraps it into the
    argmax convention: one-hot target on channel loss_channels[class],
    accuracy = argmax over the last n_classes channels at the last masked step.
    """

    def __init__(self, duration_params: Dict, n_classes: int):
        super().__init__(duration_params)
        self.is_argmax_task = True
        self.n_classes = int(n_classes)
        self.loss_channels = list(range(-self.n_classes, 0))

    def _make(self):
        raise NotImplementedError

    def _build(self):
        inputs, cls, durs = self._make()
        targets = torch.zeros(self.output_dim)
        targets[self.loss_channels[cls]] = 1.0
        mask = torch.zeros(inputs.shape[0])
        mask[-durs[-1]:] = 1.0  # response phase is the last block
        return inputs, targets, mask, durs

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, durs = self._build()
        phases = np.concatenate([np.full(d, i, dtype=np.int64) for i, d in enumerate(durs)])
        return inputs, targets, mask, phases


class MagnitudeTertile(_ArgmaxBase):
    """Show an analog level s in [0,1]; after a delay classify it low/mid/high
    (3-way). Categorical readout of a remembered magnitude (cf. ScalarInvert,
    which reports the scalar; here the model must BIN it)."""

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params, n_classes=3)

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"]); Ts = np.random.randint(*dp["stimulus"])
        Td = np.random.randint(*dp["delay"]); Tr = np.random.randint(*dp["response"])
        s = np.random.uniform(0.02, 0.98)
        cls = min(int(s * 3), 2)  # tertiles
        inputs = torch.zeros(Tc + Ts + Td + Tr, 2)
        inputs[:Tc + Ts + Td, 0] = 1.0
        inputs[Tc:Tc + Ts, 1] = s
        return inputs, cls, (Tc, Ts, Td, Tr)


class DurationTertile(_ArgmaxBase):
    """A stimulus is on for d steps; after it ends classify the duration
    short/mid/long (3-way). Categorical TIMING (cf. TemporalBisection, binary;
    IntervalReproduction, scalar)."""

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, dur_range=(4, 22)):
        super().__init__(duration_params, n_classes=3)
        self.dur_range = dur_range

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"]); Tr = np.random.randint(*dp["response"])
        lo, hi = self.dur_range
        d = int(np.random.randint(lo, hi))
        cls = min(int((d - lo) / (hi - lo) * 3), 2)
        inputs = torch.zeros(Tc + d + Tr, 2)
        inputs[:Tc + d, 0] = 1.0
        inputs[Tc:Tc + d, 1] = 1.0  # stimulus held on for d steps
        return inputs, cls, (Tc, d, Tr)


class CountBin(_ArgmaxBase):
    """Count pulses in a window; after a delay classify the count into 3 bins
    (0-2 / 3-5 / 6-8). Categorical evidence accumulation (cf. Numerosity,
    scalar; PoissonClicks, binary)."""

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params, n_classes=3)

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = max(np.random.randint(*dp["stimulus"]), 9)
        Td = np.random.randint(*dp["delay"]); Tr = np.random.randint(*dp["response"])
        n_pulses = int(np.random.randint(0, 9))  # 0..8 -> balanced 3 bins
        cls = min(n_pulses // 3, 2)
        inputs = torch.zeros(Tc + Ts + Td + Tr, 2)
        inputs[:Tc + Ts + Td, 0] = 1.0
        if n_pulses > 0:
            pos = np.random.choice(Ts, size=n_pulses, replace=False)
            inputs[Tc + pos, 1] = 1.0
        return inputs, cls, (Tc, Ts, Td, Tr)


class ParityClass(_ArgmaxBase):
    """Count pulses in a window; after a delay report the count's PARITY
    (even/odd, 2-way). A modular function of the count -- a hard nonlinear
    accumulation nothing in the battery does."""

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params, n_classes=2)

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = max(np.random.randint(*dp["stimulus"]), 9)
        Td = np.random.randint(*dp["delay"]); Tr = np.random.randint(*dp["response"])
        n_pulses = int(np.random.randint(0, 9))
        cls = n_pulses % 2
        inputs = torch.zeros(Tc + Ts + Td + Tr, 2)
        inputs[:Tc + Ts + Td, 0] = 1.0
        if n_pulses > 0:
            pos = np.random.choice(Ts, size=n_pulses, replace=False)
            inputs[Tc + pos, 1] = 1.0
        return inputs, cls, (Tc, Ts, Td, Tr)


class MaxOfK(_ArgmaxBase):
    """Four noisy evidence channels; report which had the largest mean (4-way
    K-alternative forced choice). Generalizes 2-alternative decisions (binary
    PerceptualDM) to a K-way argmax selection."""

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, noise: float = 0.5, gap: float = 0.35):
        super().__init__(duration_params, n_classes=4)
        self.noise = noise
        self.gap = gap

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"]); Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        means = np.random.uniform(0, 1, size=4)
        cls = int(np.argmax(means))
        # ensure the winner is clearly ahead (avoid ambiguous trials)
        means[cls] = means.max() + self.gap
        inputs = torch.zeros(Tc + Ts + Tr, 5)
        inputs[:Tc + Ts, 0] = 1.0
        ev = means[None, :] + np.random.normal(0, self.noise, size=(Ts, 4))
        inputs[Tc:Tc + Ts, 1:5] = torch.from_numpy(ev.astype(np.float32))
        return inputs, cls, (Tc, Ts, Tr)


class StimulusIdentity(_ArgmaxBase):
    """One of 4 discrete cue patterns (one-hot on inputs) during the stimulus;
    after a delay report which cue it was (4-way delayed match-to-identity).
    Pure identity memory across a delay."""

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params, n_classes=4)

    def _make(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"]); Ts = np.random.randint(*dp["stimulus"])
        Td = np.random.randint(*dp["delay"]); Tr = np.random.randint(*dp["response"])
        cls = int(np.random.randint(0, 4))
        inputs = torch.zeros(Tc + Ts + Td + Tr, 5)
        inputs[:Tc + Ts + Td, 0] = 1.0
        inputs[Tc:Tc + Ts, 1 + cls] = 1.0  # one-hot cue on channels 1..4
        return inputs, cls, (Tc, Ts, Td, Tr)


# Registry used by the fine-tune driver: name -> factory(duration_params).
# DelayRotate variants included so one flag selects any held-out task.
HELDOUT_TASKS = {
    "DelayRotate90": lambda dp: DelayRotate(dp, rotation_deg=90.0),
    "DelayRotate45": lambda dp: DelayRotate(dp, rotation_deg=45.0),
    "DelayRotate135": lambda dp: DelayRotate(dp, rotation_deg=135.0),
    "AngleReflect": lambda dp: AngleReflect(dp),
    "AngleAverage": lambda dp: AngleAverage(dp),
    "AngleCompose": lambda dp: AngleCompose(dp),  # bilinear: report theta1 + theta2
    "ScalarInvert": lambda dp: ScalarInvert(dp),
    "FreqDouble": lambda dp: FreqDouble(dp),
    "AngleQuadrant": lambda dp: AngleQuadrant(dp, n_classes=4),      # 4-way spatial sector
    "MagnitudeTertile": lambda dp: MagnitudeTertile(dp),            # 3-way magnitude bin
    "DurationTertile": lambda dp: DurationTertile(dp),             # 3-way timing bin
    "CountBin": lambda dp: CountBin(dp),                            # 3-way count bin
    "ParityClass": lambda dp: ParityClass(dp),                     # 2-way count parity
    "MaxOfK": lambda dp: MaxOfK(dp),                               # 4-way K-alt decision
    "StimulusIdentity": lambda dp: StimulusIdentity(dp),          # 4-way identity memory
}

# The categorical (argmax) held-out tasks, for convenience.
ARGMAX_HELDOUT = [
    "AngleQuadrant", "MagnitudeTertile", "DurationTertile",
    "CountBin", "ParityClass", "MaxOfK", "StimulusIdentity",
]
