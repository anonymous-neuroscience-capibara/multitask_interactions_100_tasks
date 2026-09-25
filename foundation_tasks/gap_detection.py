from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class GapDetection(CognitiveTask):
    """
    Gap detection / temporal acuity (Phillips 1999; Green 1971).

    An ongoing stimulus is presented at a steady level over the stimulus window.
    On half the trials a brief silent GAP (the level drops to ~0 for a few
    timesteps) is inserted at a random position; on the other half the stream is
    continuous. The network reports whether a gap occurred. A classic measure of
    auditory temporal resolution -- transient within-stream change detection,
    distinct from onset detection (SignalDetection) and order/coincidence
    (Temporal-Order / Simultaneity). No random tokens -> generalizes readily.

    Input channels (2): [fixation, stimulus].
    Output: channel 3 = "gap present" (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, gap_w: int = 3, level: float = 1.0,
                 noise: float = 0.05):
        super().__init__(duration_params)
        self.gap_w = gap_w
        self.level = level
        self.noise = noise
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        Ts = max(Ts, self.gap_w + 4)
        T_total = Tc + Ts + Tr

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0

        stim = self.level + np.random.normal(0, self.noise, size=Ts)
        gap = np.random.rand() < 0.5
        if gap:
            g0 = np.random.randint(1, Ts - self.gap_w - 1)
            stim[g0:g0 + self.gap_w] = np.random.normal(0, self.noise, size=self.gap_w)
        inputs[Tc:Tc + Ts, 1] = torch.from_numpy(stim.astype(np.float32))

        t_resp = Tc + Ts
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if gap else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, Ts, Tr)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
