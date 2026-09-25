from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class TemporalBisection(CognitiveTask):
    """
    Temporal bisection (Church & Gibbon 1982; Leon & Shadlen 2003).

    A single stimulus is presented for a sample duration d; after it ends the
    network classifies that duration as 'short' (0) or 'long' (1) relative to a
    fixed, learned midpoint. This is *categorical* timing with a learned
    decision boundary -- distinct from reproducing an interval as a scalar
    (IntervalReproduction) or comparing two intervals (IntervalDiscrimination).

    Input channels (2): [fixation, stimulus].  (stimulus held on for d steps)
    Output: channel 0 = fixation; channel 3 = short/long decision (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, dur_range=(4, 20),
                 midpoint: float = 11.5):
        super().__init__(duration_params)
        self.dur_range = dur_range
        self.midpoint = midpoint
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        d = int(np.random.randint(*self.dur_range))
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + d + T_response

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + d, 1] = 1.0  # stimulus on for d steps
        t_resp = T_context + d
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if d > self.midpoint else 0.0  # long vs short

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, d, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Td, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Td, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
