from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class IntervalReproduction(CognitiveTask):
    """
    Interval reproduction (Jazayeri & Shadlen, 2010).

    Two flashes mark a sample interval; after a go cue the network reports the
    measured interval as a scalar. Requires timing the gap between flashes and
    holding the estimate -- the supervised, scalar-readout form of "produce the
    same interval".

    Input channels (3): [fixation, flash, go_cue].
    Output: channel 3 (scalar readout) = measured interval (normalised to [0,1]).
    """

    PHASE_NAMES = ["context", "interval", "go", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict, interval_range=(4, 20)):
        super().__init__(duration_params)
        self.interval_range = interval_range
        self.imax = float(interval_range[1])
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        d = int(np.random.randint(*self.interval_range))  # sample interval
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + d + 1 + T_response

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context, 1] = 1.0          # first flash
        inputs[T_context + d, 1] = 1.0      # second flash (interval = d)
        t_go = T_context + d + 1
        inputs[t_go - 0:t_go, 2] = 1.0      # go cue just before response
        inputs[t_go:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = d / self.imax  # normalised interval

        mask = torch.zeros(T_total)
        mask[t_go:] = 1.0
        return inputs, targets, mask, (T_context, d, 1, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, d, Tg, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(d, 1, dtype=np.int64),
            np.full(Tg, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
