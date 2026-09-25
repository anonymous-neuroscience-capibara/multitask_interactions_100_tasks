from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class OneTwoThreeGo(CognitiveTask):
    """
    One-Two-Three-Go (Sohn, Narain, Meirhaeghe & Jazayeri, 2019).

    Three flashes are presented at a fixed inter-flash interval, establishing a
    rhythm; the network must estimate that interval (predicting when the "Go"
    would fall) and report it as a scalar. Averaging across the three flashes
    gives a more reliable estimate than a single interval -- a timing-integration
    task distinct from the two-flash IntervalReproduction.

    Input channels (2): [fixation, flash].
    Output: channel 3 (scalar readout) = inter-flash interval (normalised).
    """

    PHASE_NAMES = ["context", "rhythm", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict, interval_range=(4, 16)):
        super().__init__(duration_params)
        self.interval_range = interval_range
        self.imax = float(interval_range[1])
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        d = int(np.random.randint(*self.interval_range))
        T_response = np.random.randint(*dp["response"])
        T_rhythm = 2 * d + 1  # flashes at 0, d, 2d
        T_total = T_context + T_rhythm + T_response

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        for k in range(3):
            inputs[T_context + k * d, 1] = 1.0
        t_resp = T_context + T_rhythm
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = d / self.imax

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_rhythm, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Trh, Tr) = self._build()
        phases = np.concatenate(
            [
                np.full(Tc, 0, dtype=np.int64),
                np.full(Trh, 1, dtype=np.int64),
                np.full(Tr, 2, dtype=np.int64),
            ]
        )
        return inputs, targets, mask, phases
