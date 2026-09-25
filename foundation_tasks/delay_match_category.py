from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class DelayMatchCategory(CognitiveTask):
    """
    Delay match-to-category (Freedman & Assad 2006; Yang19 dmc/dmcgo/dmcnogo).

    A sample direction is shown, held over a delay, then a test direction; the
    network reports whether the test is in the *same category* as the sample
    (not the same exact direction -- the key difference from DMS). Directions
    fall into two categories (two opposite halves of the ring), so the network
    must abstract a category boundary and hold it across the delay.

    Modes: 'match' -> respond on same-category; 'nonmatch' -> respond on
    different-category (mirrors DelayedMatchToSample).

    Input channels (3): [fixation, stim_cos, stim_sin].
    Output: channel 0 = fixation; channel 3 = decision (binary).
    """

    PHASE_NAMES = ["context", "sample", "delay", "test", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, mode="match", spread=np.pi / 3):
        super().__init__(duration_params)
        self.mode = mode
        self.spread = spread
        self.centers = [np.pi / 2, 3 * np.pi / 2]  # two category centers
        self.loss_channels = [self.BINARY_CHANNEL]

    def _angle(self, cat):
        return self.centers[cat] + np.random.uniform(-self.spread, self.spread)

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim1 = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_stim2 = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim1 + T_delay + T_stim2 + T_response

        c1 = np.random.randint(2)
        same_cat = np.random.rand() < 0.5
        c2 = c1 if same_cat else 1 - c1
        theta1, theta2 = self._angle(c1), self._angle(c2)

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        t1 = T_context
        inputs[t1:t1 + T_stim1, 1] = np.cos(theta1)
        inputs[t1:t1 + T_stim1, 2] = np.sin(theta1)
        t2 = T_context + T_stim1 + T_delay
        inputs[t2:t2 + T_stim2, 1] = np.cos(theta2)
        inputs[t2:t2 + T_stim2, 2] = np.sin(theta2)
        t_resp = t2 + T_stim2
        inputs[t_resp:, 0] = 0.0

        if self.mode == "match":
            decision = same_cat
        else:
            decision = not same_cat
        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if decision else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim1, T_delay, T_stim2, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts1, Td, Ts2, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts1, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Ts2, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
