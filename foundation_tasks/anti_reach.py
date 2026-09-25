from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class AntiReach1D(CognitiveTask):
    """
    Anti-reach on a 1D line (Georgopoulos-style reaching; anti mapping).

    A target appears at scalar position x in [-1, 1]; the network must reach to
    the opposite position -x. Uses a scalar output (not a ring), exercising the
    scalar-readout path and a sign-flip sensorimotor transform.

    Input channels (2): [fixation, target_position].
    Output: channel 3 (scalar readout) = -x.
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        x = np.random.uniform(-1.0, 1.0)

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = x
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = -x  # anti reach

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_response)

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
