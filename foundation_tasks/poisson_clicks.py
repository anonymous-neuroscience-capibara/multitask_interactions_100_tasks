from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PoissonClicks(CognitiveTask):
    """
    Pulse-based evidence accumulation (Brunton, Brody & Hanks, 2013, rats).

    Two streams of Poisson 'clicks' arrive on a left and a right channel during
    the stimulus window; after the streams end the network reports which side
    delivered MORE clicks. The canonical accumulation-to-bound (drift-diffusion)
    task -- distinct from PerceptualDM (a coherence/direction decision): here the
    evidence is discrete, stochastic pulses that must be counted and compared.

    Input channels (3): [fixation, left_click, right_click].
    Output: channel 0 = fixation; channel 3 = right>left decision (binary).
    """

    PHASE_NAMES = ["context", "clicks", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, rate_range=(0.15, 0.55)):
        super().__init__(duration_params)
        self.rate_range = rate_range
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        # Independent Poisson-like click trains; resample until counts differ.
        for _ in range(50):
            p_left = np.random.uniform(*self.rate_range)
            p_right = np.random.uniform(*self.rate_range)
            left = (np.random.rand(T_stim) < p_left).astype(np.float32)
            right = (np.random.rand(T_stim) < p_right).astype(np.float32)
            if left.sum() != right.sum():
                break
        decision = 1.0 if right.sum() > left.sum() else 0.0

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = torch.from_numpy(left)
        inputs[T_context:T_context + T_stim, 2] = torch.from_numpy(right)
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = decision

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
