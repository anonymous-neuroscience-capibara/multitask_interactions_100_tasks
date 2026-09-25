from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class BackwardMasking(CognitiveTask):
    """
    Backward / metacontrast masking (Breitmeyer & Ganz, 1976) -- visual
    perception under temporal masking.

    A brief target (a category, +/-1) is shown, then immediately followed by a
    high-noise mask on the same channel; the network reports the target's
    category. Because the mask overwrites the target shortly after onset, the
    answer must be captured quickly -- a probe of perception when a stimulus is
    masked by a following one.

    Input channels (2): [fixation, signal].
    Output: channel 0 = fixation; channel 3 = target category (binary).
    """

    PHASE_NAMES = ["context", "target", "mask", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, target_len: int = 2,
                 mask_noise: float = 0.8):
        super().__init__(duration_params)
        self.target_len = target_len
        self.mask_noise = mask_noise
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_mask = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + self.target_len + T_mask + T_response

        val = float(np.random.choice([-1.0, 1.0]))

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        t_t = T_context
        inputs[t_t:t_t + self.target_len, 1] = val           # brief target
        t_m = t_t + self.target_len
        inputs[t_m:t_m + T_mask, 1] = torch.from_numpy(       # masking noise
            np.random.normal(0, self.mask_noise, size=T_mask).astype(np.float32))
        t_resp = t_m + T_mask
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if val > 0 else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, self.target_len, T_mask, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tt, Tm, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tt, 1, dtype=np.int64),
            np.full(Tm, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
