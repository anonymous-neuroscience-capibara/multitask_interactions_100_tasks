from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class MentalRotation(CognitiveTask):
    """
    Mental rotation (Shepard & Metzler, 1971) -- mental imagery / spatial
    transformation.

    A sample orientation is shown together with a cued rotation amount; after a
    delay a probe orientation appears, and the network reports whether the probe
    is the sample rotated by the cued amount (match = 1) or not (0). In humans
    reaction time scales with the rotation angle; here it is a supervised
    same/different judgment requiring an internal rotation-and-compare.

    Input channels (4): [fixation, dir_cos, dir_sin, rotation_amount].
    Output: channel 0 = fixation; channel 3 = match decision (binary).
    """

    PHASE_NAMES = ["context", "sample", "delay", "probe", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, n_angles: int = 8):
        super().__init__(duration_params)
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_sample = 2
        T_delay = np.random.randint(*dp["delay"])
        T_probe = 2
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_sample + T_delay + T_probe + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        theta = np.random.randint(self.n_angles)
        rot = np.random.randint(self.n_angles)            # rotation in steps
        rotated = (theta + rot) % self.n_angles
        match = np.random.rand() < 0.5
        psi = rotated if match else np.random.choice(
            [a for a in range(self.n_angles) if a != rotated])

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        inputs[:, 3] = rot / self.n_angles                # rotation cue, held throughout
        t_s = T_context
        inputs[t_s:t_s + T_sample, 1] = np.cos(angles[theta])
        inputs[t_s:t_s + T_sample, 2] = np.sin(angles[theta])
        t_p = T_context + T_sample + T_delay
        inputs[t_p:t_p + T_probe, 1] = np.cos(angles[psi])
        inputs[t_p:t_p + T_probe, 2] = np.sin(angles[psi])
        t_resp = t_p + T_probe
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if match else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_sample, T_delay, T_probe, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tp, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tp, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
