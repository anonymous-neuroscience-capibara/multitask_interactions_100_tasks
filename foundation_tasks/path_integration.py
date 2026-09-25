from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PathIntegration1D(CognitiveTask):
    """
    1D path integration / dead reckoning (Cueva & Wei 2018; line attractor,
    Seung 1996).

    A stream of 1D velocities is provided; the network integrates them into a
    running position, then must report the final position after the stream ends
    (held through a response gap with no input). A correct solution maintains a
    line (continuous 1D) attractor that both integrates and stores -- a `**`
    probe for continuous-attractor dynamics.

    Input channels (2): [fixation, velocity].
    Output: channel 3 (scalar readout) = final integrated position.
    """

    PHASE_NAMES = ["context", "integration", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict, path_len_range=(15, 30),
                 vmax: float = 0.15):
        super().__init__(duration_params)
        self.path_len_range = path_len_range
        self.vmax = vmax
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_path = int(np.random.randint(*self.path_len_range))
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_path + T_response

        vel = np.random.uniform(-self.vmax, self.vmax, size=T_path)
        pos = np.cumsum(vel)
        pos = np.clip(pos, -1.0, 1.0)  # keep within the bounded readout range
        final_pos = float(pos[-1])

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_path, 1] = torch.from_numpy(vel.astype(np.float32))
        t_resp = T_context + T_path
        inputs[t_resp:, 0] = 0.0  # response gap: no input, hold the estimate

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = final_pos

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_path, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tp, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tp, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
