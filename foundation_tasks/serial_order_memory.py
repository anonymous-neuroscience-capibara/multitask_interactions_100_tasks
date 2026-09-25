from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class SerialOrderMemory(CognitiveTask):
    """
    Memory for serial order (Marshuetz, 2005; episodic order memory).

    A list of distinct items (directions) is presented one at a time; then two
    of them are re-presented as probes, and the network reports whether the
    FIRST probe occurred BEFORE the second in the list. This taxes memory for
    temporal order, distinct from item recognition (Sternberg: was it in the
    set) and from perceptual temporal-order judgment (which event came first).

    Input channels (4): [fixation, item_cos, item_sin, probe_flag].
    Output: channel 0 = fixation; channel 3 = probe1-before-probe2 (binary).
    """

    PHASE_NAMES = ["context", "list", "probe1", "probe2", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, list_len: int = 4, n_angles: int = 8):
        super().__init__(duration_params)
        self.list_len = list_len
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_probe = 2
        T_response = np.random.randint(*dp["response"])
        T_list = self.list_len * T_item
        T_total = T_context + T_list + 2 * T_probe + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        items = np.random.choice(self.n_angles, size=self.list_len, replace=False)
        i, j = np.sort(np.random.choice(self.list_len, size=2, replace=False))
        probe1_first = np.random.rand() < 0.5
        if probe1_first:           # probe1 = earlier item -> decision 1
            p1, p2 = items[i], items[j]
        else:                      # probe1 = later item   -> decision 0
            p1, p2 = items[j], items[i]

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        for k, a in enumerate(items):
            t0 = T_context + k * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
        t_p1 = T_context + T_list
        inputs[t_p1:t_p1 + T_probe, 1] = np.cos(angles[p1])
        inputs[t_p1:t_p1 + T_probe, 2] = np.sin(angles[p1])
        inputs[t_p1:t_p1 + T_probe, 3] = 1.0
        t_p2 = t_p1 + T_probe
        inputs[t_p2:t_p2 + T_probe, 1] = np.cos(angles[p2])
        inputs[t_p2:t_p2 + T_probe, 2] = np.sin(angles[p2])
        inputs[t_p2:t_p2 + T_probe, 3] = 1.0
        t_resp = t_p2 + T_probe
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if probe1_first else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_list, T_probe, T_probe, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tl, Tp1, Tp2, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tl, 1, dtype=np.int64),
            np.full(Tp1, 2, dtype=np.int64), np.full(Tp2, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
