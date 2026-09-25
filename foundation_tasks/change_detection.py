from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class ChangeDetection(CognitiveTask):
    """
    Change detection (Luck & Vogel, 1997), sequential-array version.

    A sample array of `array_size` items (directions) is presented one per step,
    held over a delay, then a test array is presented; the network reports
    whether any item changed between sample and test. Requires maintaining the
    whole array in memory and comparing element-wise -- WM load scales with
    array_size.

    Input channels (4): [fixation, item_cos, item_sin, test_phase_flag].
    Output: channel 0 = fixation; channel 3 = change decision (binary).
    """

    PHASE_NAMES = ["context", "sample", "delay", "test", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, array_size: int = 3, n_angles: int = 8):
        super().__init__(duration_params)
        self.array_size = array_size
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_delay = np.random.randint(*dp["delay"])
        T_response = np.random.randint(*dp["response"])
        T_arr = self.array_size * T_item
        T_total = T_context + T_arr + T_delay + T_arr + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        sample = np.random.randint(self.n_angles, size=self.array_size)
        test = sample.copy()
        changed = np.random.rand() < 0.5
        if changed:
            pos = np.random.randint(self.array_size)
            alt = [a for a in range(self.n_angles) if a != sample[pos]]
            test[pos] = np.random.choice(alt)

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        for i, a in enumerate(sample):
            t0 = T_context + i * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
        t_test = T_context + T_arr + T_delay
        for i, a in enumerate(test):
            t0 = t_test + i * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
            inputs[t0:t0 + T_item, 3] = 1.0  # test-phase flag
        t_resp = t_test + T_arr
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if changed else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_arr, T_delay, T_arr, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tt, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tt, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
