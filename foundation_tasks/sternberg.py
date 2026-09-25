from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class Sternberg(CognitiveTask):
    """
    Sternberg item-recognition (Sternberg, 1966).

    A memory set of `set_size` items (directions on a ring) is presented one at
    a time, held over a delay, then a probe item appears: is the probe in the
    set? Working-memory load scales with set_size, giving a graded difficulty
    axis. Requires holding a *set* of items, not a single value.

    Input channels (4): [fixation, item_cos, item_sin, probe_flag].
    Output: channel 0 = fixation; channel 3 = in-set decision (binary).
    """

    PHASE_NAMES = ["context", "encoding", "delay", "probe", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, set_size: int = 3, n_angles: int = 8):
        super().__init__(duration_params)
        self.set_size = set_size
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_delay = np.random.randint(*dp["delay"])
        T_probe = 2
        T_response = np.random.randint(*dp["response"])
        T_enc = self.set_size * T_item
        T_total = T_context + T_enc + T_delay + T_probe + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        mem_set = np.random.choice(self.n_angles, size=self.set_size, replace=False)
        in_set = np.random.rand() < 0.5
        if in_set:
            probe = np.random.choice(mem_set)
        else:
            outside = [a for a in range(self.n_angles) if a not in mem_set]
            probe = np.random.choice(outside)

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0  # fixation on until response
        # Encoding: present each set item.
        for i, a in enumerate(mem_set):
            t0 = T_context + i * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
        # Probe.
        t_probe = T_context + T_enc + T_delay
        inputs[t_probe:t_probe + T_probe, 1] = np.cos(angles[probe])
        inputs[t_probe:t_probe + T_probe, 2] = np.sin(angles[probe])
        inputs[t_probe:t_probe + T_probe, 3] = 1.0  # probe flag
        # Response: fixation off.
        t_resp = t_probe + T_probe
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0
        targets[self.BINARY_CHANNEL] = 1.0 if in_set else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_enc, T_delay, T_probe, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Te, Td, Tp, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Te, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tp, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
