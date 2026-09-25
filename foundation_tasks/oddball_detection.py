from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class OddballDetection(CognitiveTask):
    """
    Oddball / deviance detection (Naatanen mismatch negativity; Ulanovsky,
    Las & Nelken 2003 stimulus-specific adaptation).

    A stream of identical 'standard' stimuli (a repeated direction) is
    presented; on half of trials exactly one item is a 'deviant' (a different
    direction). The network reports whether a deviant occurred. Tests detection
    of a break in a learned regularity (predictive-coding / adaptation) -- with
    no cue and no comparison array, distinct from cued visual search and from
    array change-detection.

    Input channels (3): [fixation, stim_cos, stim_sin].
    Output: channel 0 = fixation; channel 3 = deviant-present decision (binary).
    """

    PHASE_NAMES = ["context", "stream", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, n_items: int = 5, n_angles: int = 8):
        super().__init__(duration_params)
        self.n_items = n_items
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_response = np.random.randint(*dp["response"])
        T_stream = self.n_items * T_item
        T_total = T_context + T_stream + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        standard = np.random.randint(self.n_angles)
        seq = [standard] * self.n_items
        present = np.random.rand() < 0.5
        if present:
            # Deviant not at position 0, so a standard is established first.
            pos = np.random.randint(1, self.n_items)
            seq[pos] = np.random.choice(
                [a for a in range(self.n_angles) if a != standard]
            )

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        for i, a in enumerate(seq):
            t0 = T_context + i * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
        t_resp = T_context + T_stream
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if present else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stream, T_response)

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
