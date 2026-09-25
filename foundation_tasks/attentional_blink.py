from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class AttentionalBlink(CognitiveTask):
    """
    Attentional blink / RSVP dual-target (Raymond, Shapiro & Arnell, 1992) --
    temporal attention.

    A rapid stream of items (directions) is presented; two of them are flagged
    as targets, T1 and T2, separated by a variable lag. The network reports the
    *category* of the SECOND target (T2) -- here, whether T2 points to the upper
    half of the ring. Reporting a second target shortly after the first taxes
    temporal attention (in humans, short T1-T2 lags impair T2 report -- the
    'blink'). Distinct from spatial search (VisualSearch) and spatial cueing
    (PosnerCueing): the selection is over *time* within a stream.

    Input channels (4): [fixation, item_cos, item_sin, target_flag].
    Output: channel 0 = fixation; channel 3 = T2 category (binary).
    """

    PHASE_NAMES = ["context", "stream", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, stream_len: int = 8, n_angles: int = 8):
        super().__init__(duration_params)
        self.stream_len = stream_len
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_response = np.random.randint(*dp["response"])
        T_stream = self.stream_len * T_item
        T_total = T_context + T_stream + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        seq = np.random.randint(self.n_angles, size=self.stream_len)
        t1_pos, t2_pos = np.sort(
            np.random.choice(self.stream_len, size=2, replace=False))
        # T2 category: does it point to the upper half of the ring (sin > 0)?
        decision = 1.0 if np.sin(angles[seq[t2_pos]]) > 0 else 0.0

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        for k, a in enumerate(seq):
            t0 = T_context + k * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
            if k in (t1_pos, t2_pos):
                inputs[t0:t0 + T_item, 3] = 1.0  # target flag (T1 and T2)
        t_resp = T_context + T_stream
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = decision

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
