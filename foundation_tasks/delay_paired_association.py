from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class DelayPairedAssociation(CognitiveTask):
    """
    Delayed paired-association (Sakai & Miyashita, 1991).

    A cue A is shown, held over a delay, then a cue B is shown; the network must
    report whether B is the learned associate of A. Here the association is a
    fixed deterministic rule (associate angle = A + pi, i.e. the diametric
    partner), so the task is supervised: respond match if B is A's partner.
    Requires holding A across the delay and comparing to B.

    Input channels (4): [fixation, cue_cos, cue_sin, cueB_flag].
    Output: channel 0 = fixation; channel 3 = associate-match decision (binary).
    """

    PHASE_NAMES = ["context", "cueA", "delay", "cueB", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, n_angles: int = 8):
        super().__init__(duration_params)
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _partner(self, a):
        # Fixed associative rule: diametric partner on the ring.
        return (a + self.n_angles // 2) % self.n_angles

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_cue = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_cueB = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_cue + T_delay + T_cueB + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        a = np.random.randint(self.n_angles)
        is_match = np.random.rand() < 0.5
        if is_match:
            b = self._partner(a)
        else:
            # Any non-partner (and not identical) angle.
            choices = [x for x in range(self.n_angles)
                       if x != self._partner(a) and x != a]
            b = np.random.choice(choices)

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        t_a = T_context
        inputs[t_a:t_a + T_cue, 1] = np.cos(angles[a])
        inputs[t_a:t_a + T_cue, 2] = np.sin(angles[a])
        t_b = T_context + T_cue + T_delay
        inputs[t_b:t_b + T_cueB, 1] = np.cos(angles[b])
        inputs[t_b:t_b + T_cueB, 2] = np.sin(angles[b])
        inputs[t_b:t_b + T_cueB, 3] = 1.0  # cue-B flag
        t_resp = t_b + T_cueB
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if is_match else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_cue, T_delay, T_cueB, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ta, Td, Tb, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ta, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tb, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
