from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class AXCPT(CognitiveTask):
    """
    AX-CPT continuous performance / context maintenance (Servan-Schreiber 1996).

    A cue letter is followed by a probe letter. Respond "target" (1) only on the
    AX sequence (cue A then probe X); respond 0 for AY, BX, BY. The cue must be
    held across a delay to disambiguate the probe -- a context-maintenance WM
    demand with three non-target conditions.

    Input channels (3): [fixation, letter_A_vs_B, letter_X_vs_Y].
    Output: channel 0 = fixation; channel 3 = AX-target decision (binary).
    """

    PHASE_NAMES = ["context", "cue", "delay", "probe", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, target_prob: float = 0.5):
        super().__init__(duration_params)
        self.target_prob = target_prob
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_cue = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_probe = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_cue + T_delay + T_probe + T_response

        if np.random.rand() < self.target_prob:
            cue_A, probe_X = True, True  # AX target
        else:
            cue_A = np.random.rand() < 0.5
            probe_X = np.random.rand() < 0.5
            if cue_A and probe_X:  # avoid accidental AX in non-target draw
                probe_X = False
        is_target = cue_A and probe_X

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        t_cue = T_context
        inputs[t_cue:t_cue + T_cue, 1] = 1.0 if cue_A else -1.0
        t_probe = T_context + T_cue + T_delay
        inputs[t_probe:t_probe + T_probe, 2] = 1.0 if probe_X else -1.0
        t_resp = t_probe + T_probe
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if is_target else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_cue, T_delay, T_probe, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tq, Td, Tp, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tq, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tp, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
