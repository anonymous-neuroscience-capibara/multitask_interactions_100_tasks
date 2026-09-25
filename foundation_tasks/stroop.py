from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class Stroop(CognitiveTask):
    """
    Stroop conflict task (Stroop, 1935).

    Two features are presented simultaneously -- a "color" and a "word" -- each
    +/-1 (two categories). A rule cue selects which feature is relevant; the
    network reports that feature's category, ignoring the other. On incongruent
    trials the features disagree, requiring conflict resolution.

    Input channels (4): [fixation, color, word, rule].  rule>0 -> report color.
    Output: channel 0 = fixation; channel 3 = reported category (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        color = np.random.choice([-1.0, 1.0])
        word = np.random.choice([-1.0, 1.0])
        report_color = np.random.rand() < 0.5
        relevant = color if report_color else word

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = color
        inputs[T_context:T_context + T_stim, 2] = word
        inputs[:, 3] = 1.0 if report_color else -1.0  # rule cue throughout
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if relevant > 0 else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_response)

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
