from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class TaskSwitch(CognitiveTask):
    """
    Cued task switching (Sohn et al. 2000; Sakai 2008).

    A rule cue selects which of two judgments to apply to a scalar stimulus:
    rule 0 -> report its sign; rule 1 -> report whether its magnitude exceeds a
    threshold. The same stimulus maps to different responses depending on the
    cued rule, requiring flexible rule-based gating.

    Input channels (3): [fixation, stimulus_value, rule].  rule>0 -> magnitude judgment.
    Output: channel 0 = fixation; channel 3 = rule-dependent decision (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, mag_threshold: float = 0.5):
        super().__init__(duration_params)
        self.mag_threshold = mag_threshold
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        x = np.random.uniform(-1.0, 1.0)
        magnitude_rule = np.random.rand() < 0.5
        if magnitude_rule:
            decision = abs(x) > self.mag_threshold
        else:
            decision = x > 0

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = x
        inputs[:, 2] = 1.0 if magnitude_rule else -1.0  # rule cue throughout
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if decision else 0.0

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
