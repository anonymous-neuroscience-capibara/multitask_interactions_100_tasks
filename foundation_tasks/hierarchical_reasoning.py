from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class HierarchicalReasoning(CognitiveTask):
    """
    Hierarchical / nested-rule reasoning (Badre & D'Esposito 2009;
    NeuroGym HierarchicalReasoning).

    Two cues at different levels of a hierarchy jointly determine the response:
    a low-level cue picks which stimulus feature to judge (sign vs magnitude),
    and a high-level cue determines whether to apply that judgment normally or
    inverted. The network must combine a rule-of-rules -- distinct from
    TaskSwitch, which only has a single-level cued rule.

    Input channels (4): [fixation, stimulus, high_cue, low_cue].
    Output: channel 0 = fixation; channel 3 = decision (binary).
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
        high = np.random.randint(2)  # 0 = apply rule, 1 = invert it
        low = np.random.randint(2)  # 0 = sign rule, 1 = magnitude rule

        base = (x > 0) if low == 0 else (abs(x) > self.mag_threshold)
        decision = base if high == 0 else (not base)

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        inputs[T_context : T_context + T_stim, 1] = x
        inputs[:, 2] = 1.0 if high else -1.0  # high-level cue (whole trial)
        inputs[:, 3] = 1.0 if low else -1.0  # low-level cue (whole trial)
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
        phases = np.concatenate(
            [
                np.full(Tc, 0, dtype=np.int64),
                np.full(Ts, 1, dtype=np.int64),
                np.full(Tr, 2, dtype=np.int64),
            ]
        )
        return inputs, targets, mask, phases
