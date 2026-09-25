from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class ConditionalReasoning(CognitiveTask):
    """
    Conditional-rule evaluation (Wason, 1968) -- deductive reasoning.

    A case presents an antecedent p and a consequent q, each +/-1 (true/false).
    Under the material conditional 'if p then q', the case VIOLATES the rule iff
    p is true and q is false; every other combination is consistent. The network
    reports consistent (1) vs violation (0). Cases are sampled so the two
    outcomes are balanced. Tests evaluation of a logical conditional (detecting
    the falsifying p & not-q case) -- distinct from the cued feature-gating of
    Stroop / TaskSwitch and the nested rules of HierReason.

    Input channels (3): [fixation, antecedent, consequent].
    Output: channel 0 = fixation; channel 3 = consistent/violation (binary).
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

        violate = np.random.rand() < 0.5
        if violate:
            p, q = 1.0, -1.0                       # the unique falsifying case
        else:
            p, q = [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)][np.random.randint(3)]
        consistent = not (p > 0 and q < 0)

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = p
        inputs[T_context:T_context + T_stim, 2] = q
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if consistent else 0.0

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
