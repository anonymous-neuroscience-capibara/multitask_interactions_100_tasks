from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class RelationalMatch(CognitiveTask):
    """
    Relational match-to-sample / analogy (Premack 1983; Gentner 1983).

    Two pairs of items are shown in sequence. Each pair has a *relation*: SAME
    (the two items match) or DIFFERENT. The network reports whether the two
    pairs share the same relation ("AA is to AA as BB is to CC?"). This is
    reasoning about relations-between-relations -- the item identities are
    irrelevant, only same/different structure matters. Distinct from
    ConditionalReasoning (if-then) and HierarchicalReasoning (nested rules).

    Input channels (2): [fixation, item].
    Output: channel 3 = "relations match" (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, min_diff: float = 0.5):
        super().__init__(duration_params)
        self.min_diff = min_diff
        self.loss_channels = [self.BINARY_CHANNEL]

    def _pair(self, same):
        a = np.random.uniform(-1, 1)
        if same:
            return a, a
        b = a + np.random.choice([-1, 1]) * np.random.uniform(self.min_diff, 1.0)
        b = float(np.clip(b, -1, 1))
        if abs(b - a) < self.min_diff:
            b = -a if abs(a) > self.min_diff / 2 else a + self.min_diff
        return a, b

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        seg = max(2, np.random.randint(*dp["stimulus"]) // 4)
        Tr = np.random.randint(*dp["response"])
        T_stim = 4 * seg
        T_total = Tc + T_stim + Tr

        rel1_same = np.random.rand() < 0.5
        rel2_same = np.random.rand() < 0.5
        a1, b1 = self._pair(rel1_same)
        a2, b2 = self._pair(rel2_same)
        items = [a1, b1, a2, b2]

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        t = Tc
        for it in items:
            inputs[t:t + seg, 1] = it
            t += seg
        t_resp = Tc + T_stim
        inputs[t_resp:, 0] = 0.0

        match = rel1_same == rel2_same
        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if match else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, T_stim, Tr)

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
