from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class OddOneOut(CognitiveTask):
    """
    Odd-one-out: deviance LOCALIZATION among K simultaneous items.

    K scalar items are shown at once, one per channel; K-1 share a common level
    and exactly ONE differs (its level is offset up OR down by a fixed margin,
    chosen at random). The network reports WHICH item is the deviant. Because
    the outlier can be either higher or lower than the crowd, the answer is not
    the max -- so this is distinct from MaxOfK (report the largest) and from
    OddballDetection (binary present/absent): the net must compare every item
    against the group consensus and localize the one that breaks it.

    Input channels (1 + K): [fixation, item_0, ..., item_{K-1}].
    Output (argmax over the last K channels): index of the deviant item.
    Accuracy: argmax over loss_channels at the last masked step (chance = 1/K).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, n_classes: int = 4,
                 sigma: float = 0.05, delta: float = 0.35):
        super().__init__(duration_params)
        self.is_argmax_task = True
        self.n_classes = int(n_classes)
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.loss_channels = list(range(-self.n_classes, 0))

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        K = self.n_classes
        T_total = Tc + Ts + Tr

        base = float(np.random.uniform(0.35, 0.65))
        odd = int(np.random.randint(K))
        direction = 1.0 if np.random.rand() < 0.5 else -1.0  # deviant is high OR low
        odd_val = float(np.clip(base + direction * self.delta, 0.02, 0.98))
        vals = np.full(K, base, dtype=np.float32)
        vals[odd] = odd_val

        inputs = torch.zeros(T_total, 1 + K)
        inputs[:Tc + Ts, 0] = 1.0
        stim = vals[None, :] + np.random.normal(0, self.sigma, size=(Ts, K))
        inputs[Tc:Tc + Ts, 1:1 + K] = torch.from_numpy(stim.astype(np.float32))
        t_resp = Tc + Ts

        targets = torch.zeros(self.output_dim)
        targets[self.loss_channels[odd]] = 1.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, Ts, Tr)

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
