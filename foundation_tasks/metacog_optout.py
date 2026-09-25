from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class MetacognitiveOptOut(CognitiveTask):
    """
    Uncertainty-monitoring with an opt-out (Kiani & Shadlen 2009).

    Noisy 1-D evidence is integrated over the stimulus window. When the evidence
    is strong the network should commit to left/right; when it is weak (below a
    confidence band) the correct action is to take the OPT-OUT ("sure") option.
    Choosing the opt-out on hard trials -- rather than guessing -- requires the
    network to represent its own uncertainty, a metacognitive readout no other
    battery task probes.

    Input channels (2): [fixation, evidence].
    Output (argmax over channels 2,3,4): [left, right, opt-out].
    """

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, noise: float = 0.6,
                 conf_band: float = 0.25):
        super().__init__(duration_params)
        self.noise = noise
        self.conf_band = conf_band
        self.is_argmax_task = True
        self.n_classes = 3
        self.loss_channels = list(range(-self.n_classes, 0))  # channels 2,3,4

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        T_total = Tc + Ts + Tr

        mean = float(np.random.uniform(-1, 1))
        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        ev = mean + np.random.normal(0, self.noise, size=Ts)
        inputs[Tc:Tc + Ts, 1] = torch.from_numpy(ev.astype(np.float32))
        t_resp = Tc + Ts
        inputs[t_resp:, 0] = 0.0

        # class: 0=left, 1=right, 2=opt-out (weak evidence).
        if abs(mean) < self.conf_band:
            cls = 2
        else:
            cls = 1 if mean > 0 else 0

        targets = torch.zeros(self.output_dim)
        targets[self.loss_channels[cls]] = 1.0

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
