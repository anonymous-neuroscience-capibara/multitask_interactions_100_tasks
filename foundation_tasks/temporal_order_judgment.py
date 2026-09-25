from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class TemporalOrderJudgment(CognitiveTask):
    """
    Temporal order judgment (Hirsh & Sherrick 1961).

    Two brief pulses arrive on two separate channels, separated by a small
    stimulus-onset asynchrony (SOA). The network reports which channel's pulse
    came *first*. Small SOAs make this a fine temporal-order discrimination --
    perceptual timing of ORDER rather than DURATION, so it is distinct from the
    interval-timing tasks (IntervalReproduction, TemporalBisection).

    Input channels (3): [fixation, pulse_A, pulse_B].
    Output: channel 3 = "A came first" (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, max_soa: int = 6, pulse_w: int = 2):
        super().__init__(duration_params)
        self.max_soa = max_soa
        self.pulse_w = pulse_w
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        Ts = max(Ts, self.max_soa + self.pulse_w + 2)
        T_total = Tc + Ts + Tr

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0

        a_first = np.random.rand() < 0.5
        soa = np.random.randint(1, self.max_soa + 1)
        t0 = Tc + np.random.randint(0, max(1, Ts - self.max_soa - self.pulse_w - 1))
        tA = t0 if a_first else t0 + soa
        tB = t0 + soa if a_first else t0
        inputs[tA:tA + self.pulse_w, 1] = 1.0
        inputs[tB:tB + self.pulse_w, 2] = 1.0

        t_resp = Tc + Ts
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if a_first else 0.0

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
