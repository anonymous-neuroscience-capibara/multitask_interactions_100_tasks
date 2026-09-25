from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class StopSignal(CognitiveTask):
    """
    Stop-signal / countermanding (Logan & Cowan 1984; Hanes & Schall 1996).

    A go stimulus appears; on a fraction of trials a stop signal follows shortly
    after (variable stop-signal delay), and the prepared response must be
    *withheld*. Tests response inhibition: respond (1) on go trials, withhold (0)
    on stop trials. The shorter the stop-signal delay, the easier to cancel.

    Input channels (3): [fixation, go_stim, stop_signal].
    Output: channel 0 = fixation; channel 3 = respond/withhold (binary).
    """

    PHASE_NAMES = ["context", "go", "decision"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, stop_prob: float = 0.35):
        super().__init__(duration_params)
        self.stop_prob = stop_prob
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_go = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_go + T_response

        is_stop = np.random.rand() < self.stop_prob

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_go, 1] = 1.0  # go stimulus
        if is_stop:
            ssd = np.random.randint(1, max(2, T_go))  # stop-signal delay
            t_stop = T_context + ssd
            inputs[t_stop:t_stop + 2, 2] = 1.0  # stop signal
        t_resp = T_context + T_go
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 0.0 if is_stop else 1.0  # withhold vs respond

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_go, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tg, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tg, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
