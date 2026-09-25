from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class Numerosity(CognitiveTask):
    """
    Numerosity estimation / counting (approximate number system; Nieder &
    Dehaene).

    A number of discrete pulses is presented over the stimulus window; after a
    go cue the network reports *how many* it saw, as a scalar. Distinct from
    PulseDM (which accumulates pulses to make a binary L/R decision) -- here the
    goal is to estimate a *quantity* and emit a continuous count, exercising a
    magnitude (not categorical) readout.

    Input channels (2): [fixation, pulse].
    Output: channel 3 (scalar readout) = count / max_count, in [0,1].
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict, max_count: int = 8):
        super().__init__(duration_params)
        self.max_count = max_count
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        n = int(np.random.randint(1, self.max_count + 1))
        # Place n pulses at distinct random timesteps within the stimulus window.
        pulse_steps = np.random.choice(T_stim, size=min(n, T_stim), replace=False)

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        for k in pulse_steps:
            inputs[T_context + int(k), 1] = 1.0
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = n / self.max_count  # normalised count

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
