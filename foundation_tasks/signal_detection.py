from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class SignalDetection(CognitiveTask):
    """
    Yes/no signal detection in noise (Green & Swets, 1966) -- psychophysics.

    On half of trials a weak constant signal is added to a noisy stimulus
    stream; on the other half there is only noise. The network reports whether
    the signal is present. Detecting a small mean shift against noise requires
    integrating the evidence over the window -- the classic detection-theory
    paradigm (sensitivity / criterion), distinct from search (VisualSearch) and
    direction decisions (PerceptualDM).

    Input channels (2): [fixation, stimulus].
    Output: channel 0 = fixation; channel 3 = signal-present decision (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, signal_amp: float = 0.4,
                 noise: float = 0.4):
        super().__init__(duration_params)
        self.signal_amp = signal_amp
        self.noise = noise
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        present = np.random.rand() < 0.5
        level = self.signal_amp if present else 0.0

        inputs = torch.zeros(T_total, 2)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_stim, 1] = torch.from_numpy(
            (level + np.random.normal(0, self.noise, size=T_stim)).astype(np.float32))
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if present else 0.0

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
