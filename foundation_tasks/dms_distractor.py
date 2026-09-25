from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class DMSDistractor(CognitiveTask):
    """
    Delay match-to-sample with a distractor (Miller, Erickson & Desimone 1996;
    NeuroGym DelayMatchSampleDistractor1D).

    A sample direction is shown, then -- crucially -- a *distractor* direction
    appears during the delay and must be ignored, before the test direction is
    presented. Respond match if the test matches the *sample* (not the
    distractor). Unlike plain DMS, this requires a memory robust to
    intervening, behaviourally-irrelevant input (a stronger attractor-memory
    demand).

    Input channels (3): [fixation, stim_cos, stim_sin].
    Output: channel 0 = fixation; channel 3 = match decision (binary).
    """

    PHASE_NAMES = [
        "context",
        "sample",
        "delay1",
        "distractor",
        "delay2",
        "test",
        "response",
    ]
    is_binary_task = True

    def __init__(self, duration_params: Dict, match_threshold=np.pi / 4):
        super().__init__(duration_params)
        self.match_threshold = match_threshold
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_sample = np.random.randint(*dp["stimulus"])
        T_delay = np.random.randint(*dp["delay"])
        T_distract = np.random.randint(*dp["stimulus"])
        T_delay2 = max(1, T_delay // 2)
        T_test = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = (
            T_context + T_sample + T_delay + T_distract + T_delay2 + T_test + T_response
        )

        theta_s = np.random.uniform(0, 2 * np.pi)
        theta_d = np.random.uniform(0, 2 * np.pi)  # distractor (irrelevant)
        is_match = np.random.rand() < 0.5
        if is_match:
            theta_t = theta_s + np.random.uniform(
                -self.match_threshold / 2, self.match_threshold / 2
            )
        else:
            theta_t = theta_s + np.random.uniform(
                self.match_threshold, 2 * np.pi - self.match_threshold
            )
        theta_t %= 2 * np.pi

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        t = T_context
        inputs[t : t + T_sample, 1] = np.cos(theta_s)
        inputs[t : t + T_sample, 2] = np.sin(theta_s)
        t += T_sample + T_delay  # gap, then distractor
        inputs[t : t + T_distract, 1] = np.cos(theta_d)
        inputs[t : t + T_distract, 2] = np.sin(theta_d)
        t += T_distract + T_delay2  # gap, then test
        inputs[t : t + T_test, 1] = np.cos(theta_t)
        inputs[t : t + T_test, 2] = np.sin(theta_t)
        t_resp = t + T_test
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if is_match else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        durations = (
            T_context,
            T_sample,
            T_delay,
            T_distract,
            T_delay2,
            T_test,
            T_response,
        )
        return inputs, targets, mask, durations

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, d = self._build()
        phases = np.concatenate(
            [np.full(d[i], i, dtype=np.int64) for i in range(len(d))]
        )
        return inputs, targets, mask, phases
