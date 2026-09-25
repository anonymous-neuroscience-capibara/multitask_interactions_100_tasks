from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DelayComparison(CognitiveTask):
    """
    Delayed Comparison task: compare magnitudes of two stimuli separated by delay.

    Timeline:
    - Context: fixation only
    - Stimulus 1: noisy scalar magnitude presented
    - Delay: must hold estimated magnitude in memory
    - Stimulus 2: second noisy scalar magnitude presented
    - Response: report which stimulus was stronger

    Input channels (2): [fixation, stimulus_magnitude]

    The stimulus channel carries a normalized scalar ∈ [0.5, 1.0] + noise.
    Value pairs are chosen so difficulty varies (closer pairs = harder).

    Tests: temporal averaging + working memory for scalar magnitude + comparison

    Reference: Romo et al. (1999), Neuronal Population Coding of Parametric Working Memory
    """

    PHASE_NAMES = ["context", "stim1", "delay", "stim2", "response"]

    def __init__(
        self,
        duration_params: Dict,
        sigma: float = 0.15,
        min_diff: float = 0.1,
        max_diff: float = 0.4,
    ):
        super().__init__(duration_params)
        self.sigma = sigma
        self.min_diff = min_diff  # Hardest trials
        self.max_diff = max_diff  # Easiest trials
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay + T_stim2 + T_response

        # Sample magnitudes continuously
        gap = np.random.uniform(self.min_diff, self.max_diff)
        base = np.random.uniform(0.5, 1.0 - gap)
        mag_strong = base + gap
        mag_weak = base

        # Randomly assign which comes first
        if np.random.rand() < 0.5:
            mag1, mag2 = mag_strong, mag_weak
            stim1_stronger = True
        else:
            mag1, mag2 = mag_weak, mag_strong
            stim1_stronger = False

        # Create inputs: [fixation, stimulus_magnitude]
        inputs = torch.zeros(T_total, 2)

        # Context period
        inputs[:T_context, 0] = 1  # fixation

        # Stimulus 1: fixation + noisy magnitude
        t_s1_start = T_context
        t_s1_end = t_s1_start + T_stim1
        noise1 = torch.randn(T_stim1) * self.sigma
        inputs[t_s1_start:t_s1_end, 0] = 1  # fixation
        inputs[t_s1_start:t_s1_end, 1] = mag1 + noise1  # noisy magnitude

        # Delay: fixation only, no stimulus
        t_delay_start = t_s1_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1  # fixation

        # Stimulus 2: fixation + noisy magnitude
        t_s2_start = t_delay_end
        t_s2_end = t_s2_start + T_stim2
        noise2 = torch.randn(T_stim2) * self.sigma
        inputs[t_s2_start:t_s2_end, 0] = 1  # fixation
        inputs[t_s2_start:t_s2_end, 1] = mag2 + noise2  # noisy magnitude

        # Response period: no fixation
        t_resp_start = t_s2_end

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if stim1_stronger else 0.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay + T_stim2 + T_response

        gap = np.random.uniform(self.min_diff, self.max_diff)
        base = np.random.uniform(0.5, 1.0 - gap)
        mag_strong = base + gap
        mag_weak = base

        if np.random.rand() < 0.5:
            mag1, mag2 = mag_strong, mag_weak
            stim1_stronger = True
        else:
            mag1, mag2 = mag_weak, mag_strong
            stim1_stronger = False

        inputs = torch.zeros(T_total, 2)

        inputs[:T_context, 0] = 1

        t_s1_start = T_context
        t_s1_end = t_s1_start + T_stim1
        noise1 = torch.randn(T_stim1) * self.sigma
        inputs[t_s1_start:t_s1_end, 0] = 1
        inputs[t_s1_start:t_s1_end, 1] = mag1 + noise1

        t_delay_start = t_s1_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1

        t_s2_start = t_delay_end
        t_s2_end = t_s2_start + T_stim2
        noise2 = torch.randn(T_stim2) * self.sigma
        inputs[t_s2_start:t_s2_end, 0] = 1
        inputs[t_s2_start:t_s2_end, 1] = mag2 + noise2

        t_resp_start = t_s2_end

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if stim1_stronger else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim1, 1, dtype=np.int64),
                np.full(T_delay, 2, dtype=np.int64),
                np.full(T_stim2, 3, dtype=np.int64),
                np.full(T_response, 4, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
