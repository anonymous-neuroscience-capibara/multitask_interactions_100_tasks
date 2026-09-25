from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class IntervalDiscrimination(CognitiveTask):
    """
    Interval Discrimination task: compare the duration of two stimuli.

    Timeline:
    - Context: fixation only
    - Stimulus 1: ON for a variable duration
    - Delay: hold duration estimate in memory
    - Stimulus 2: ON for a different variable duration
    - Response: which stimulus was longer?

    Each stimulus is a constant signal (1.0) on a dedicated channel.
    The network must estimate how long each was ON, hold the first
    estimate across the delay, then compare.

    Output: decision on last dimension (1.0 = stim1 longer, 0.0 = stim2 longer)
            fixation on dim 0

    Input channels (4): [fixation, stim1, stim2, rule]

    Tests: temporal estimation + working memory for duration + comparison

    Reference: Genovesio et al. (2009), Feature- and Order-Based Timing
    Representations in the Frontal Cortex
    """

    PHASE_NAMES = ["context", "stim1", "delay", "stim2", "response"]

    def __init__(
        self,
        duration_params: Dict,
        min_stim: int = 5,
        max_stim: int = 20,
    ):
        super().__init__(duration_params)
        self.min_stim = min_stim
        self.max_stim = max_stim
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(self.min_stim, self.max_stim + 1)
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(self.min_stim, self.max_stim + 1)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay + T_stim2 + T_response

        # Which stimulus was longer?
        stim1_longer = T_stim1 > T_stim2

        # Create inputs: [fixation, stim1, stim2, rule]
        inputs = torch.zeros(T_total, 4)

        # Context period
        inputs[:T_context, 0] = 1

        # Stimulus 1: constant signal on stim1 channel
        t_s1_start = T_context
        t_s1_end = t_s1_start + T_stim1
        inputs[t_s1_start:t_s1_end, 0] = 1
        inputs[t_s1_start:t_s1_end, 1] = 1

        # Delay: fixation only
        t_delay_start = t_s1_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1

        # Stimulus 2: constant signal on stim2 channel
        t_s2_start = t_delay_end
        t_s2_end = t_s2_start + T_stim2
        inputs[t_s2_start:t_s2_end, 0] = 1
        inputs[t_s2_start:t_s2_end, 2] = 1

        # Response period: no fixation
        t_resp_start = t_s2_end
        inputs[t_resp_start:, 3] = 1

        # Target: fixation on dim 0, decision on last dim
        targets = torch.zeros(self.output_dim)
        if stim1_longer:
            targets[0] = 0.0
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[0] = 0.0
            targets[self.BINARY_CHANNEL] = 0.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(self.min_stim, self.max_stim + 1)
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(self.min_stim, self.max_stim + 1)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay + T_stim2 + T_response

        stim1_longer = T_stim1 > T_stim2

        inputs = torch.zeros(T_total, 4)

        inputs[:T_context, 0] = 1

        t_s1_start = T_context
        t_s1_end = t_s1_start + T_stim1
        inputs[t_s1_start:t_s1_end, 0] = 1
        inputs[t_s1_start:t_s1_end, 1] = 1

        t_delay_start = t_s1_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1

        t_s2_start = t_delay_end
        t_s2_end = t_s2_start + T_stim2
        inputs[t_s2_start:t_s2_end, 0] = 1
        inputs[t_s2_start:t_s2_end, 2] = 1

        t_resp_start = t_s2_end
        inputs[t_resp_start:, 3] = 1

        targets = torch.zeros(self.output_dim)
        if stim1_longer:
            targets[0] = 0.0
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[0] = 0.0
            targets[self.BINARY_CHANNEL] = 0.0

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
