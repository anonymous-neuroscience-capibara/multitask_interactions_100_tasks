from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DualDelayMatchSample(CognitiveTask):
    """
    Two-item Delayed Match-to-Sample task (simplified to single response).

    Timeline:
    - Context: fixation only
    - Sample: two stimuli shown simultaneously (on separate channels)
    - Delay 1: hold both stimuli in memory
    - Cue: indicates which stimulus will be tested (cue1 or cue2 channel)
    - Delay 2: selectively retrieve cued stimulus
    - Test: single test stimulus shown
    - Response: report match or non-match

    Output: decision on last dimension (1.0 = match, 0.0 = non-match)

    Input channels (7): [fixation, stim1_cos, stim1_sin, stim2_cos, stim2_sin, cue1, cue2]

    Tests: multi-item working memory + selective retrieval + comparison

    Reference: Rose et al. (2016), Reactivation of latent working memories
    with transcranial magnetic stimulation
    """

    def __init__(
        self,
        duration_params: Dict,
        sigma: float = 0.1,
        match_threshold: float = np.pi / 4,
    ):
        super().__init__(duration_params)
        self.sigma = sigma
        self.match_threshold = match_threshold
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_sample = np.random.randint(*self.duration_params["stimulus"])
        T_delay1 = np.random.randint(*self.duration_params["delay"])
        T_cue = np.random.randint(5, 10)
        T_delay2 = np.random.randint(*self.duration_params["delay"])
        T_test = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = (
            T_context + T_sample + T_delay1 + T_cue + T_delay2 + T_test + T_response
        )

        # Sample two stimulus angles
        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)

        # Randomly choose which stimulus to cue
        cued = np.random.choice([1, 2])
        cued_theta = theta1 if cued == 1 else theta2

        # Generate test stimulus: match or non-match with cued stimulus
        is_match = np.random.rand() < 0.5
        if is_match:
            test_theta = cued_theta + np.random.uniform(
                -self.match_threshold / 2, self.match_threshold / 2
            )
        else:
            test_theta = cued_theta + np.random.uniform(
                self.match_threshold, 2 * np.pi - self.match_threshold
            )
        test_theta = test_theta % (2 * np.pi)

        # Create inputs: [fixation, stim1_cos, stim1_sin, stim2_cos, stim2_sin, cue1, cue2]
        inputs = torch.zeros(T_total, 7)

        # Context period
        inputs[:T_context, 0] = 1

        # Sample period: both stimuli shown simultaneously
        t_sample_start = T_context
        t_sample_end = t_sample_start + T_sample
        noise = torch.randn(T_sample, 4) * self.sigma
        inputs[t_sample_start:t_sample_end, 0] = 1
        inputs[t_sample_start:t_sample_end, 1] = np.cos(theta1) + noise[:, 0]
        inputs[t_sample_start:t_sample_end, 2] = np.sin(theta1) + noise[:, 1]
        inputs[t_sample_start:t_sample_end, 3] = np.cos(theta2) + noise[:, 2]
        inputs[t_sample_start:t_sample_end, 4] = np.sin(theta2) + noise[:, 3]

        # Delay 1: fixation only, hold both stimuli
        t_delay1_start = t_sample_end
        t_delay1_end = t_delay1_start + T_delay1
        inputs[t_delay1_start:t_delay1_end, 0] = 1

        # Cue period: indicate which stimulus to retrieve
        t_cue_start = t_delay1_end
        t_cue_end = t_cue_start + T_cue
        inputs[t_cue_start:t_cue_end, 0] = 1
        if cued == 1:
            inputs[t_cue_start:t_cue_end, 5] = 1  # cue1
        else:
            inputs[t_cue_start:t_cue_end, 6] = 1  # cue2

        # Delay 2: selectively retrieve cued stimulus
        t_delay2_start = t_cue_end
        t_delay2_end = t_delay2_start + T_delay2
        inputs[t_delay2_start:t_delay2_end, 0] = 1

        # Test period: single test stimulus on stim1 channels
        t_test_start = t_delay2_end
        t_test_end = t_test_start + T_test
        noise_test = torch.randn(T_test, 2) * self.sigma
        inputs[t_test_start:t_test_end, 0] = 1
        inputs[t_test_start:t_test_end, 1] = np.cos(test_theta) + noise_test[:, 0]
        inputs[t_test_start:t_test_end, 2] = np.sin(test_theta) + noise_test[:, 1]

        # Response period: no fixation
        t_resp_start = t_test_end

        # Target: decision on last dimension
        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if is_match else 0.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
