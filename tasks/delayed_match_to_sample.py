from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DelayedMatchToSample(CognitiveTask):
    """
    Delayed Match-to-Sample task: remember first stimulus, compare to second stimulus
    Match: respond if both stimuli are in same direction (within tolerance)
    NonMatch: respond if stimuli are in different directions
    """

    PHASE_NAMES = ["context", "stim1", "delay", "stim2", "response"]

    def __init__(self, duration_params: Dict, mode="match"):
        super().__init__(duration_params)
        self.mode = mode  # 'match' or 'nonmatch'
        self.match_threshold = np.pi / 4  # 45 degrees tolerance
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_delay1 = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay1 + T_stim2 + T_response

        # Sample two stimulus angles
        theta1 = np.random.uniform(0, 2 * np.pi)
        # Second stimulus: sometimes match, sometimes don't
        if np.random.rand() < 0.5:
            # Match trial
            theta2 = theta1 + np.random.uniform(
                -self.match_threshold / 2, self.match_threshold / 2
            )
            is_match = True
        else:
            # Non-match trial (ensure they're different)
            theta2 = theta1 + np.random.uniform(
                self.match_threshold, 2 * np.pi - self.match_threshold
            )
            is_match = False

        # Wrap angles
        theta2 = theta2 % (2 * np.pi)

        # Create inputs: [fixation, stim_cos, stim_sin, rule_match, rule_nonmatch]
        inputs = torch.zeros(T_total, 5)

        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == "match":
            inputs[:T_context, 3] = 1  # rule_match
        else:
            inputs[:T_context, 4] = 1  # rule_nonmatch

        # First stimulus
        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1
        inputs[t_stim1_start:t_stim1_end, 1] = np.cos(theta1)
        inputs[t_stim1_start:t_stim1_end, 2] = np.sin(theta1)
        if self.mode == "match":
            inputs[t_stim1_start:t_stim1_end, 3] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 4] = 1

        # First delay
        t_delay1_start = t_stim1_end
        t_delay1_end = t_delay1_start + T_delay1
        inputs[t_delay1_start:t_delay1_end, 0] = 1
        if self.mode == "match":
            inputs[t_delay1_start:t_delay1_end, 3] = 1
        else:
            inputs[t_delay1_start:t_delay1_end, 4] = 1

        # Second stimulus
        t_stim2_start = t_delay1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = np.cos(theta2)
        inputs[t_stim2_start:t_stim2_end, 2] = np.sin(theta2)
        if self.mode == "match":
            inputs[t_stim2_start:t_stim2_end, 3] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 4] = 1

        # Response period: no fixation
        t_resp_start = t_stim2_end
        if self.mode == "match":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        # Target output
        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0  # always break fixation during response

        if self.mode == "match":
            targets[self.BINARY_CHANNEL] = 1.0 if is_match else 0.0
        else:
            targets[self.BINARY_CHANNEL] = 0.0 if is_match else 1.0

        # Mask: evaluate at all response timesteps
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_delay1 = np.random.randint(*self.duration_params["delay"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_delay1 + T_stim2 + T_response

        theta1 = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.5:
            theta2 = theta1 + np.random.uniform(
                -self.match_threshold / 2, self.match_threshold / 2
            )
            is_match = True
        else:
            theta2 = theta1 + np.random.uniform(
                self.match_threshold, 2 * np.pi - self.match_threshold
            )
            is_match = False

        theta2 = theta2 % (2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        inputs[:T_context, 0] = 1
        if self.mode == "match":
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1

        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1
        inputs[t_stim1_start:t_stim1_end, 1] = np.cos(theta1)
        inputs[t_stim1_start:t_stim1_end, 2] = np.sin(theta1)
        if self.mode == "match":
            inputs[t_stim1_start:t_stim1_end, 3] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 4] = 1

        t_delay1_start = t_stim1_end
        t_delay1_end = t_delay1_start + T_delay1
        inputs[t_delay1_start:t_delay1_end, 0] = 1
        if self.mode == "match":
            inputs[t_delay1_start:t_delay1_end, 3] = 1
        else:
            inputs[t_delay1_start:t_delay1_end, 4] = 1

        t_stim2_start = t_delay1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = np.cos(theta2)
        inputs[t_stim2_start:t_stim2_end, 2] = np.sin(theta2)
        if self.mode == "match":
            inputs[t_stim2_start:t_stim2_end, 3] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 4] = 1

        t_resp_start = t_stim2_end
        if self.mode == "match":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0

        if self.mode == "match":
            targets[self.BINARY_CHANNEL] = 1.0 if is_match else 0.0
        else:
            targets[self.BINARY_CHANNEL] = 0.0 if is_match else 1.0

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim1, 1, dtype=np.int64),
                np.full(T_delay1, 2, dtype=np.int64),
                np.full(T_stim2, 3, dtype=np.int64),
                np.full(T_response, 4, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
