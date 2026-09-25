from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class Arithmetics(CognitiveTask):
    """
    Arithmetic task (1D): perform arithmetic on two scalar values based on rule.

    Timeline:
    - Context: fixation + rule
    - Stimulus 1: first scalar value
    - Stimulus 2: second scalar value
    - Response: output the result

    Modes:
    - avg: (a + b)/2
    - multiply: a * b

    Values sampled from [0, 1]. Output on last dimension.

    Input channels (4): [fixation, stimulus, rule_add, rule_mul]

    Tests: compositional computation + rule-switching
    """

    PHASE_NAMES = ["context", "stimulus1", "stimulus2", "response"]

    def __init__(self, duration_params: Dict, mode="add"):
        super().__init__(duration_params)
        self.mode = mode  # 'add' or 'multiply'
        self.is_scalar_task = True
        self.loss_channels = [self.BINARY_CHANNEL]  # Only evaluate last channel

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_stim2 + T_response

        # Sample two scalar values in [0, 1]
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)

        if self.mode == "avg":
            result = (a + b) / 2  # range [0, 1]
            rule_channel = 2
        elif self.mode == "multiply":
            result = a * b  # range [0, 1]
            rule_channel = 3
        else:
            raise ValueError(
                f"Unknown Arithmetics mode: '{self.mode}'. Use 'avg' or 'multiply'."
            )

        # Create inputs: [fixation, stimulus, rule_add, rule_mul]
        inputs = torch.zeros(T_total, 4)
        inputs[:, rule_channel] = 1  # rule throughout trial

        # Context period
        inputs[:T_context, 0] = 1  # fixation

        # First stimulus
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim1
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = a

        # Second stimulus (same channel, sequential)
        t_stim2_start = t_stim_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = b

        # Response period: no fixation
        t_resp_start = t_stim2_end

        # Target: result on last dimension
        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[self.BINARY_CHANNEL] = result

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_stim2 + T_response

        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)

        if self.mode == "avg":
            result = (a + b) / 2
            rule_channel = 2
        elif self.mode == "multiply":
            result = a * b
            rule_channel = 3
        else:
            raise ValueError(
                f"Unknown Arithmetics mode: '{self.mode}'. Use 'avg' or 'multiply'."
            )

        inputs = torch.zeros(T_total, 4)
        inputs[:, rule_channel] = 1

        inputs[:T_context, 0] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim1
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = a

        t_stim2_start = t_stim_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = b

        t_resp_start = t_stim2_end

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[self.BINARY_CHANNEL] = result

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim1, 1, dtype=np.int64),
                np.full(T_stim2, 2, dtype=np.int64),
                np.full(T_response, 3, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
