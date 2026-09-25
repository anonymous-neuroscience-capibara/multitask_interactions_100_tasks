from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class CategoryDecision(CognitiveTask):
    """
    Category decision task: respond based on whether stimulus is above/below threshold
    Pro: respond in category 1 if theta < pi, category 2 if theta > pi
    Anti: opposite
    """

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        # Context
        inputs[:T_context, 0] = 1
        if self.mode == "pro":
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1

        # Stimulus
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)
        if self.mode == "pro":
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1

        # Response
        t_resp_start = t_stim_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        # Determine category (binary: 0 if theta < pi, 1 if theta > pi)
        if self.mode == "pro":
            category = 0 if theta < np.pi else 1
        else:  # anti
            category = 1 if theta < np.pi else 0

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[self.BINARY_CHANNEL] = category

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        inputs[:T_context, 0] = 1
        if self.mode == "pro":
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)
        if self.mode == "pro":
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1

        t_resp_start = t_stim_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        if self.mode == "pro":
            category = 0 if theta < np.pi else 1
        else:
            category = 1 if theta < np.pi else 0

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[self.BINARY_CHANNEL] = category

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim, 1, dtype=np.int64),
                np.full(T_response, 2, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
