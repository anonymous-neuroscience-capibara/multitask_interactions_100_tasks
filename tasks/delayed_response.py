from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DelayedResponse(CognitiveTask):
    """
    Delayed Response task: remember stimulus direction and respond after delay
    Pro: respond in same direction as stimulus
    Anti: respond in opposite direction
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode  # 'pro' or 'anti'
        self.loss_channels = [1, 2]  # Only evaluate cosine/sine of response direction

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        # Sample stimulus angle
        theta = np.random.uniform(0, 2 * np.pi)

        # Create inputs: [fixation, stim_cos, stim_sin, rule_pro, rule_anti]
        inputs = torch.zeros(T_total, 5)

        # Context period: fixation + rule
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == "pro":
            inputs[:T_context, 3] = 1  # rule_pro
        else:
            inputs[:T_context, 4] = 1  # rule_anti

        # Stimulus period: fixation + stimulus + rule
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)  # stim_cos
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)  # stim_sin

        if self.mode == "pro":
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1

        # Delay period: fixation + rule
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.mode == "pro":
            inputs[t_delay_start:t_delay_end, 3] = 1
        else:
            inputs[t_delay_start:t_delay_end, 4] = 1

        # Response period: no fixation, rule still on
        t_resp_start = t_delay_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        # Target output: [fixation, response_cos, response_sin]
        targets = torch.zeros(self.output_dim)
        targets[0] = 0  # no fixation during response
        if self.mode == "pro":
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:  # anti
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

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

        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.mode == "pro":
            inputs[t_delay_start:t_delay_end, 3] = 1
        else:
            inputs[t_delay_start:t_delay_end, 4] = 1

        t_resp_start = t_delay_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        if self.mode == "pro":
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim, 1, dtype=np.int64),
                np.full(T_delay, 2, dtype=np.int64),
                np.full(T_response, 3, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
