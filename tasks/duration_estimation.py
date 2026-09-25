from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DurationEstimation(CognitiveTask):
    """
    Duration Estimation task: judge whether the interval between two flashes
    is short or long.

    Timeline:
    - Context: fixation + rule
    - Flash 1: brief stimulus pulse
    - Delay: variable duration (the thing being measured)
    - Flash 2: brief stimulus pulse
    - Response: was the delay long or short?

    Rules:
    - pro:  output 1 if long, 0 if short
    - anti: output 1 if short, 0 if long

    The delay is sampled uniformly from [min_delay, max_delay] and the
    threshold is the midpoint. The network must develop an internal
    representation of elapsed time.

    Output: decision on last dimension + fixation on dim 0

    Input channels (5): [fixation, flash_cos, flash_sin, rule_pro, rule_anti]

    Tests: temporal estimation + internal clock + rule-conditional mapping

    Reference: Sarafyazd & Bhatt (2019), Hierarchical reasoning by neural
    circuits in the frontal cortex
    """

    PHASE_NAMES = ["context", "flash1", "interval", "flash2", "response"]

    def __init__(
        self,
        duration_params: Dict,
        mode: str = "pro",
        min_delay: int = 10,
        max_delay: int = 20,
        flash_duration: int = 5,
        interval_noise: float = 0.0,
    ):
        super().__init__(duration_params)
        self.mode = mode  # 'pro' or 'anti'
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.flash_duration = flash_duration
        self.threshold = (min_delay + max_delay) / 2
        self.interval_noise = interval_noise  # noise amplitude during interval
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_flash1 = max(1, self.flash_duration + np.random.randint(-2, 3))  # variable ±2
        T_flash2 = max(1, self.flash_duration + np.random.randint(-2, 3))  # variable ±2
        T_interval = np.random.randint(self.min_delay, self.max_delay + 1)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_flash1 + T_interval + T_flash2 + T_response

        # Was the interval long or short?
        is_long = T_interval > self.threshold

        # Determine output based on rule
        if self.mode == "pro":
            should_go = is_long
        else:
            should_go = not is_long

        # Rule channel
        rule_channel = 3 if self.mode == "pro" else 4

        # Sample independent flash directions
        theta_flash1 = np.random.uniform(0, 2 * np.pi)
        theta_flash2 = np.random.uniform(0, 2 * np.pi)

        # Create inputs: [fixation, flash_cos, flash_sin, rule_pro, rule_anti]
        inputs = torch.zeros(T_total, 5)

        # Context period
        inputs[:T_context, 0] = 1
        inputs[:T_context, rule_channel] = 1

        # Flash 1
        t_f1_start = T_context
        t_f1_end = t_f1_start + T_flash1
        inputs[t_f1_start:t_f1_end, 0] = 1
        inputs[t_f1_start:t_f1_end, 1] = np.cos(theta_flash1)
        inputs[t_f1_start:t_f1_end, 2] = np.sin(theta_flash1)
        inputs[t_f1_start:t_f1_end, rule_channel] = 1

        # Interval: fixation + rule + noise on stimulus channels
        t_int_start = t_f1_end
        t_int_end = t_int_start + T_interval
        inputs[t_int_start:t_int_end, 0] = 1
        inputs[t_int_start:t_int_end, rule_channel] = 1
        if self.interval_noise > 0:
            inputs[t_int_start:t_int_end, 1] = (
                torch.randn(T_interval) * self.interval_noise
            )
            inputs[t_int_start:t_int_end, 2] = (
                torch.randn(T_interval) * self.interval_noise
            )

        # Flash 2 (independent direction)
        t_f2_start = t_int_end
        t_f2_end = t_f2_start + T_flash2
        inputs[t_f2_start:t_f2_end, 0] = 1
        inputs[t_f2_start:t_f2_end, 1] = np.cos(theta_flash2)
        inputs[t_f2_start:t_f2_end, 2] = np.sin(theta_flash2)
        inputs[t_f2_start:t_f2_end, rule_channel] = 1

        # Response period: no fixation
        t_resp_start = t_f2_end
        inputs[t_resp_start:, rule_channel] = 1

        # Target: fixation on dim 0, decision on last dim
        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0  # break fixation during response
        if should_go:
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[self.BINARY_CHANNEL] = 0.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_flash1 = max(1, self.flash_duration + np.random.randint(-2, 3))
        T_flash2 = max(1, self.flash_duration + np.random.randint(-2, 3))
        T_interval = np.random.randint(self.min_delay, self.max_delay + 1)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_flash1 + T_interval + T_flash2 + T_response

        is_long = T_interval > self.threshold

        if self.mode == "pro":
            should_go = is_long
        else:
            should_go = not is_long

        rule_channel = 3 if self.mode == "pro" else 4

        theta_flash1 = np.random.uniform(0, 2 * np.pi)
        theta_flash2 = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        inputs[:T_context, 0] = 1
        inputs[:T_context, rule_channel] = 1

        t_f1_start = T_context
        t_f1_end = t_f1_start + T_flash1
        inputs[t_f1_start:t_f1_end, 0] = 1
        inputs[t_f1_start:t_f1_end, 1] = np.cos(theta_flash1)
        inputs[t_f1_start:t_f1_end, 2] = np.sin(theta_flash1)
        inputs[t_f1_start:t_f1_end, rule_channel] = 1

        t_int_start = t_f1_end
        t_int_end = t_int_start + T_interval
        inputs[t_int_start:t_int_end, 0] = 1
        inputs[t_int_start:t_int_end, rule_channel] = 1
        if self.interval_noise > 0:
            inputs[t_int_start:t_int_end, 1] = (
                torch.randn(T_interval) * self.interval_noise
            )
            inputs[t_int_start:t_int_end, 2] = (
                torch.randn(T_interval) * self.interval_noise
            )

        t_f2_start = t_int_end
        t_f2_end = t_f2_start + T_flash2
        inputs[t_f2_start:t_f2_end, 0] = 1
        inputs[t_f2_start:t_f2_end, 1] = np.cos(theta_flash2)
        inputs[t_f2_start:t_f2_end, 2] = np.sin(theta_flash2)
        inputs[t_f2_start:t_f2_end, rule_channel] = 1

        t_resp_start = t_f2_end
        inputs[t_resp_start:, rule_channel] = 1

        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0
        if should_go:
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[self.BINARY_CHANNEL] = 0.0

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_flash1, 1, dtype=np.int64),
                np.full(T_interval, 2, dtype=np.int64),
                np.full(T_flash2, 3, dtype=np.int64),
                np.full(T_response, 4, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
