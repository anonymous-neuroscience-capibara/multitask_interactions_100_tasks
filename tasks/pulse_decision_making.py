from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class PulseDecisionMaking(CognitiveTask):
    """
    Pulse-based Decision Making task: count discrete pulses and compare.

    Timeline:
    - Context: fixation only
    - Evidence bins: n_bins periods, each with a brief pulse window followed
      by a silent gap. Each channel independently fires (1) or not (0).
    - Response: which channel received more pulses?

    One channel has higher pulse probability (p_high), the other lower (p_low).
    Which channel is favored is randomized per trial.

    Output: fixation on dim 0, decision on last dimension
            (1.0 = channel 1 had more, 0.0 = channel 2 had more)

    Input channels (3): [fixation, pulse_ch1, pulse_ch2]

    Tests: discrete event detection + counting/accumulation + comparison

    Reference: Brunton et al. (2013), Sources of noise during accumulation
    of evidence in unrestrained and voluntarily head-restrained rats
    """

    def __init__(
        self,
        duration_params: Dict,
        n_bins: int = 6,
        pulse_duration: int = 1,
        gap_duration: int = 5,
        p_high: float = 0.7,
        p_low: float = 0.3,
    ):
        super().__init__(duration_params)
        self.n_bins = n_bins
        self.pulse_duration = pulse_duration
        self.gap_duration = gap_duration
        self.p_high = p_high
        self.p_low = p_low
        self.is_binary_task = True
        self.loss_channels = [self.BINARY_CHANNEL]

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_evidence = self.n_bins * (self.pulse_duration + self.gap_duration)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_evidence + T_response

        # Randomly assign which channel is favored
        if np.random.rand() < 0.5:
            p1, p2 = self.p_high, self.p_low
        else:
            p1, p2 = self.p_low, self.p_high

        # Generate pulses for each bin
        pulses1 = (np.random.rand(self.n_bins) < p1).astype(float)
        pulses2 = (np.random.rand(self.n_bins) < p2).astype(float)

        # Determine ground truth from actual pulse counts
        count1 = pulses1.sum()
        count2 = pulses2.sum()

        # Handle ties by adding small random jitter (same as neurogym)
        count2_jittered = count2 + np.random.uniform(-0.1, 0.1)
        ch1_more = count1 > count2_jittered

        # Create inputs: [fixation, pulse_ch1, pulse_ch2]
        inputs = torch.zeros(T_total, 3)

        # Context period
        inputs[:T_context, 0] = 1

        # Evidence bins: pulse + gap for each bin
        t_bin_start = T_context
        for i in range(self.n_bins):
            # Pulse window
            t_pulse_start = t_bin_start + i * (self.pulse_duration + self.gap_duration)
            t_pulse_end = t_pulse_start + self.pulse_duration

            inputs[t_pulse_start:t_pulse_end, 0] = 1  # fixation
            inputs[t_pulse_start:t_pulse_end, 1] = pulses1[i]  # ch1
            inputs[t_pulse_start:t_pulse_end, 2] = pulses2[i]  # ch2

            # Gap (silence)
            t_gap_start = t_pulse_end
            t_gap_end = t_gap_start + self.gap_duration
            inputs[t_gap_start:t_gap_end, 0] = 1  # fixation

        # Response period: no fixation
        t_resp_start = T_context + T_evidence

        # Target: decision on last dim
        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if ch1_more else 0.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
