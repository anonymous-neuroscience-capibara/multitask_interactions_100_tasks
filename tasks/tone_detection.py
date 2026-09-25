from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class ToneDetection(CognitiveTask):
    """
    Tone Detection task: detect a brief tone embedded in noise and report
    its temporal position.

    Timeline:
    - Stimulus: continuous noise on a single channel, with a possible brief
      tone pulse at one of n_positions evenly spaced time points
    - Response: report whether a tone was present and where

    Conditions:
    - No tone: all position outputs = 0
    - Tone at position k: output k = 1, rest = 0

    The tone is a brief pulse (tone_duration timesteps) of amplitude 1.0
    added on top of Gaussian noise.

    Output: fixation on dim 0, n_positions + 1 decision channels on last dims
            [no_tone, pos_1, pos_2, ..., pos_n]

    Input channels (2): [fixation, stimulus]

    Tests: transient detection in noise + temporal localization

    Reference: Young & Sachs (1979), Representation of Tones in Noise
    in the Responses of Auditory Nerve Fibers in Cats
    """

    def __init__(
        self,
        duration_params: Dict,
        sigma: float = 0.2,
        n_positions: int = 3,
        tone_duration: int = 2,
        stim_length: int = 40,
    ):
        super().__init__(duration_params)
        self.sigma = sigma
        self.n_positions = n_positions
        self.tone_duration = tone_duration
        self.stim_length = stim_length
        self.is_argmax_task = True
        self.n_classes = n_positions + 1  # no_tone + n positions
        self.loss_channels = list(range(-self.n_classes, 0))

        # Evenly space tone positions within stimulus period
        spacing = stim_length // (n_positions + 1)
        self.tone_onsets = [spacing * (i + 1) for i in range(n_positions)]

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = self.stim_length
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        # Choose condition: 0 = no tone, 1..n = tone at position k
        condition = np.random.randint(0, self.n_positions + 1)

        # Create inputs: [fixation, stimulus]
        inputs = torch.zeros(T_total, 2)

        # Context period
        inputs[:T_context, 0] = 1

        # Stimulus period: noise on stimulus channel
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        noise = torch.randn(T_stim) * self.sigma
        inputs[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs[t_stim_start:t_stim_end, 1] = noise  # background noise

        # Add tone if present
        if condition > 0:
            tone_onset = t_stim_start + self.tone_onsets[condition - 1]
            tone_offset = tone_onset + self.tone_duration
            inputs[tone_onset:tone_offset, 1] += 1.0  # tone pulse on top of noise

        # Response period: no fixation
        t_resp_start = t_stim_end

        # Target: one-hot on last n_classes dims [no_tone, pos_1, ..., pos_n]
        targets = torch.zeros(self.output_dim)
        targets[-self.n_classes + condition] = 1.0

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
