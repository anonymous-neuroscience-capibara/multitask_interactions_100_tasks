from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class PerceptualDecisionMaking(CognitiveTask):
    """
    Perceptual Decision Making (Random Dots) task:
    Integrate noisy directional evidence over time to report net direction.

    Timeline:
    - Context: fixation
    - Stimulus: noisy directional evidence at varying coherence
    - Response: report integrated direction

    At each stimulus timestep, input = coherence * true_direction + noise.
    Higher coherence → easier trial. Network must temporally integrate
    to extract signal from noise.

    Based on: Yang et al. 2019 — random-dot motion discrimination

    Input channels (4): [fixation, evidence_cos, evidence_sin]
    """

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, coherence_range=(0.5, 0.75)):
        super().__init__(duration_params)
        self.coherence_range = coherence_range
        self.loss_channels = [1, 2]  # Only evaluate cosine/sine of response direction

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        # True motion direction
        theta_true = np.random.uniform(0, 2 * np.pi)

        # Trial coherence (difficulty)
        coherence = np.random.uniform(*self.coherence_range)

        # Noise level scales inversely with coherence
        noise_std = 0.5 * (1 - coherence)

        # Build inputs: [fixation, evidence_cos, evidence_sin]
        inputs_np = np.zeros((T_total, 3))

        # Context period
        inputs_np[:T_context, 0] = 1  # fixation

        # Stimulus period: noisy evidence each timestep
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim

        noise_cos = np.random.normal(0, noise_std, size=T_stim)
        noise_sin = np.random.normal(0, noise_std, size=T_stim)

        inputs_np[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs_np[t_stim_start:t_stim_end, 1] = (
            coherence * np.cos(theta_true) + noise_cos
        )
        inputs_np[t_stim_start:t_stim_end, 2] = (
            coherence * np.sin(theta_true) + noise_sin
        )

        # Response period: no fixation
        t_resp_start = t_stim_end

        inputs = torch.from_numpy(inputs_np).float()

        # Target: true motion direction
        targets = torch.zeros(self.output_dim)
        targets[0] = 0  # no fixation
        targets[1] = np.cos(theta_true)
        targets[2] = np.sin(theta_true)

        # Mask: evaluate at first response timestep
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta_true = np.random.uniform(0, 2 * np.pi)
        coherence = np.random.uniform(*self.coherence_range)
        noise_std = 0.5 * (1 - coherence)

        inputs_np = np.zeros((T_total, 3))

        inputs_np[:T_context, 0] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim

        noise_cos = np.random.normal(0, noise_std, size=T_stim)
        noise_sin = np.random.normal(0, noise_std, size=T_stim)

        inputs_np[t_stim_start:t_stim_end, 0] = 1
        inputs_np[t_stim_start:t_stim_end, 1] = (
            coherence * np.cos(theta_true) + noise_cos
        )
        inputs_np[t_stim_start:t_stim_end, 2] = (
            coherence * np.sin(theta_true) + noise_sin
        )

        t_resp_start = t_stim_end

        inputs = torch.from_numpy(inputs_np).float()

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_true)
        targets[2] = np.sin(theta_true)

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
